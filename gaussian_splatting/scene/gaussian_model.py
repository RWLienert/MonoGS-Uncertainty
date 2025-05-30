#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, logging
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import torch, torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p

# ───────── Fisher-EMA Global Configuration ───────── #
#   Specifies (epsilon, initial variance ceiling) for each parameter type
FISHER_CFG = {
    "xyz":      (1e-4, 1e2),
    "scaling":  (1e-4, 1e2),
    "rotation": (1e-4, 1e2),
    "f_dc":     (1e-4, 5e2),
    "opacity":  (1e-5, 30.0),
    "f_rest":   (5e-6, 8000.0),
}
DEFAULT_EPS, DEFAULT_CEIL = 1e-4, 1e2

class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.start = True

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        # Covariance and Fisher information buffers
        self._xyz_cov = self._scaling_cov = self._rotation_cov = torch.empty(0)
        self._features_dc_cov = self._features_rest_cov = torch.empty(0)
        self._opacity_cov = torch.empty(0)

        self._xyz_fisher_buf = self._scaling_fisher_buf = self._rotation_fisher_buf = torch.empty(0)
        self._f_dc_fisher_buf = self._f_rest_fisher_buf = torch.empty(0)
        self._opacity_fisher_buf = torch.empty(0)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False
    
    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    @torch.no_grad()
    def update_covariance(self,
                          grads_dict: Dict[str, torch.Tensor], *,
                          cur_iter: int,
                          max_iter: int,
                          loss_scalar: Optional[float] = None,
                          var_floor: float = 1e-6,
                          max_pts_per_group: int = 300_000):

        """
        Update diagonal covariance via Fisher-EMA using parameter gradients.
        Skips update if loss change <2% to save computation.
        """   
        
        # Skip if loss change is too small
        if loss_scalar is not None and hasattr(self, '_prev_loss_scalar'):
            if abs(loss_scalar - self._prev_loss_scalar) < 0.02 * self._prev_loss_scalar:
                return
        if loss_scalar is not None:
            self._prev_loss_scalar = loss_scalar

        # Compute dynamic alpha from training progress
        prog = cur_iter / max_iter
        alpha_dyn = 0.9 * (1.0 - 0.5 * prog)
        if cur_iter % 2000 == 0:
            logging.info(f"[Fisher-EMA] iteration {cur_iter}: alpha={alpha_dyn:.3f}")

        # Update each parameter group
        for pname, grad in grads_dict.items():
            if grad is None or grad.numel() == 0:
                continue
            eps, var_ceil0 = FISHER_CFG.get(pname, (DEFAULT_EPS, DEFAULT_CEIL))
            var_ceil_dyn = max(var_floor * 10.0, var_ceil0 * (1.0 - 0.5 * prog))

            N, D = grad.shape
            buf_name = {
                'xyz':      '_xyz_fisher_buf',
                'scaling':  '_scaling_fisher_buf',
                'rotation': '_rotation_fisher_buf',
                'opacity':  '_opacity_fisher_buf',
                'f_dc':     '_f_dc_fisher_buf',
                'f_rest':   '_f_rest_fisher_buf',
            }[pname]

            fisher_buf = getattr(self, buf_name)
            if fisher_buf.numel() == 0:
                fisher_buf = torch.zeros_like(grad)

            # Block-wise EMA update for memory efficiency
            for s in range(0, N, max_pts_per_group):
                e = min(s + max_pts_per_group, N)
                chunk = fisher_buf[s:e]
                g2 = grad[s:e].square()
                chunk.mul_(alpha_dyn).add_(g2, alpha=1.0 - alpha_dyn)
            setattr(self, buf_name, fisher_buf)

            # Fisher → Σ
            var = torch.reciprocal(fisher_buf + eps)
            var.clamp_(var_floor, var_ceil_dyn)

            cov_name = {
                "xyz":      "_xyz_cov",
                "scaling":  "_scaling_cov",
                "rotation": "_rotation_cov",
                "opacity":  "_opacity_cov",
                "f_dc":     "_features_dc_cov",
                "f_rest":   "_features_rest_cov",
            }[pname]

            if pname in ("f_rest", "opacity"):
                setattr(self, cov_name, var)
            else:
                eye = torch.eye(D, device=var.device).unsqueeze(0)
                setattr(self, cov_name, var.unsqueeze(-1) * eye)

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        if self.start == True:
            N = fused_point_cloud.shape[0]
            
            self._xyz_cov = (
                dist2[:, None, None] * torch.eye(3, device="cuda")[None, :, :]
            ).detach()

            color_var = torch.var(fused_color, dim=0, keepdim=True)
            self._features_dc_cov = (
                color_var * torch.eye(3, device="cuda")[None, :, :]
            ).expand(N, 3, 3).detach()

            C = features[:, :, 1:].shape[1]  # Should be 3 (channels)
            D_ = features[:, :, 1:].shape[2]  # SH Coeff count (excluding DC term)
            D_total = C * D_
            self._features_rest_cov = torch.full((N, D_total), 0.1, device="cuda").detach()

            self._scaling_cov = (
                dist2[:, None, None] * 0.1 * torch.eye(3, device="cuda")[None, :, :]
            ).detach()

            self._rotation_cov = (
                0.1 * torch.eye(4, device="cuda")[None, :, :].repeat(N, 1, 1)
            ).detach()

            op_var = torch.var(opacities, dim=0, keepdim=True)
            self._opacity_cov = (op_var * 0.1 * torch.ones_like(opacities)).detach()

            self._xyz_fisher_buf = torch.zeros((N, 3), device="cuda")
            self._f_dc_fisher_buf = torch.zeros((N, 3), device="cuda")
            self._scaling_fisher_buf = torch.zeros((N, 3), device="cuda")
            self._rotation_fisher_buf = torch.zeros((N, 4), device="cuda")
            self._opacity_fisher_buf = torch.zeros((N, 1), device="cuda")
            self._f_rest_fisher_buf = torch.zeros((N, D_total), device="cuda")

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr
    
    def get_covariance_dict(self, batch_size=5000):
        return {
            "xyz":      self._xyz_cov,
            "opacity":  self._opacity_cov,
            "scaling":  self._scaling_cov,
            "rotation": self._rotation_cov,
            "f_dc":     self._features_dc_cov,
            "f_rest":   self._features_rest_cov
        }

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        n_points = self._xyz.shape[0]
        if mask.shape[0] != n_points:
            logging.warning(f"[prune_points] shape mismatch: param pts={n_points}, mask={mask.shape[0]}")
            mask = mask[:n_points]
            
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        oldN = self._xyz_cov.shape[0]
        newN = valid_points_mask.sum().item()
        
        if newN < oldN:
            # --- 1) Cov Clip ---
            self._xyz_cov          = self._xyz_cov[valid_points_mask]
            self._opacity_cov      = self._opacity_cov[valid_points_mask]
            self._scaling_cov      = self._scaling_cov[valid_points_mask]
            self._rotation_cov     = self._rotation_cov[valid_points_mask]
            self._features_dc_cov  = self._features_dc_cov[valid_points_mask]
            self._features_rest_cov= self._features_rest_cov[valid_points_mask]

            # --- 2) Fisher buf Clip ---
            self._xyz_fisher_buf      = self._xyz_fisher_buf[valid_points_mask]
            self._opacity_fisher_buf  = self._opacity_fisher_buf[valid_points_mask]
            self._scaling_fisher_buf  = self._scaling_fisher_buf[valid_points_mask]
            self._rotation_fisher_buf = self._rotation_fisher_buf[valid_points_mask]
            self._f_dc_fisher_buf     = self._f_dc_fisher_buf[valid_points_mask]
            self._f_rest_fisher_buf   = self._f_rest_fisher_buf[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_covariances(self, xyz_count, new_count):
        device = self._xyz_cov.device

        # ----- Cov expansions -----
        new_xyz_cov = 0.1 * torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._xyz_cov = torch.cat((self._xyz_cov, new_xyz_cov), dim=0)

        new_fdc_cov = 0.1 * torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._features_dc_cov = torch.cat((self._features_dc_cov, new_fdc_cov), dim=0)

        D_ = self._features_rest_cov.shape[1]
        new_frest_cov = 0.1 * torch.ones((new_count, D_), device=device)
        self._features_rest_cov = torch.cat((self._features_rest_cov, new_frest_cov), dim=0)

        new_scaling_cov = 0.1 * torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._scaling_cov = torch.cat((self._scaling_cov, new_scaling_cov), dim=0)

        new_rotation_cov = 0.1 * torch.eye(4, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._rotation_cov = torch.cat((self._rotation_cov, new_rotation_cov), dim=0)

        new_opacity_cov = 0.1 * torch.ones((new_count, 1), device=device)
        self._opacity_cov = torch.cat((self._opacity_cov, new_opacity_cov), dim=0)

        new_xyz_fish = torch.zeros((new_count, 3), device=device)
        self._xyz_fisher_buf = torch.cat((self._xyz_fisher_buf, new_xyz_fish), dim=0)

        new_fdc_fish = torch.zeros((new_count, 3), device=device)
        self._f_dc_fisher_buf = torch.cat((self._f_dc_fisher_buf, new_fdc_fish), dim=0)

        new_frest_fish = torch.zeros((new_count, D_), device=device)
        self._f_rest_fisher_buf = torch.cat((self._f_rest_fisher_buf, new_frest_fish), dim=0)

        new_scaling_fish = torch.zeros((new_count, 3), device=device)
        self._scaling_fisher_buf = torch.cat((self._scaling_fisher_buf, new_scaling_fish), dim=0)

        new_rotation_fish = torch.zeros((new_count, 4), device=device)
        self._rotation_fisher_buf = torch.cat((self._rotation_fisher_buf, new_rotation_fish), dim=0)

        new_opacity_fish = torch.zeros((new_count, 1), device=device)
        self._opacity_fisher_buf = torch.cat((self._opacity_fisher_buf, new_opacity_fish), dim=0)

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        old_count = self.get_xyz.shape[0]
        new_count = new_xyz.shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.start == False:
            # 2) Update covariance
            self.cat_covariances(old_count, new_count)
        
        self.start = False

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        
        # clamp
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        if grads.numel()>0:
            padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.zeros(n_init_points,dtype=torch.bool, device="cuda")
        if grads.numel()>0:
            selected_pts_mask = (torch.norm(grads,dim=-1)>=grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,dim=1).values<=self.percent_dense*scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)
