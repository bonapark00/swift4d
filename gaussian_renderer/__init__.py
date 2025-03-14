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
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, bwd_dynamic = False, dyn_mask2= None, 
             override_color = None, stage="coarse", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix = viewpoint_camera.world_view_transform.cuda(),
            projmatrix = viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            bwd_depth= False,
            bwd_dynamic = bwd_dynamic,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    dynamics = pc.get_dynamic
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation





    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:

        dyn_mask= ( pc.get_dynamic > pc.dynamic_thro ).squeeze(1)
        means3D_final = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, device="cuda")
        scales_final = torch.zeros_like(scales, dtype=pc.get_xyz.dtype, device="cuda")
        rotations_final = torch.zeros_like(rotations, dtype=pc.get_xyz.dtype, device="cuda")
        opacity_final = torch.zeros_like(opacity, dtype=pc.get_xyz.dtype, device="cuda")
        shs_final = torch.zeros_like(shs, dtype=pc.get_xyz.dtype, device="cuda")

        means3D_final[dyn_mask], scales_final[dyn_mask], rotations_final[dyn_mask], opacity_final[dyn_mask], shs_final[dyn_mask] \
            = pc._FDhash(means3D[dyn_mask], scales[dyn_mask], rotations[dyn_mask], opacity[dyn_mask], shs[dyn_mask], time[dyn_mask])
            
        means3D_final[~dyn_mask], scales_final[~dyn_mask], rotations_final[~dyn_mask], opacity_final[~dyn_mask], \
                    shs_final[~dyn_mask] = means3D[~dyn_mask], scales[~dyn_mask], rotations[~dyn_mask], opacity[~dyn_mask], shs[~dyn_mask]
    
    
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)  # exp
    rotations_final = pc.rotation_activation(rotations_final)  #  normalize
    opacity = pc.opacity_activation(opacity_final)  # sigmoid
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    if  dyn_mask2 !=None:  # our test code , not important
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth, dynamics_map, max_weight = rasterizer(
            means3D = means3D_final[dyn_mask2],
            means2D = means2D[dyn_mask2],
            shs = shs_final[dyn_mask2],
            colors_precomp = colors_precomp,
            opacities = opacity[dyn_mask2],
            scales = scales_final[dyn_mask2],
            rotations = rotations_final[dyn_mask2],
            dynamics = dynamics[dyn_mask2],
            cov3D_precomp = cov3D_precomp
            )
    else:
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth, dynamics_map, max_weight = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            dynamics = dynamics,
            cov3D_precomp = cov3D_precomp
        )
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "dynamic_map": dynamics_map,
            "max_weight":max_weight
            }


