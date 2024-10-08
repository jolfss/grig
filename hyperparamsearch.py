
from typing import Optional
import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from cluster import cluster, get_colors, plot_elbow_graph
import itertools
import os

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

REMOVE_BACKGROUND = True  # False or True
# REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

#---------------------------------#
#   NOTE: Original source code.   #
#---------------------------------#
def __load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

#---------------------------------#
#   NOTE: Custom-color testing.   #
#---------------------------------#
def load_scene_data(seq, exp, seg_as_col=False, use_elbow=True):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    
    optimal_k = 10 # User has to specify this if use_elbow is False

    # Pass the elbow flag here
    if use_elbow:
        optimal_k = plot_elbow_graph(seq, exp, max_clusters=30, stride = 3)

    colors = get_colors(seq, exp, num_clusters=optimal_k)

    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': colors,
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def search(filepath, K:int=26):
    for POS, DPOS, DROT in itertools.product(range(1, 6), repeat=3):
        params = {
           "timestride": 5,
           "POS": np.exp(POS - 3),
           "DPOS": np.exp(DPOS - 3),
           "DROT": np.exp(DROT - 3)
        }
        scene_data, _, _, cluster_centers  = cluster(filepath, **params)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(320 * view_scale), height=int(h * view_scale), visible=True)

        def init_camera(y_angle=0., center_dist=3, cam_height=0.75, f_ratio=0.9):
            ry = y_angle * np.pi / 180
            w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                            [0.,         1., 0.,          cam_height],
                            [np.sin(ry), 0., np.cos(ry),  center_dist],
                            [0.,         0., 0.,          1.]])
            k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
            return w2c, k
        
        w2c, k = init_camera()
        im, depth = render(w2c, k, scene_data[0])
        init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols
        vis.add_geometry(pcd)
        
        spheres = []
        for _ in range(K):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Adjust the radius as needed
            sphere.paint_uniform_color([1, 1, 1])  # Color the sphere red
            spheres.append(sphere)
            vis.add_geometry(sphere)

        view_k = k * view_scale
        view_k[2, 2] = 1
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(h * view_scale)
        cparams.intrinsic.width = int(w * view_scale)
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

        render_options = vis.get_render_option()
        render_options.point_size = view_scale
        render_options.light_on = False

        start_time = time.time()
        num_timesteps = len(scene_data)

        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        t = int(passed_frames % num_timesteps)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        im, depth = render(w2c, k, scene_data[t])
        pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
        pcd.points = pts  
        pcd.colors = cols
        vis.update_geometry(pcd)

        centers = torch.cat([cluster_centers[t],torch.ones((K,1))],dim=-1) # (K,4)
        centers = centers.numpy() @ w2c.T
        centers[:,:3] *= 0.25
        centers = centers @ np.linalg.inv(w2c).T
        
        for c in range(K):
            spheres[c].translate(centers[c,:3], relative=False)
            vis.update_geometry(spheres[c])

        if not vis.poll_events():
            break
        vis.update_renderer()

        output_folder = 'hpsearch'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        vis.capture_screen_image(F"hpsearch/POS={POS}_DPOS={DPOS}_DROT={DROT}.png")

        vis.destroy_window()
        del view_control
        del vis
        del render_options

if __name__ == "__main__":
    search("./output/pretrained/softball/params.npz")
