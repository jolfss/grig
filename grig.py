#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
#og imports
import time
import open3d as o3d
from visualize import init_camera, render, rgbd2pcd 
import torch
import numpy as np

# our imports
from dataclasses import dataclass
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

#-----------#
#   types   #
#-----------#

@dataclass
class Parameters:
    """
    The inputs necessary to visualize a Dynamic 3D Gaussian.
        `T`: timestep
        `N`: gaussians
    """

    means3D:torch.Tensor
    """The centers (x,y,z) of the gaussians; y is vertical.\n
    `(T,N,3)@cuda float`"""

    unnorm_rotations : torch.Tensor 
    """The unnormalized quaternions (qx,qy,qz,qw)? representation of the gaussians' rotations.
    `(T,N,4)@cuda float`"""

    log_scales : torch.Tensor 
    """The log scales/multivariances of the gaussians (sx, sy, sz)?\n
    `(N,3)@cuda float`"""

    rgb_colors : torch.Tensor 
    """The colors of the gaussians; (-inf,inf) but intended range [0,1].\n
    (<0 black, 1 is peak intensity, >1 limits to solid ellipse of peak intensity)\n
    `(N,3)@cuda float`"""

    seg_colors : torch.Tensor
    """The segmentation color of the gaussians; used to segment foreground and background natively.
    `(N,3)@cuda float`"""
    
    logit_opacities : torch.Tensor
    """The logits representing the opacities of the gaussians.\n
    `(N)@cuda float`""" 

@dataclass
class GrigConfig:
    K:int
    "The number of means to cluster over."

    timestride:int
    "How many time frames should be skipped over."

    POS:float
    "The weighting of the `POS` feature."

    DPOS:float
    "The weighting of the `DPOS` feature."

    DROT:float
    "The weighting of the `DROT` feature."

    remove_bg:bool
    "Whether to render the background gaussians (with their default rbg colors)."

    normalize_features:bool
    "Whether each feature component should be normalized.."

    color_mode:str
    """Which color mode to use for rendering.
    Options: CLUSTER POS DPOS ROT DROT"""


class Features:
    """
    Representation of the features extracted from the parameter file.
    """
    features : torch.Tensor
    """The output of some function on the input parameters.\n
    `(N,D*T//config.timestride)@cuda float`"""
    
    is_fg : torch.Tensor
    """A boolean mask along gaussians specifying which are foreground and background.\n
    `(N,)@cuda bool`"""
    
    pos : torch.Tensor
    """The centers of the gaussians (x,y,z).
    `(T,N,3)@cuda float`"""
    
    rot : torch.Tensor
    """The orientations of the gaussians as normalized quaternions (qx,qy,qz,qw)\n
    TODO: Have not fully considered the implications of the double cover of `SO(3)`.\n
    `(T,N,4)@cuda float`"""

    dpos_dt : torch.Tensor
    """The time derivative `(/s)` of `pos`.\n
    `(T,N,3)@cuda float`"""
    
    drot_dt : torch.Tensor
    """The time derivative `(/s)` of `rot`.\n
    `(T,N,4)@cuda float`"""
    
    T : int
    "The number of timesteps."
    N : int
    "The number of gaussians."
    D : int
    "The size of the feature dimension (at each timestep)."

    def __init__(self, params : Parameters, config : GrigConfig):
        # initialize  
        self.pos = params.means3D
        self.rot = torch.nn.functional.normalize(params.unnorm_rotations, dim=-1)
        self.dpos_dt = torch.gradient(self.pos, spacing=30*config.timestride, dim=0)[0] # NOTE: Spacing is adjustment for 30fps input
        self.drot_dt = torch.gradient(self.rot, spacing=30*config.timestride, dim=0)[0] 

        # size checks
        T_pos, N_pos, D_pos = self.pos.size()
        T_posdt, N_posdt, D_posdt = self.dpos_dt.size()
        T_rot, N_rot, D_rot = self.rot.size()
        T_rotdt, N_rotdt, D_rotdt = self.drot_dt.size()
        assert T_pos == T_posdt == T_rot == T_rotdt >= 2
        assert N_pos == N_posdt == N_rot == N_rotdt >= 2
        assert D_pos == D_posdt == 3
        assert D_rot == D_rotdt == 4
        self.N = N_pos
        self.T = T_pos
        self.D = self.pos.size(-1) + self.dpos_dt.size(-1) + self.drot_dt.size(-1) # TODO: adjust for feature ablation/mod/aug

        # prepare raw feature vector 
        self.features = torch.cat((
            config.POS*self.pos[::config.timestride], 
            config.DPOS*self.dpos_dt[::config.timestride], 
            config.DROT*self.drot_dt[::config.timestride]), dim=-1).permute(1, 0, 2).reshape((self.N, -1))  # -1 = feature_dim*T//timestride

        # foreground mask
        self.is_fg=params.seg_colors[:,0] > 0.5,

    def __getattr__(self, name):
        """Called when the requested attribute or method isn't found in the object."""
        return getattr(self.features, name)
    
    def __getitem__(self, key):
        """Forward item access (slicing or indexing)."""
        return self.features[key]
    
    def __setitem__(self, key, value):
        """Forward item assignment."""
        self.features[key] = value
    
    def __delitem__(self, key):
        """Forward item deletion."""
        del self.features[key]


#----------------------#
#   global constants   #
#----------------------#
# camera configuration
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20

# clustering config
default_config = GrigConfig(
    timestride=1,
    POS=1,
    DPOS=1,
    DROT=1,
    K=24,
    remove_bg=True,
    normalize_features=True,
    color_mode="CLUSTERS"
)

#-------------#
#   methods   #
#-------------#
@torch.no_grad()
def grig(filepath_npz:str, config:GrigConfig) -> Tuple[Dict[str, torch.Tensor], Features, torch.Tensor, torch.Tensor]:
    """
    The clustering algorithm `(param_filepath, config) -> (scene_data, features, labels, centers)`. 
    """
    
    print(F"Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    params = {key : params[key] for key in Parameters.__dataclass_fields__ if key in params}
    params = Parameters(**params)

    print("Preparing Foreground Features")
    features = Features(params, config)

    fg_features = features[features.is_fg]
    fg_features = fg_features.cpu().numpy()

    if config.normalize_features:
        print("Normalizing Foreground Features")
        scaler = StandardScaler()
        fg_features = scaler.fit_transform(fg_features)

    print(F"Clustering Foreground Gaussians")
    kmeans = KMeans(n_clusters=config.K)
    kmeans.fit(fg_features)
    fg_labels = kmeans.labels_

    print(F"Coloring Gaussians (Mode={config.color_mode})")
    cmap = cm.get_cmap('turbo', config.K)
    feature_colors = None
    if config.color_mode == "POS":
        feature_colors = torch.abs(features.pos[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "DPOS":
        feature_colors = torch.abs(features.dpos_dt[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "DROT":
        feature_colors = torch.abs(features.drot_dt[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "CLUSTERS":
        cluster_colors = cmap(fg_labels)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((150,-1,3))
    else: #RGB
        pass

    colors = params.rgb_colors #NOTE: color is constant w.r.t. time
    if feature_colors is not None:
        colors[:,features.is_fg] = feature_colors 

    print(F"Calculating Centers")
    cluster_labels = torch.zeros((features.N, 1), device="cuda").long() - 1
    cluster_labels[features.is_fg] = torch.from_numpy(fg_labels).long().to("cuda").unsqueeze(-1)
    cluster_centers = torch.zeros((features.T, config.K, 3))  # T time steps, K clusters, 3 dimensions for position
    
    for c in range(config.K):
        mask = (cluster_labels==c)
        cluster_centers[:, c, :] = (features.pos * mask).sum(dim=1) / mask.sum() # NOTE: Assumes no empty clusters
    
    print(F"Preparing Rendervars")
    scene_data = []
    for t in range(len(features.pos)):
        rendervar = {
            'means3D': features.pos[t],
            'colors_precomp': colors[t], 
            'rotations': features.rot[t],
            'opacities': torch.sigmoid(params.logit_opacities),
            'scales': torch.exp(params.log_scales),
            'means2D': torch.zeros_like(params.means3D[0], device="cuda")
        }
        if config.remove_bg:
            rendervar = {k: v[features.is_fg] for k, v in rendervar.items()}  
        scene_data.append(rendervar)

    return scene_data, features, cluster_labels, cluster_centers
    
def visualize(filepath_npz:str, config:GrigConfig):
    scene_data, _, _, cluster_centers  = grig(filepath_npz, config)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    w2c, k = init_camera()
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)
    
    cluster_centers = []
    for _ in range(config.K):
        cluster_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Adjust the radius as needed
        cluster_center.paint_uniform_color([1, 1, 1])  # Color the sphere red
        cluster_centers.append(cluster_center)
        vis.add_geometry(cluster_center)

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
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        t = int(passed_frames % num_timesteps)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        # render gaussian splat
        im, depth = render(w2c, k, scene_data[t])
        pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
        pcd.points = pts  
        pcd.colors = cols
        vis.update_geometry(pcd)

        #  centers visualization 
        centers = torch.cat([cluster_centers[t],torch.ones((config.K,1))],dim=-1) # (K,4)
        centers = centers.numpy() @ w2c.T
        centers[:,:3] *= 0.25
        centers = centers @ np.linalg.inv(w2c).T
        
        # update the position of each sphere based on the camera coordinates
        for c in range(config.K):
            cluster_centers[c].translate(centers[c,:3], relative=False)

            # Update each sphere individually in the visualizer
            vis.update_geometry(cluster_centers[c])

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options

import sys
if __name__ == "__main__":
    default_config.color_mode = sys.argv[2]
    visualize(F"./output/pretrained/{sys.argv[1]}/params.npz", default_config)
