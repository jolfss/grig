#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
#og imports
import time
import open3d as o3d
from visualize import init_camera, render, rgbd2pcd 
import torch
import numpy as np

# our imports
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R  # To handle quaternion-based rotations

# types
from parameters import Parameters
from config import Config, BaseConfig, KMeansConfig
from features import Features
from clustering import Clustering

###################################################################################################
##   NOTE:                                                                                       ##
##   1) There is a useful method [helpers.o3d_knn] for if we want to do shearing calculations.   ##
###################################################################################################

#----------------------#
#   global constants   #
#----------------------#
# camera configuration
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20

#-------------#
#   methods   #
#-------------#
@torch.no_grad()
def grig(filepath_npz:str, config:Config) -> Tuple[Dict[str, torch.Tensor],Clustering]:
    """
    The clustering algorithm `(param_filepath, config) -> (scene_data, clustering)`. 
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

    clustering = Clustering(
        config.K, 
        features, 
        torch.from_numpy(fg_labels).long().to("cuda")
    )

    # print(F"Calculating Centers")
    # cluster_labels = torch.zeros((features.N, 1), device="cuda").long() - 1
    # cluster_labels[features.is_fg] = torch.from_numpy(fg_labels).long().to("cuda").unsqueeze(-1)
    # cluster_centers = torch.zeros((features.T, config.K, 3))  # T time steps, K clusters, 3 dimensions for position
    # cluster_label_masks = [] 
    # for c in range(config.K):
    #     mask = (cluster_labels==c)
    #     cluster_label_masks.append(mask)
    #     cluster_centers[:, c, :] = (features.pos * mask).sum(dim=1) / mask.sum() # NOTE: Assumes no empty clusters
    
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

    return scene_data, clustering


def initialize_cluster_center_dots(clustering, vis) -> List[o3d.geometry.TriangleMesh]:
    """Initializes the memory for the center dots visualization."""
    cluster_center_dots = []
    for _ in range(clustering.num_clusters):
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.005) 
        center.paint_uniform_color([1, 1, 1])   
        cluster_center_dots.append(center)
        vis.add_geometry(center)
    return cluster_center_dots

def update_cluster_centers(t:int, clustering:Clustering, w2c:np.ndarray, cluster_center_dots:List[o3d.geometry.TriangleMesh], vis:o3d.visualization.Visualizer):
    """Updates the center dots visualization to timestep `t`."""
    centers_t = clustering.centers[t]
    
    # NOTE: we project closer to the camera to be visible as an overlay
    __new_cluster_center_points = torch.cat([centers_t,torch.ones((clustering.num_clusters,1))],dim=-1) # (K,4)
    __new_cluster_center_points = __new_cluster_center_points.numpy() @ w2c.T
    __new_cluster_center_points[:,:3] *= 0.25
    __new_cluster_center_points = __new_cluster_center_points @ np.linalg.inv(w2c).T
    
    for c in range(clustering.num_clusters):
        cluster_center_dots[c].translate(__new_cluster_center_points[c,:3], relative=False)
        vis.update_geometry(cluster_center_dots[c])

def initialize_xform_lineset(clustering:Clustering, vis:o3d.visualization.Visualizer) -> o3d.geometry.LineSet:
    """Initializes the lineset for the xform visualization."""
    lineset = o3d.geometry.LineSet()
    cluster_center_xform_points = np.zeros((4*clustering.num_clusters,3)) # 4 points (center,->x,->y,->z) for each cluster
    cluster_center_xform_lines =  np.zeros((3*clustering.num_clusters,2)) # connect center to each unit vector
    cluster_center_xform_colors = np.zeros((3*clustering.num_clusters,3)) # r,g,b axis colorings
    for c in range(config.K):
        cluster_center_xform_lines[3*c]   = [4*c, 4*c+1]
        cluster_center_xform_lines[3*c+1] = [4*c, 4*c+2]
        cluster_center_xform_lines[3*c+2] = [4*c, 4*c+3]
        cluster_center_xform_colors[3*c  ]  = [1,0,0]
        cluster_center_xform_colors[3*c+1]  = [0,1,0]
        cluster_center_xform_colors[3*c+2]  = [0,0,1]
    lineset.points = o3d.utility.Vector3dVector(cluster_center_xform_points)
    lineset.lines = o3d.utility.Vector2iVector(cluster_center_xform_lines)
    lineset.colors = o3d.utility.Vector3dVector(cluster_center_xform_colors)
    vis.add_geometry(lineset)
    return lineset

#@GPT
def update_lineset(t:int, clustering:Clustering, w2c:np.ndarray, xform_lineset:o3d.geometry.LineSet, vis:o3d.visualization.Visualizer):
    """Updates the xform lineset visualization to timestep `t`."""
    centers_t = clustering.centers[t]  # Assuming you have access to this
    xforms_t = clustering.transformations[t]  # Get transformations for this time step

    new_cluster_center_xform_points = np.zeros((4 * clustering.num_clusters, 3))  # 4 points per cluster

    for c in range(clustering.num_clusters):
        # Extract the origin (translation) for cluster `c`
        origin = centers_t[c].cpu().numpy()

        # Extract the rotation as a quaternion [qw, qx, qy, qz]
        quaternion = xforms_t[c, 3:7].cpu().numpy()  # Extract [qw, qx, qy, qz]
        
        # Create a Rotation object from the quaternion (note the order for SciPy)
        rotation = R.from_quat(quaternion[[1, 2, 3, 0]])  # [qx, qy, qz, qw]
        
        # Define unit vectors for local X, Y, Z axes in the cluster's local frame
        local_x = np.array([0.2, 0, 0])  # X-axis unit vector scaled for visibility
        local_y = np.array([0, 0.2, 0])  # Y-axis unit vector
        local_z = np.array([0, 0, 0.2])  # Z-axis unit vector
        
        # Rotate the unit vectors from local cluster frame to world coordinates
        world_x = rotation.apply(local_x)  # Rotate the X-axis unit vector
        world_y = rotation.apply(local_y)  # Rotate the Y-axis unit vector
        world_z = rotation.apply(local_z)  # Rotate the Z-axis unit vector

        # Set the new points in world space
        new_cluster_center_xform_points[4 * c] = origin  # Cluster center
        new_cluster_center_xform_points[4 * c + 1] = origin + world_x  # X-axis in world coordinates
        new_cluster_center_xform_points[4 * c + 2] = origin + world_y  # Y-axis in world coordinates
        new_cluster_center_xform_points[4 * c + 3] = origin + world_z  # Z-axis in world coordinates

    # Projection: project the points closer to the camera to ensure visibility
    homogeneous_points = np.hstack([new_cluster_center_xform_points, np.ones((4 * clustering.num_clusters, 1))])  # Add homogeneous coord
    projected_points = homogeneous_points @ w2c.T  # Apply world-to-camera transformation
    projected_points[:, :3] *= 0.25  # Scale points down to move closer to the camera
    projected_points = projected_points @ np.linalg.inv(w2c).T  # Reproject back to world coordinates
    new_cluster_center_xform_points = projected_points[:, :3]  # Extract 3D coordinates

    # Update the LineSet points
    xform_lineset.points = o3d.utility.Vector3dVector(new_cluster_center_xform_points)
    
    # Update the visualization
    vis.update_geometry(xform_lineset)

def visualize(filepath_npz:str, config:Config):
    scene_data, clustering  = grig(filepath_npz, config)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    #-------------------------#
    #   initialize geometry   #
    #-------------------------#

    ## gaussian splat
    w2c, k = init_camera()
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    cluster_center_dots = initialize_cluster_center_dots(clustering, vis)
    xform_lineset = initialize_xform_lineset(clustering, vis)

    #----------------------#
    #   set up viewpoint   #
    #----------------------#
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

    #-----------------#
    #   render loop   #
    #-----------------#
    while True:
        ## time tracking
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        t = int(passed_frames % num_timesteps)

        ## update camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        # update gaussian splat
        im, depth = render(w2c, k, scene_data[t])
        pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
        pcd.points = pts  
        pcd.colors = cols
        vis.update_geometry(pcd)

        update_cluster_centers(t, clustering, w2c, cluster_center_dots, vis)

        update_lineset(t, clustering, w2c, xform_lineset, vis)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options

import sys
if __name__ == "__main__":
    config = KMeansConfig(
        timestride=10,
        POS=1,
        DPOS=1,
        DROT=1,
        K=24,
        remove_bg=True,
        normalize_features=True,
        color_mode=sys.argv[2]
    )
    visualize(F"./output/pretrained/{sys.argv[1]}/params.npz", config)
