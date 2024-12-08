#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
#og imports
import time
import open3d as o3d
from visualize import init_camera, render, rgbd2pcd 
import torch
import numpy as np

# our imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R  # To handle quaternion-based rotations
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster

# types
from parameters import Parameters
from config import Config, BaseConfig, KMeansConfig
from features import Features
from clustering import Clustering

###################################################################################################
##   NOTE:                                                                                       ##
##   1) There is a useful method [helpers.o3d_knn] for if we want to do shearing calculations.   ##
###################################################################################################

#--------------#
#   settings   #
#--------------#
USE_CLUSTER_TRANSFORMS = False

#----------------------#
#   global constants   #
#----------------------#
# camera configuration
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 15

#-------------#
#   methods   #
#-------------#
@torch.no_grad()
def solve(filepath_npz: str, config: Config, save_path: str = None) -> Tuple[Dict[str, torch.Tensor], Clustering]:
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
    np_fg_features = features.features[features.is_fg].cpu().numpy()

    if config.normalize_features:
        print("Normalizing Foreground Features")
        scaler = StandardScaler()
        np_fg_features = scaler.fit_transform(np_fg_features)

    print(F"Clustering Foreground Gaussians") 
    kmeans = KMeans(n_clusters=config.K)
    kmeans.fit(np_fg_features)
    np_fg_labels = kmeans.labels_
    np_feature_means = kmeans.cluster_centers_

    clustering = Clustering(
        config.K,
        features,
        torch.from_numpy(np_feature_means).long().to("cuda")
    )

    __analyze_cluster_pairs(np_fg_features, np_feature_means)

    if USE_CLUSTER_TRANSFORMS: __make_clusters_rigid(clustering, features)

    print(F"Coloring Gaussians (Mode={config.color_mode})")
    cmap = cm.get_cmap('turbo', config.K)
    feature_colors = None
    if config.color_mode == "POS": # Color by magnitude of position
        feature_colors = torch.abs(features.pos[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "DPOS": # Color by magnitude of velocity
        feature_colors = torch.abs(features.dpos_dt[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "DROT": # Color by magnitude of angular velocity
        feature_colors = torch.abs(features.drot_dt[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    elif config.color_mode == "CLUSTERS": # Color by cluster assignment
        cluster_colors = cmap(np_fg_labels)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((features.T,-1,3))
    elif config.color_mode == "2NN": # Color by second nearest neighbor 
        distances = pairwise_distances(np_fg_features, kmeans.cluster_centers_)
        second_nearest_clusters = distances.argsort(axis=1)[:, 1] # (N,)
        cluster_colors = cmap(second_nearest_clusters)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((features.T,-1,3))
    elif config.color_mode == "PENALIZE_FIRST": # Color by nearest neighbor after applying a penalty to the 1NN
        distances = pairwise_distances(np_fg_features, kmeans.cluster_centers_)
        distances_indices = distances.argsort(axis=1)
        for i in range(distances.shape[0]):
            distances[i, distances_indices[i, 0]] *= 1.2
        distances_first_penalized = distances.argsort(axis=1)[:, 0] # (N,)
        cluster_colors = cmap(distances_first_penalized)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((features.T,-1,3))
    elif config.color_mode == "INDECISION": # Color by if nearest neighbor changed after applying 1NN penalty
        cmap = cm.get_cmap("viridis",2)
        distances = pairwise_distances(np_fg_features, kmeans.cluster_centers_)
        distances_indices = distances.argsort(axis=1)
        initial_min_indices = distances.argsort(axis=1)[:, 0]
        for i in range(distances.shape[0]):
            distances[i, distances_indices[i, 0]] *= 1.2
        final_min_indices = distances.argsort(axis=1)[:, 0]
        flipped_array = (initial_min_indices != final_min_indices).astype(int)
        _feature_colors = cmap(flipped_array)[:, :3]  
        feature_colors = torch.from_numpy(_feature_colors).float().to("cuda").expand((features.T,-1,3))
    elif config.color_mode == "INDECISION_CONTINUOUS": # Color by penalty factor required to cause flip from 1NN to 2NN
        cmap = cm.get_cmap("inferno")
        distances = pairwise_distances(np_fg_features, kmeans.cluster_centers_)
        distances_indices = distances.argsort(axis=1)
        initial_min_indices = distances.argsort(axis=1)[:, 0]
        sorted_distances = distances.copy()
        sorted_indices = distances_indices.copy()
        initial_min_values = sorted_distances[np.arange(distances.shape[0]), sorted_indices[:, 0]]
        initial_second_min_values = sorted_distances[np.arange(distances.shape[0]), sorted_indices[:, 1]]
        flip_factors = initial_second_min_values / initial_min_values
        percentile_5 = np.percentile(flip_factors, 5)
        percentile_95 = np.percentile(flip_factors, 95)
        flip_factors_clipped = np.clip(flip_factors, percentile_5, percentile_95)
        flip_factors_normalized = (1 - ((flip_factors_clipped - percentile_5) / (percentile_95 - percentile_5))) ** 3
        _feature_colors = cmap(flip_factors_normalized)[:, :3]  
        feature_colors = torch.from_numpy(_feature_colors).float().to("cuda").expand((features.T, -1, 3))
    else: # The default coloring is to use RGB 
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

    if save_path is not None:
        save_scene_and_clustering(scene_data, clustering, save_path)

    return scene_data, clustering

#@GPT
def __quaternion_to_rotation_matrix(quat : torch.Tensor):
    """
    Convert a quaternion (w, x, y, z) to a rotation matrix.
    quat: Tensor of shape (..., 4), where the last dimension is (w, x, y, z)
    """
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x**2, y**2, z**2
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return torch.stack([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dim=-1)

#@GPT
def __make_clusters_rigid(clustering : Clustering, features : Features):
    """
    Mutate features such that position and rotation are a fixed transform from the cluster center.
    """
    if USE_CLUSTER_TRANSFORMS:
        for c in range(clustering.num_clusters):
            mask = clustering.masks[c]

            # Translate positions to cluster-centered coordinates
            features.pos[:, mask] -= clustering.center_pos[:, c].unsqueeze(1)

            for t in range(features.T):
                quat_wtoc = clustering.transformations[t, c, [3, 4, 5, 6]]  # wxyz format
                R_wtoc = __quaternion_to_rotation_matrix(quat_wtoc)

                # Rotate positions and rotations into cluster-aligned coordinates
                features.pos[t, mask] = __rotate_points(features.pos[t, mask], R_wtoc)

                quat_features = features.rot[t, mask]  # xyzw format
                quat_features = torch.cat([quat_features[..., 3:], quat_features[..., :3]], dim=-1)  # xyzw to wxyz

                features.rot[t, mask] = (
                    __quaternion_to_rotation_matrix(R_wtoc).matmul(quat_features.unsqueeze(-1)).squeeze(-1)
                )

            # Average over time in cluster coordinates
            features.pos[:, mask] = features.pos[:, mask].mean(dim=0, keepdim=True)
            features.rot[:, mask] = features.rot[:, mask].mean(dim=0, keepdim=True)

            # Transform back to world coordinates
            for t in range(features.T):
                quat_ctow = clustering.transformations[t, c, [3, 4, 5, 6]]  # wxyz format
                R_ctow = __quaternion_to_rotation_matrix(quat_ctow)

                features.pos[t, mask] = clustering.centers[t, c] + __rotate_points(features.pos[t, mask], R_ctow)

                quat_features = features.rot[t, mask]
                features.rot[t, mask] = R_ctow.matmul(quat_features.unsqueeze(-1)).squeeze(-1)

#@GPT 
def __rotate_points(points : torch.Tensor, rotation_matrix : torch.Tensor):
    """
    Apply a rotation matrix to a set of points.
    points: Tensor of shape (N, 3)
    rotation_matrix: Tensor of shape (3, 3) or (N, 3, 3)
    """
    return torch.einsum("nij,nj->ni", rotation_matrix, points)

def __analyze_cluster_pairs(np_fg_features, cluster_center_features):
    distances = pairwise_distances(np_fg_features, cluster_center_features)
    sorted_clusters = distances.argsort(axis=1)  # (N, K)

    _1NN_ids = sorted_clusters[:, 0]
    _2NN_ids = sorted_clusters[:, 1]

    matrix = np.zeros((config.K, config.K), dtype=int)
    for i, j in zip(_1NN_ids, _2NN_ids):
        matrix[i, j] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', origin='upper')
    plt.colorbar(label="Count")
    plt.title("2D Matrix of 1NN and 2NN Counts")
    plt.xlabel("2NN Cluster ID")
    plt.ylabel("1NN Cluster ID")
    plt.grid(False)
    plt.show()

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
    centers_t = clustering.center_pos[t].cpu()
    
    # NOTE: we project closer to the camera to be visible as an overlay
    __new_cluster_center_points = torch.cat([centers_t,torch.ones((clustering.num_clusters,1))],dim=-1) # (K,4)
    __new_cluster_center_points = __new_cluster_center_points.cpu().numpy() @ w2c.T
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

def save_scene_and_clustering(scene_data: List[Dict[str, torch.Tensor]], clustering: Clustering, filepath: str):
    """ Saves scene_data and clustering information to a file. """
    data = {
        'scene_data': scene_data,
        'clustering': clustering
    }
    torch.save(data, filepath)


def load_scene_and_clustering(filepath: str) -> Tuple[List[Dict[str, torch.Tensor]], Clustering]:
    """ Loads scene_data and clustering information from a file. """
    data = torch.load(filepath)
    scene_data = data['scene_data']
    clustering = data['clustering']
    return scene_data, clustering

def find_nearest_neighbor_chains_all_timesteps(clustering: Clustering, max_distance: float = 0.5) -> List[List[int]]:
    """ Find chains of clusters by considering all timesteps to determine the best adjacent cluster pairs. """
    centers_t = clustering.center_pos.cpu().numpy()       # (T, K, 3)
    dpos_t = clustering.center_dpos.cpu().numpy()      # (T, K, 3)
    drot_t = clustering.center_drot.cpu().numpy()      # (T, K, 3)
    
    T, K, _ = centers_t.shape
    
    # Aggregate features over all timesteps. This is bad but it does achieve a better result ...
    centers_mean = centers_t.mean(axis=0)              # (K, 3)
    centers_std = centers_t.std(axis=0)                # (K, 3)
    
    dpos_mean = dpos_t.mean(axis=0)                    # (K, 3)
    dpos_std = dpos_t.std(axis=0)                      # (K, 3)
    
    drot_mean = drot_t.mean(axis=0)                    # (K, 3)
    drot_std = drot_t.std(axis=0)                      # (K, 3)
    
    # Combine aggregated features into a single feature vector per cluster
    combined_features = np.hstack((
        centers_mean, centers_std,
        dpos_mean, dpos_std,
        drot_mean, drot_std
    ))  # Shape: (K, 18)
    
    num_clusters = combined_features.shape[0]
    
    visited = np.zeros(num_clusters, dtype=bool)
    cluster_chains = []
    
    # Use KDTree with aggregated combined features
    tree = KDTree(combined_features)
    
    for i in range(num_clusters):
        if visited[i]:
            continue
    
        chain = [i]
        visited[i] = True
        current_cluster = i
    
        while True:
            # Query for neighbors sorted by distance
            distances, indices = tree.query(combined_features[current_cluster].reshape(1, -1), k=num_clusters)
            distances = distances.flatten()
            indices = indices.flatten()
    
            # Find the nearest unvisited neighbor within max_distance
            nearest_neighbor = -1
            for dist, idx in zip(distances[1:], indices[1:]):  # Skip the first entry (itself)
                if dist > max_distance:
                    break  # Neighbors are sorted by distance, break early
                if not visited[idx]:
                    nearest_neighbor = idx
                    break
    
            if nearest_neighbor == -1:
                break
    
            # Add the nearest neighbor to the chain
            chain.append(nearest_neighbor)
            visited[nearest_neighbor] = True
            current_cluster = nearest_neighbor
    
        cluster_chains.append(chain)
    
    print(f"Found {len(cluster_chains)} chains of clusters.")
    return cluster_chains

def compute_joints(clustering: Clustering, cluster_chains: List[List[int]]) -> np.ndarray:
    """ Computes the joint positions as the midpoint between consecutive clusters in each chain."""
    centers_t = clustering.center_pos.cpu().numpy()  # (T, K, 3)
    num_timesteps = centers_t.shape[0]

    # Compute the number of joints based on consecutive pairs in chains
    num_joints = sum(len(chain) - 1 for chain in cluster_chains)
    
    # Initialize the joints tensor
    joints_t = np.zeros((num_timesteps, num_joints, 3))  # (T, num_joints, 3)

    joint_idx = 0
    for chain in cluster_chains:
        pairs = [(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]

        for i, j in pairs:
            # Vectorized midpoint computation for all timesteps
            joints_t[:, joint_idx, :] = (centers_t[:, i, :] + centers_t[:, j, :]) / 2.0
            joint_idx += 1

    print(f"Computed {num_joints} joints.")
    return joints_t

def initialize_joint_dots(joints: np.ndarray, vis: o3d.visualization.Visualizer) -> List[o3d.geometry.TriangleMesh]:
    """ Initializes the memory for the joint dots visualization. """
    joint_dots = []
    for _ in range(joints.shape[1]):
        joint = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        joint.paint_uniform_color([1, 0, 0])
        joint_dots.append(joint)
        vis.add_geometry(joint)
    
    return joint_dots

def update_joints(t: int, joints_t: np.ndarray, joint_dots: List[o3d.geometry.TriangleMesh], w2c: np.ndarray, vis: o3d.visualization.Visualizer):
    """ Updates the joint dots visualization to the current time step `t`, projecting them closer to the camera. """
    # Get the joint positions for the current time step
    joints_at_t = joints_t[t]
    
    # Add homogeneous coordinates to the joint positions
    __new_joint_points = np.concatenate([joints_at_t, np.ones((joints_at_t.shape[0], 1))], axis=-1)  # (num_joints, 4)

    # Project the joint points to bring them closer to the camera
    __new_joint_points = __new_joint_points @ w2c.T
    __new_joint_points[:, :3] *= 0.25  # Move the points closer to the camera for visibility
    __new_joint_points = __new_joint_points @ np.linalg.inv(w2c).T  # Reproject back to world coordinates

    # Update the joint dot positions
    for idx, joint in enumerate(joint_dots):
        joint.translate(__new_joint_points[idx, :3], relative=False)
        vis.update_geometry(joint)


#@GPT
def update_lineset(t:int, clustering:Clustering, w2c:np.ndarray, xform_lineset:o3d.geometry.LineSet, vis:o3d.visualization.Visualizer):
    """Updates the xform lineset visualization to timestep `t`."""
    centers_t = clustering.center_pos[t]  # Assuming you have access to this
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

def main(filepath_npz: str, config: Config, clustering_filepath: str = None , save_path: str = "clustering.npz"):
    joint = False
    if clustering_filepath is not None:
        print("Loading clustering from file.")
        scene_data, clustering = load_scene_and_clustering(clustering_filepath)
    else:
        print("Running clustering algorithm.")
        scene_data, clustering = solve(filepath_npz, config, save_path=save_path)

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

    #cluster_center_dots = initialize_cluster_center_dots(clustering, vis)
    #xform_lineset = initialize_xform_lineset(clustering, vis)


    #----------------------#
    #     Evan's Joint     #
    #----------------------#
    # Add joint initialization
    if joint: 
        cluster_chains = find_nearest_neighbor_chains_all_timesteps(clustering, max_distance=0.5)
        joints_t = compute_joints(clustering, cluster_chains)

        joint_dots = initialize_joint_dots(joints_t, vis)

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

        #update_cluster_centers(t, clustering, w2c, cluster_center_dots, vis)
        #update_lineset(t, clustering, w2c, xform_lineset, vis)

        #----------------------#
        #     Evan's Joint     #
        #----------------------#
        if joint: 
            update_joints(t, joints_t, joint_dots, w2c, vis)

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
        timestride=1,
        POS=1,
        DPOS=1,
        DROT=1,
        K=28,
        remove_bg=True,
        normalize_features=True,
        color_mode=sys.argv[2]
    )
    main(F"./output/pretrained/{sys.argv[1]}/params.npz", config)
