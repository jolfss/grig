#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
#og imports
import time
import open3d as o3d
from visualize import render, rgbd2pcd 
import torch
import numpy as np

# our imports
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R  # To handle quaternion-based rotations
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster
from torch.nn.functional import normalize

# types
from parameters import Parameters
from config import Config, KMeansConfig, KMedoidsConfig
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
w, h = 350,350
near, far = 0.01, 100.0
view_scale = 1
fps = 20

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82, 
                cam_right=0.5, zoom_factor=1.5):
    ry = y_angle * np.pi / 180
    # Move camera to the right by cam_right
    w2c = np.array([
        [np.cos(ry), 0., -np.sin(ry), cam_right],
        [0.,         1., 0.,          cam_height],
        [np.sin(ry), 0., np.cos(ry),  center_dist],
        [0.,         0., 0.,          1.]
    ])
    # Zoom in by zoom_factor
    k = np.array([
        [f_ratio * w * zoom_factor, 0,                          w / 2],
        [0,                         f_ratio * w * zoom_factor,  h / 2],
        [0,                         0,                          1]
    ])
    return w2c, k


#-------------#
#   methods   #
#-------------#
def solve(filepath_npz: str, config: Config, save_path: str = None, silent=False) -> Tuple[Dict[str, torch.Tensor], Clustering, Dict[str,Any]]:
    """
    The clustering algorithm `(param_filepath, config) -> (scene_data, clustering, miscellaneous)`. 
    """
    def _print(string): 
        if not silent: _print(string)
    
    _print(F"Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    params = {key : params[key] for key in Parameters.__dataclass_fields__ if key in params}
    params = Parameters(**params)

    _print("Preparing Foreground Features")
    features = Features(params, config)
    np_fg_features : np.ndarray = features.features[features.is_fg].cpu().numpy()

    def custom_dist(x,y):
        """Combines linear and angular distance scalars into a single distance.
        NOTE: Oddly, doesn't seem to work very well at all for angles."""
        pos_dist=0
        rot_dist=0
        dpos_dist=0
        drot_dist=0

        if features.pos_slice:
            pos_dist =  np.linalg.norm(x[features.pos_slice] - y[features.pos_slice])
        if features.rot_slice:
            qrot1 = x[features.rot_slice]
            qrot2 = y[features.rot_slice]
            dot_product = np.dot(qrot1, qrot2)
            dot_product = max(-1.0, min(1.0, dot_product))
            rot_dist = 2 * np.arccos(abs(dot_product))
        if features.dpos_slice:
            dpos_dist = np.linalg.norm(x[features.dpos_slice]- y[features.dpos_slice])
        if features.drot_slice:
            qdrot1 = x[features.drot_slice]
            qdrot2 = y[features.drot_slice]
            dot_product = np.dot(qdrot1, qdrot2)
            dot_product = max(-1.0, min(1.0, dot_product))
            drot_dist = 2 * np.arccos(abs(dot_product))

        return pos_dist + rot_dist + dpos_dist + drot_dist

    if isinstance(config, KMeansConfig):
        _print(F"Clustering Foreground Gaussians")
        kmeans = KMeans(n_clusters=config.K, n_init=1)
        kmeans.fit(np_fg_features)
        np_fg_labels = kmeans.labels_
        np_fg_feature_means = kmeans.cluster_centers_
        torch_fg_labels = torch.from_numpy(np_fg_labels).long()  # Needs to be on CPU for apply_
    elif isinstance(config, KMedoidsConfig):
        # TODO: implementation of KMedoids
        _print("Clustering Foreground Gaussians using KMedoids")
        from sklearn_extra.cluster import CLARA
        kmedoids = CLARA(n_clusters=config.K,n_sampling=config.sample_size, n_sampling_iter=config.samplings)#, metric=custom_dist)
        kmedoids.fit(np_fg_features)
        np_fg_labels = kmedoids.labels_
        np_fg_feature_means = kmedoids.cluster_centers_
        torch_fg_labels = torch.from_numpy(np_fg_labels).long()  # Needs to be on CPU for apply_
    else:
        _print(F"Unsupported Config Type: {type(config), config}")
        return
    
    # sort based on cluster population
    label_counts = torch.bincount(torch_fg_labels, minlength=config.K)
    __sort_order = np.argsort(-label_counts.numpy())  # Descending order of cluster sizes
    __relabeling_fn = {old_label: new_label for new_label, old_label in enumerate(__sort_order)}.get
    
    # map labels
    torch_fg_labels.apply_(__relabeling_fn)
    np_fg_labels = np.array([__relabeling_fn(label) for label in np_fg_labels], dtype=int)
    np_fg_feature_means = np_fg_feature_means[__sort_order]

    clustering = Clustering(
        config.K,
        features,
        torch_fg_labels
        #torch.from_numpy(np_feature_means).to("cuda")
    )
    if config.use_cluster_transforms: __make_clusters_rigid(clustering, features)

    _print(F"Coloring Gaussians (Mode={config.color_mode})")
    cmap = cm.get_cmap('turbo', config.K)
    feature_colors = None
    if config.color_mode == "POS": # Color by magnitude of position
        feature_colors = torch.abs(features.pos[:,:,:3][:,features.is_fg]) #(T,N[is_fg],3)
    if config.color_mode == "ROT": # Color by magnitude of quaternion xyz components 
        feature_colors = torch.abs(features.rot[:,:,1:][:,features.is_fg]) #(T,N[is_fg],3)
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

    _print(F"Preparing Rendervars")
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
        save_scene_and_clustering(
            scene_data, clustering, save_path, 
            # optional kwargs
            np_fg_features=np_fg_features,
            np_fg_feature_means=np_fg_feature_means,
        )

    return scene_data, clustering, {'np_fg_features':np_fg_features,'np_fg_feature_means':np_fg_feature_means}


#@GPT--------------------------------------------------------------------------------------------------------------------
#@PROMPT
# def __make_clusters_rigid(clustering: Clustering, features: Features):
#     T = clustering.center_pos.shape[0]  # Number of time steps

#     for c in range(clustering.num_clusters):
#         mask = clustering.masks[c]  # Points belonging to cluster `c`

#         #   given: 
#         #   features.pos[t,mask] (device=cuda) are the positions of every point which belong to cluster c in world space
#         #   features.rot[t,mask] (device=cuda) are the rotations of every point which belong to cluster c in world space
#         #   clustering.center_pos[t,c] (device=cuda) is the position of cluster c's coordinate system's origin
#         #   clustering.transformations[t,c,3:7] (device=cuda) is the quaternion representing cluster c's coordinate system rotation in world space

#         # for all timesteps
#         #   transform features.pos[t, mask] from world space to cluster coordinate space
#         #   transform features.rot[t, mask] from world space to cluster coordinate space

#         # for each point
#             #   compute average offset in cluster coordinates across time
#             #   compute average rotation in cluster coordinates across time

#         # for all timesteps
#         #   set features.pos[t,mask] to the average cluster position + the per-point average position over time
#         #   set features.rot[t,mask] to the average cluster rotation + the per-point average rotation over time
def quaternion_inverse(q):
    """Compute the inverse of a quaternion assuming q is normalized."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quaternion_mul(q1, q2):
    """Multiply two quaternions q1 and q2. Shape: (..., 4)"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_rotate_point(q, p):
    """Rotate point p by quaternion q."""
    w, x, y, z = q.unbind(-1)
    px, py, pz = p.unbind(-1)
    t2 = w * px + y * pz - z * py
    t3 = w * py + z * px - x * pz
    t4 = w * pz + x * py - y * px
    t1 = -x * px - y * py - z * pz
    rx = t2*w - t1*x - t3*z + t4*y
    ry = t3*w - t1*y - t4*x + t2*z
    rz = t4*w - t1*z - t2*y + t3*x
    return torch.stack([rx, ry, rz], dim=-1)

def quaternion_average(quats):
    """Average a set of quaternions by normalizing their sum."""
    sum_q = quats.sum(dim=0, keepdim=True)
    avg_q = normalize(sum_q, p=2, dim=-1)
    return avg_q.squeeze(0)

def __make_clusters_rigid(clustering, features):
    T = clustering.center_pos.shape[0]
    num_clusters = clustering.num_clusters

    for c in range(num_clusters):
        mask = clustering.masks[c]  # boolean or index mask
        # Extract per-cluster per-point data
        # pos_world: (T, M, 3) where M = number of points in cluster c
        pos_world = features.pos[:, mask, :]  # (T, M, 3)
        rot_world = features.rot[:, mask, :]  # (T, M, 4)

        # Extract cluster transform
        cluster_pos = clustering.center_pos[:, c, :]        # (T, 3)
        cluster_quat = clustering.transformations[:, c, 3:7] # (T, 4)
        cluster_quat_inv = quaternion_inverse(cluster_quat)  # (T, 4)

        # Transform points into cluster coords
        pos_cluster = quaternion_rotate_point(cluster_quat_inv.unsqueeze(1).expand(-1, pos_world.shape[1], -1), 
                                              pos_world - cluster_pos.unsqueeze(1))

        # Transform rotations into cluster coords
        # rot_cluster[t,m] = q_c_inv[t] * rot_world[t,m]
        rot_cluster = quaternion_mul(cluster_quat_inv.unsqueeze(1).expand(-1, rot_world.shape[1], -1),
                                     rot_world)

        # Compute average position and rotation for each point in cluster coordinates
        # Average over T
        avg_pos_cluster = pos_cluster.mean(dim=0)   # (M, 3)
        # For rotation averaging, do it point-by-point
        avg_rot_cluster = []
        for m in range(rot_cluster.shape[1]):
            avg_rot_cluster.append(quaternion_average(rot_cluster[:, m, :]))
        avg_rot_cluster = torch.stack(avg_rot_cluster, dim=0)  # (M, 4)



        # Transform back to world:
        pos_world_rigid = quaternion_rotate_point(cluster_quat.unsqueeze(1).expand(-1, avg_pos_cluster.shape[0], -1),
                                                  avg_pos_cluster.unsqueeze(0)) + cluster_pos.unsqueeze(1)
        rot_world_rigid = quaternion_mul(cluster_quat.unsqueeze(1).expand(-1, avg_rot_cluster.shape[0], -1),
                                         avg_rot_cluster.unsqueeze(0))

        # Assign back
        features.pos[:, mask, :] = pos_world_rigid
        features.rot[:, mask, :] = rot_world_rigid

    return features
#@ENDGPT--------------------------------------------------------------------------------------------------------------------

def analyze_cluster_adjacency(masks, np_fg_features, np_fg_feature_means, save_path: str = None, show=False) -> np.ndarray:
    distances = pairwise_distances(np_fg_features, np_fg_feature_means)
    sorted_clusters = distances.argsort(axis=1)
    _1NN_ids = sorted_clusters[:, 0]
    _2NN_ids = sorted_clusters[:, 1]

    matrix = np.zeros((config.K, config.K), dtype=float)
    for i, j in zip(_1NN_ids, _2NN_ids):
        matrix[i, j] += 1
    
    for i in np.unique(_1NN_ids):  
        mask_sum = torch.sum(masks[i]).item()
        if mask_sum > 0:
            matrix[i] /= mask_sum

    if save_path:
        matrix_8bit = (matrix * 255).astype(np.uint8)
        mpimg.imsave(save_path, matrix_8bit, cmap='Greys_r')

    elif show:
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, origin='upper', cmap='Greys_r')
        plt.title("Asymmetric Normalized 2D Matrix of 1NN and 2NN Counts")
        plt.xlabel("2NN Cluster ID")
        plt.ylabel("1NN Cluster ID")
        plt.grid(False)
        plt.tight_layout()

        plt.show()
        plt.close()

    return matrix


import networkx as nx
def initialize_adjacency_lineset(distance_matrix : np.ndarray, clustering:Clustering, vis:o3d.visualization.Visualizer, arborescence=False) -> o3d.geometry.LineSet:
    weight_matrix = np.vectorize(lambda dist:1/max(dist,1e-16))(distance_matrix) 

    adj = None
    if arborescence:
        adj = nx.minimum_spanning_arborescence(nx.from_numpy_matrix(weight_matrix, create_using=nx.DiGraph))
    else:#mst
        adj = nx.minimum_spanning_tree(nx.from_numpy_matrix(weight_matrix))

    adj_lineset = o3d.geometry.LineSet()
    adj_lines =  np.zeros((clustering.num_clusters-1,2))

    for i,(u,v) in enumerate(adj.edges):
        adj_lines[i]  = [u,v]

    adj_lineset.lines = o3d.utility.Vector2iVector(adj_lines)
    adj_lineset.paint_uniform_color([1,1,0])
    vis.add_geometry(adj_lineset)
    
    return adj_lineset, adj
    
def update_adjacency_lineset(t:int, clustering:Clustering, w2c:np.ndarray, adj_lineset:o3d.geometry.LineSet, vis:o3d.visualization.Visualizer): 
    """Updates the adjacency lineset visualization to timestep `t`."""
    centers_t = clustering.center_pos[t].cpu()
    __new_cluster_center_points = torch.cat([centers_t,torch.ones((clustering.num_clusters,1))],dim=-1)
    __new_cluster_center_points = __new_cluster_center_points.numpy() @ w2c.T
    __new_cluster_center_points[:,:3] *= 0.25
    __new_cluster_center_points = __new_cluster_center_points @ np.linalg.inv(w2c).T
    adj_lineset.points = o3d.utility.Vector3dVector(__new_cluster_center_points[:,:3])
    adj_lineset.paint_uniform_color([1,1,0])
    vis.update_geometry(adj_lineset)

def initialize_cluster_center_dots(clustering:Clustering, vis:o3d.visualization.Visualizer) -> List[o3d.geometry.TriangleMesh]:
    """Initializes the memory for the center dots visualization."""
    cluster_center_dots = []
    for _ in range(clustering.num_clusters):
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.003) 
        center.paint_uniform_color([1, 1, 1])   
        cluster_center_dots.append(center)
        vis.add_geometry(center)

    return cluster_center_dots

def update_cluster_centers(
        t: int, clustering:Clustering, 
        w2c: np.ndarray, 
        cluster_center_dots: List[o3d.geometry.TriangleMesh], 
        vis: o3d.visualization.Visualizer, 
        graph: Optional[Union[nx.DiGraph,nx.Graph]] = None,
):
    """Updates the center dots visualization to timestep `t`, with optional coloring by depth in a DiGraph."""
    centers_t = clustering.center_pos[t].cpu()

    # NOTE: project closer to the camera to be visible as an overlay
    __new_cluster_center_points = torch.cat(
        [centers_t, torch.ones((clustering.num_clusters, 1))], dim=-1
    )  # (K,4)
    __new_cluster_center_points = __new_cluster_center_points.numpy() @ w2c.T
    __new_cluster_center_points[:, :3] *= 0.25
    __new_cluster_center_points = __new_cluster_center_points @ np.linalg.inv(w2c).T

    # Handle coloring by depth in the graph if `graph` is a DiGraph
    colors = None
    if type(graph) == nx.DiGraph:
        # Compute depth for each node
        root = [node for node, in_degree in graph.in_degree() if in_degree == 0]
        if len(root) != 1:
            raise ValueError("The graph must have exactly one root node.")
        root = root[0]

        depths = nx.single_source_shortest_path_length(graph, root)
        max_depth = max(depths.values())

        # Normalize depths for colormap
        norm_depths = np.array([depths.get(i, max_depth + 1) for i in range(clustering.num_clusters)]) / max_depth
        cmap = plt.cm.get_cmap("Greys")  # Choose a colormap
        colors = cmap(norm_depths)[:, :3]  # Get RGB values (exclude alpha)

    for c in range(clustering.num_clusters):
        cluster_center_dots[c].translate(
            __new_cluster_center_points[c, :3], relative=False
        )
        
        # Update color if `graph` is provided
        if colors is not None:
            cluster_center_dots[c].paint_uniform_color(colors[c])

        vis.update_geometry(cluster_center_dots[c])

def initialize_xform_lineset(clustering:Clustering, vis:o3d.visualization.Visualizer) -> o3d.geometry.LineSet:
    """Initializes the lineset for the xform visualization."""
    lineset = o3d.geometry.LineSet()
    cluster_center_xform_points = np.zeros((4*clustering.num_clusters,3)) # 4 points (center,->x,->y,->z) for each cluster
    cluster_center_xform_lines =  np.zeros((3*clustering.num_clusters,2)) # connect center to each unit vector
    cluster_center_xform_colors = np.zeros((3*clustering.num_clusters,3)) # r,g,b axis colorings
    for c in range(clustering.num_clusters):
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
    centers_t = clustering.center_pos[t].cpu()  # Assuming you have access to this
    xforms_t = clustering.transformations[t].cpu()  # Get transformations for this time step

    new_cluster_center_xform_points = np.zeros((4 * clustering.num_clusters, 3))  # 4 points per cluster

    for c in range(clustering.num_clusters):
        # Extract the origin (translation) for cluster `c`
        origin = centers_t[c].numpy()

        # Extract the rotation as a quaternion [qw, qx, qy, qz]
        quaternion = xforms_t[c, 3:7].numpy()  # Extract [qw, qx, qy, qz]
        
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

def save_scene_and_clustering(scene_data: List[Dict[str, torch.Tensor]], 
                              clustering: Clustering, 
                              filepath: str,
                              **miscellaneous):
    """ Saves scene_data and clustering information to a file. """
    data = {
        'scene_data': scene_data,
        'clustering': clustering,
        'miscellaneous' : miscellaneous
    }
    torch.save(data, filepath)

def load_scene_and_clustering(filepath: str) -> Tuple[List[Dict[str, torch.Tensor]], Clustering, Dict[str,Any]]:
    """ Loads scene_data and clustering information from a file. """
    data = torch.load(filepath)
    scene_data = data['scene_data']
    clustering = data['clustering']
    miscellaneous = data['miscellaneous']
    
    return scene_data, clustering, miscellaneous

def find_nearest_neighbor_chains_all_timesteps(clustering: Clustering, max_distance: float = 0.5) -> List[List[int]]:
    """ Find chains of clusters by considering all timesteps to determine the best adjacent cluster pairs. """
    centers_t = clustering.center_pos.cpu().numpy()    # (T, K, 3)
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

@torch.no_grad()
def main(filepath_npz: str, config: Config, clustering_filepath: str = None , save_path: str = None):
    joint = False
    if clustering_filepath is not None and os.path.exists("./"+clustering_filepath):
        print("Loading clustering from file.")
        scene_data, clustering, miscellaneous = load_scene_and_clustering(clustering_filepath)
    else:
        print("Running clustering algorithm.")
        scene_data, clustering, miscellaneous = solve(filepath_npz, config, save_path=save_path)

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
    #xform_lineset = initialize_xform_lineset(clustering, vis)
    cluster_adjacency_matrix = analyze_cluster_adjacency(clustering.masks, miscellaneous['np_fg_features'], miscellaneous['np_fg_feature_means'], show=config.show_adjacency)
    adj_lineset, graph = initialize_adjacency_lineset(cluster_adjacency_matrix, clustering, vis, arborescence=config.arborescence)

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

        update_cluster_centers(t, clustering, w2c, cluster_center_dots, vis, graph)
        #update_lineset(t, clustering, w2c, xform_lineset, vis)
        update_adjacency_lineset(t, clustering, w2c, adj_lineset, vis)

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
    base_config = {
        "timestride":1,
        "POS" :  64,      
        "ROT" :  0, # ROT seems like a bad feature even though it contains some information
        "DPOS":  1,
        "DROT":  10,
        "pre_normalize":True,
        "remove_bg":True,
        "color_mode":sys.argv[2],
        "arborescence":False,
        "show_adjacency":True,
        "use_cluster_transforms":True,
    }

    K=28
    if len(sys.argv) > 3:
        if sys.argv[3] == "MEDOIDS":
            config = KMedoidsConfig(
                K=K, 
                sample_size=2000,
                samplings=10,
                **base_config
            )
    else:
        config = KMeansConfig(
            K=K, 
            **base_config
        )
    
    print(F"Using config: {config}")

    main(F"./output/pretrained/{sys.argv[1]}/params.npz", config)


def initialize_adjacency_lineset_offscreen(distance_matrix: np.ndarray, clustering, arborescence=False):
    # Convert distances into weights
    weight_matrix = np.vectorize(lambda dist:1/max(dist,1e-16))(distance_matrix)

    # Build graph and compute MST or Arborescence
    if arborescence:
        # Directed minimum spanning arborescence
        adj = nx.minimum_spanning_arborescence(nx.from_numpy_matrix(weight_matrix, create_using=nx.DiGraph))
    else:
        # Undirected minimum spanning tree
        adj = nx.minimum_spanning_tree(nx.from_numpy_matrix(weight_matrix))

    adj_lineset = o3d.geometry.LineSet()
    adj_lines = np.zeros((clustering.num_clusters-1, 2), dtype=np.int32)

    for i, (u, v) in enumerate(adj.edges):
        adj_lines[i] = [u, v]

    adj_lineset.lines = o3d.utility.Vector2iVector(adj_lines)
    # We'll set points later when we have camera transformations, if needed.
    adj_lineset.paint_uniform_color([0, 0, 0])

    # IMPORTANT: Do NOT add to vis here, just return geometry and adjacency.
    return adj_lineset, adj

def update_adjacency_lineset_offscreen(t: int, clustering, w2c: np.ndarray, adj_lineset: o3d.geometry.LineSet, 
                             renderer: o3d.visualization.rendering.OffscreenRenderer, name: str, material):
    """Updates the adjacency lineset visualization for timestep `t`.
    Here, `name` is the identifier used when the geometry was added to the renderer scene.
    `material` is the MaterialRecord used when originally adding the geometry."""
    centers_t = clustering.center_pos[t].cpu()
    __new_cluster_center_points = np.concatenate([centers_t.numpy(), np.ones((clustering.num_clusters, 1))], axis=-1)
    __new_cluster_center_points = __new_cluster_center_points @ w2c.T
    __new_cluster_center_points[:, :3] *= 0.25
    __new_cluster_center_points = __new_cluster_center_points @ np.linalg.inv(w2c).T

    adj_lineset.points = o3d.utility.Vector3dVector(__new_cluster_center_points[:, :3])
    adj_lineset.paint_uniform_color([0, 0, 0])

    # Remove old geometry and re-add the updated one
    renderer.scene.remove_geometry(name)
    renderer.scene.add_geometry(name, adj_lineset, material)

def initialize_cluster_center_dots_offscreen(clustering) -> List[o3d.geometry.TriangleMesh]:
    """Initializes the cluster center dots (small spheres) but does not add them to a scene.
       Just returns the list of geometries."""
    cluster_center_dots = []
    for _ in range(clustering.num_clusters):
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        center.paint_uniform_color([0,0,0])
        cluster_center_dots.append(center)
    return cluster_center_dots

def update_cluster_centers_offscreen(
    t: int, 
    clustering, 
    w2c: np.ndarray, 
    cluster_center_dots: List[o3d.geometry.TriangleMesh],
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    names: List[str],
    materials: List[o3d.visualization.rendering.MaterialRecord],
    graph: Optional[Union[nx.DiGraph, nx.Graph]] = None,
):
    """Updates the center dots visualization to timestep `t`.
       `names` is a list of geometry identifiers as added to the renderer scene.
       `materials` is a list of MaterialRecords for each dot geometry.
    """
    centers_t = clustering.center_pos[t].cpu()
    __new_cluster_center_points = np.concatenate([centers_t.numpy(), np.ones((clustering.num_clusters, 1))], axis=-1)
    __new_cluster_center_points = __new_cluster_center_points @ w2c.T
    __new_cluster_center_points[:, :3] *= 0.25
    __new_cluster_center_points = __new_cluster_center_points @ np.linalg.inv(w2c).T

    colors = None
    if isinstance(graph, nx.DiGraph):
        # Compute coloring based on depth in a directed graph.
        root = [node for node, in_degree in graph.in_degree() if in_degree == 0]
        if len(root) != 1:
            raise ValueError("The graph must have exactly one root node.")
        root = root[0]

        depths = nx.single_source_shortest_path_length(graph, root)
        max_depth = max(depths.values())
        norm_depths = np.array([depths.get(i, max_depth + 1) for i in range(clustering.num_clusters)]) / max_depth

        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("Greys")
        colors = cmap(norm_depths)[:, :3]

    # Update the position of each sphere and re-add to renderer scene
    for c in range(clustering.num_clusters):
        # Translate the geometry to the new position
        # Note: "translate(..., relative=False)" means the translation sets the absolute position.
        # We can achieve this by resetting the geometry to origin and then translating:
        old_mesh = cluster_center_dots[c]
        # To "reset" the mesh's transform, we can either recreate it or apply transformations carefully.
        # Here, for simplicity, we'll just translate relative to current position.
        # If it's already moved, we might need to reset it first. A trick is to store initial mesh before transformations.
        # For now, assume we know that each update sets an absolute position. We can do:
        old_mesh.translate(-old_mesh.get_center())  # move back to origin
        old_mesh.translate(__new_cluster_center_points[c, :3])  # move to new absolute position

        # Update color if needed
        if colors is not None:
            old_mesh.paint_uniform_color(colors[c])
        else:
            old_mesh.paint_uniform_color([0,0,0])

        # Re-add geometry with the same name and material
        renderer.scene.remove_geometry(names[c])
        renderer.scene.add_geometry(names[c], old_mesh, materials[c])


torch.no_grad()
def capture_timestep_offscreen(filepath_npz: str, config, save_name: str, timestep: int = 0):
    # Load scene data
    scene_data, clustering, miscellaneous = solve(filepath_npz, config, silent=True)

    # Initialize camera
    w2c, k = init_camera(y_angle=0., center_dist=3.7, cam_height=0.93, f_ratio=1.50, cam_right=-0.05, zoom_factor=1.5)
    width = int(w * view_scale)
    height = int(h * view_scale)

    # Create offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.scene.set_indirect_light_intensity(0.0)
    renderer.scene.scene.set_sun_light([0, 0, -1], [1, 1, 1], 0.0)
    renderer.scene.set_background([0, 0, 0, 1])  # RGBA black background

    # Render initial scene data (timestep = 0)
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    # Define a material for unlit rendering
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(view_scale)
    mat.base_color = [0.9, 0.9, 0.9, 1.0]
    mat.line_width = 5.0

    # Add pointcloud to the scene
    renderer.scene.add_geometry("pointcloud", pcd, mat)

    # Initialize cluster centers (as small spheres)
    cluster_center_geometries = initialize_cluster_center_dots_offscreen(clustering)

    # Create lists for names and materials of cluster centers
    dot_names = []
    dot_materials = []
    for i, ccdot in enumerate(cluster_center_geometries):
        name = f"cluster_center_{i}"
        dot_names.append(name)
        dot_materials.append(mat)
        renderer.scene.add_geometry(name, ccdot, mat)

    distances = pairwise_distances(miscellaneous["np_fg_features"], miscellaneous["np_fg_feature_means"])
    sorted_clusters = distances.argsort(axis=1)
    _1NN_ids = sorted_clusters[:, 0]
    _2NN_ids = sorted_clusters[:, 1]



    cluster_adjacency_matrix = np.zeros((config.K, config.K), dtype=float)
    for i, j in zip(_1NN_ids, _2NN_ids):
        cluster_adjacency_matrix[i, j] += 1
    
    for i in np.unique(_1NN_ids):  
        mask_sum = torch.sum(clustering.masks[i]).item()
        if mask_sum > 0:
            cluster_adjacency_matrix[i] /= mask_sum

    # get me the mean of the entropies of each row of cluster_adjacency_matrix
    def shannon_entropy(p_row):
        # Ensure no negative or NaN values
        p_row = np.clip(p_row, 0, 1)
        # Filter out zero probabilities to avoid log(0)
        mask = p_row > 0
        p = p_row[mask]
        # Compute entropy in bits
        return -np.sum(p * np.log2(p))

    # Assuming cluster_adjacency_matrix is a 2D NumPy array
    mean_entropy = np.mean([shannon_entropy(row) for row in cluster_adjacency_matrix])
    plt.figure(figsize=(6, 6))
    plt.imshow(cluster_adjacency_matrix, origin='upper', cmap='Greys_r')
    plt.title("Asymmetric Normalized 2D Matrix of 1NN and 2NN Counts")
    plt.xlabel("2NN Cluster ID")
    plt.ylabel("1NN Cluster ID")
    plt.grid(False)
    plt.tight_layout()

    if not os.path.exists("hpsearch/matrices/"):
        os.makedirs("hpsearch/matrices")
    plt.savefig(f"hpsearch/matrices/{save_name}.png", dpi=300)
    plt.close()

    # Initialize adjacency lineset offscreen
    adj_lineset, graph = initialize_adjacency_lineset_offscreen(cluster_adjacency_matrix, clustering, arborescence=config.arborescence)
    line_name = "adj_lineset"
    line_material = mat  # could define another if you like
    renderer.scene.add_geometry(line_name, adj_lineset, line_material)

    # Setup camera
    view_k = k * view_scale
    view_k[2, 2] = 1

    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.extrinsic = w2c
    cam_params.intrinsic.set_intrinsics(width, height, view_k[0,0], view_k[1,1], view_k[0,2], view_k[1,2])

    renderer.setup_camera(cam_params.intrinsic, cam_params.extrinsic)

    # Update scene for the given timestep
    im, depth = render(w2c, k, scene_data[timestep])
    pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=False)
    pcd.points = pts
    pcd.colors = cols
    # Update the pointcloud by removing and re-adding it
    renderer.scene.remove_geometry("pointcloud")
    renderer.scene.add_geometry("pointcloud", pcd, mat)

    # Update cluster centers and adjacency lines
    update_cluster_centers_offscreen(
        timestep, 
        clustering, 
        w2c, 
        cluster_center_geometries, 
        renderer, 
        dot_names, 
        dot_materials, 
        graph=graph
    )

    update_adjacency_lineset_offscreen(
        timestep, 
        clustering, 
        w2c, 
        adj_lineset, 
        renderer,
        line_name,
        line_material
    )

    # Render final image
    img = renderer.render_to_image()
    if not os.path.exists("hpsearch/renders"):
        os.makedirs("hpsearch/renders")
    o3d.io.write_image(f"hpsearch/renders/{save_name}_ENT={mean_entropy}.png", img)

    del renderer
