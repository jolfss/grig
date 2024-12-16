#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
#og imports
import time
import open3d as o3d
from visualize import init_camera, render, rgbd2pcd 
import torch
import numpy as np
import os

# our imports
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R  # To handle quaternion-based rotations
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

# types
from parameters import Parameters
from config import Config, BaseConfig, KMeansConfig
from features import Features
from clustering import Clustering
from enum import Enum

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

#----------------------#
#   Enumerations        #
#----------------------#
class BodyPart(Enum):
    TORSO = 'torso'
    HEAD = 'head'
    CHEST = 'chest'
    WAIST = 'waist'
    SHOULDERS_LEFT = 'shoulders_left'
    SHOULDERS_RIGHT = 'shoulders_right'
    NECK = 'neck'
    CLAVICLE_LEFT = 'clavicle_left'
    CLAVICLE_RIGHT = 'clavicle_right'
    UPPER_ARM_LEFT = 'upper_arm_left'
    LOWER_ARM_LEFT = 'lower_arm_left'
    HAND_LEFT = 'hand_left'
    UPPER_ARM_RIGHT = 'upper_arm_right'
    LOWER_ARM_RIGHT = 'lower_arm_right'
    HAND_RIGHT = 'hand_right'
    UPPER_LEG_LEFT = 'upper_leg_left'
    LOWER_LEG_LEFT = 'lower_leg_left'
    FOOT_LEFT = 'foot_left'
    UPPER_LEG_RIGHT = 'upper_leg_right'
    LOWER_LEG_RIGHT = 'lower_leg_right'
    FOOT_RIGHT = 'foot_right'
    # Add more body parts as needed

#----------------------#
#   Helper Functions   #
#----------------------#

def find_closest_cluster(clustering: Clustering, reference_idx: int, direction: str = 'down', threshold: float = 0.5) -> Optional[int]:
    """
    Finds the closest cluster to the reference cluster in a specified direction within a distance threshold.
    
    Args:
        clustering (Clustering): Clustering information containing cluster centers.
        reference_idx (int): Index of the reference cluster.
        direction (str): Direction to search ('down', 'forward', 'up', 'backward', etc.).
        threshold (float): Maximum allowable distance to consider.
    
    Returns:
        Optional[int]: Index of the closest cluster in the specified direction. Returns None if not found.
    """
    # Compute average centers across all timesteps
    centers = clustering.centers.mean(dim=0).cpu().numpy()  # Shape: (K, 3)
    num_clusters = clustering.num_clusters
    
    # Validate reference_idx
    if not (0 <= reference_idx < num_clusters):
        print(f"Error: reference_idx {reference_idx} is out of bounds for number of clusters {num_clusters}.")
        return None

    reference_center = centers[reference_idx]

    # Determine candidates based on direction
    if direction == 'down':
        candidates = np.where(centers[:, 1] < reference_center[1])[0]
    elif direction == 'forward':
        candidates = np.where(centers[:, 2] > reference_center[2])[0]
    elif direction == 'up':
        candidates = np.where(centers[:, 1] > reference_center[1])[0]
    elif direction == 'backward':
        candidates = np.where(centers[:, 2] < reference_center[2])[0]
    else:
        print(f"Warning: Unknown direction '{direction}'. No candidates found.")
        candidates = np.array([])

    if len(candidates) == 0:
        print(f"No candidates found in direction '{direction}' for reference cluster {reference_idx}.")
        return None

    # Compute Euclidean distances to the reference cluster
    distances = np.linalg.norm(centers[candidates] - reference_center, axis=1)

    # Filter candidates within the threshold
    valid_indices = np.where(distances < threshold)[0]
    if len(valid_indices) == 0:
        print(f"No candidates within threshold {threshold} in direction '{direction}' for reference cluster {reference_idx}.")
        return None

    valid_candidates = candidates[valid_indices]
    valid_distances = distances[valid_indices]

    # Find the index of the minimum distance within valid candidates
    min_distance_idx = np.argmin(valid_distances)

    closest_idx = valid_candidates[min_distance_idx]

    # Additional check to ensure closest_idx is within bounds
    if closest_idx >= num_clusters or closest_idx < 0:
        print(f"Error: closest_idx {closest_idx} is out of bounds for number of clusters {num_clusters}.")
        return None

    return closest_idx

import numpy as np
from sklearn.neighbors import KDTree
from collections import deque
from typing import Dict, List, Optional
from enum import Enum

# Assuming BodyPart Enum is defined as in your original code
class BodyPart(Enum):
    TORSO = 'torso'
    HEAD = 'head'
    CHEST = 'chest'
    WAIST = 'waist'
    SHOULDERS_LEFT = 'shoulders_left'
    SHOULDERS_RIGHT = 'shoulders_right'
    NECK = 'neck'
    CLAVICLE_LEFT = 'clavicle_left'
    CLAVICLE_RIGHT = 'clavicle_right'
    UPPER_ARM_LEFT = 'upper_arm_left'
    LOWER_ARM_LEFT = 'lower_arm_left'
    HAND_LEFT = 'hand_left'
    UPPER_ARM_RIGHT = 'upper_arm_right'
    LOWER_ARM_RIGHT = 'lower_arm_right'
    HAND_RIGHT = 'hand_right'
    UPPER_LEG_LEFT = 'upper_leg_left'
    LOWER_LEG_LEFT = 'lower_leg_left'
    FOOT_LEFT = 'foot_left'
    UPPER_LEG_RIGHT = 'upper_leg_right'
    LOWER_LEG_RIGHT = 'lower_leg_right'
    FOOT_RIGHT = 'foot_right'
    # Add more body parts as needed

def assign_clusters_to_body_parts_enforced(
    clustering: 'Clustering',
    scene_data: List[Dict[str, torch.Tensor]],
    threshold: float = 0.5,
    ignored_clusters: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, BodyPart]:
    """
    Automatically assigns clusters to body parts based on a 9-dimensional feature space
    and skeletal hierarchy, utilizing KDTree for efficient nearest neighbor searches.
    Clusters specified in ignored_clusters are excluded from assignment.

    Args:
        clustering (Clustering): Clustering information containing cluster centers.
        scene_data (List[Dict[str, torch.Tensor]]): Scene data loaded from clustering.
        threshold (float): Maximum allowable distance to consider for assignment.
        ignored_clusters (Optional[List[int]]): List of cluster indices to ignore.
        verbose (bool): If True, prints detailed logs.

    Returns:
        Dict[int, BodyPart]: Mapping from cluster index to BodyPart.
    """

    # Compute average and standard deviation features across all timesteps
    centers_t = clustering.centers.cpu().numpy()       # Shape: (T, K, 3)
    dpos_t = clustering.center_dpos.cpu().numpy()     # Shape: (T, K, 3)
    drot_t = clustering.center_drot.cpu().numpy()     # Shape: (T, K, 3)
    
    T, K, _ = centers_t.shape
    
    # Aggregate features over all timesteps
    centers_mean = centers_t.mean(axis=0)             # Shape: (K, 3)
    dpos_mean = dpos_t.mean(axis=0)                   # Shape: (K, 3)
    drot_mean = drot_t.mean(axis=0)                   # Shape: (K, 3)
    
    # Combine aggregated features into a single feature vector per cluster (9D)
    combined_features = np.hstack((
        centers_mean,    # Mean Position: x, y, z
        dpos_mean,       # Mean Velocity: dpos_x, dpos_y, dpos_z
        drot_mean        # Mean Rotation: drot_x, drot_y, drot_z
    ))  # Shape: (K, 9)
    
    if verbose:
        print(f"Combined feature shape: {combined_features.shape}")
    
    # Initialize cluster assignments
    cluster_assignments: Dict[int, BodyPart] = {}
    assigned_clusters = set()
    
    # Add ignored clusters to assigned_clusters to exclude them from assignment
    if ignored_clusters is None:
        ignored_clusters = []
    for idx in ignored_clusters:
        if 0 <= idx < K:
            assigned_clusters.add(idx)
            if verbose:
                print(f"Ignored Cluster {idx} has been excluded from assignments.")
        else:
            if verbose:
                print(f"Ignored Cluster {idx} is out of bounds and cannot be excluded.")
    
    # Define the skeletal hierarchy connections
    BODY_CONNECTIONS: Dict[BodyPart, List[BodyPart]] = {
        BodyPart.TORSO: [
            BodyPart.CLAVICLE_LEFT,
            BodyPart.CLAVICLE_RIGHT,
            BodyPart.HEAD,
            BodyPart.UPPER_LEG_LEFT,
            BodyPart.UPPER_LEG_RIGHT
        ],
        BodyPart.CLAVICLE_LEFT: [
            BodyPart.SHOULDERS_LEFT
        ],
        BodyPart.CLAVICLE_RIGHT: [
            BodyPart.SHOULDERS_RIGHT
        ],
        BodyPart.SHOULDERS_LEFT: [
            BodyPart.UPPER_ARM_LEFT
        ],
        BodyPart.SHOULDERS_RIGHT: [
            BodyPart.UPPER_ARM_RIGHT
        ],
        BodyPart.HEAD: [
            BodyPart.NECK
        ],
        BodyPart.NECK: [
            BodyPart.CHEST
        ],
        BodyPart.CHEST: [
            BodyPart.WAIST
        ],
        BodyPart.UPPER_ARM_LEFT: [
            BodyPart.LOWER_ARM_LEFT
        ],
        BodyPart.LOWER_ARM_LEFT: [
            BodyPart.HAND_LEFT
        ],
        BodyPart.UPPER_ARM_RIGHT: [
            BodyPart.LOWER_ARM_RIGHT
        ],
        BodyPart.LOWER_ARM_RIGHT: [
            BodyPart.HAND_RIGHT
        ],
        BodyPart.UPPER_LEG_LEFT: [
            BodyPart.LOWER_LEG_LEFT
        ],
        BodyPart.LOWER_LEG_LEFT: [
            BodyPart.FOOT_LEFT
        ],
        BodyPart.UPPER_LEG_RIGHT: [
            BodyPart.LOWER_LEG_RIGHT
        ],
        BodyPart.LOWER_LEG_RIGHT: [
            BodyPart.FOOT_RIGHT
        ],
    }
    
    # Define direction vectors for body parts relative to their parents
    BODY_PART_DIRECTIONS: Dict[BodyPart, np.ndarray] = {
        BodyPart.CLAVICLE_LEFT: np.array([0, -1, 0]),  
        BodyPart.CLAVICLE_RIGHT: np.array([0.26, -1, 0]),  
        BodyPart.HEAD: np.array([0, -1, 0]),
        BodyPart.UPPER_LEG_LEFT: np.array([-0.5, 1, 0]),
        BodyPart.UPPER_LEG_RIGHT: np.array([0.5, 1, 0]),
        BodyPart.SHOULDERS_LEFT: np.array([0, -1, 0]),
        BodyPart.SHOULDERS_RIGHT: np.array([0, -1, 0]),
        BodyPart.NECK: np.array([0, -1, 0]),
        BodyPart.CHEST: np.array([0, -1, 0]), 
        BodyPart.WAIST: np.array([0, 1, 0]), 
        BodyPart.UPPER_ARM_LEFT: np.array([-1, 0, 0]),
        BodyPart.UPPER_ARM_RIGHT: np.array([1, 0, 0]),
        BodyPart.LOWER_ARM_LEFT: np.array([0, 1, 0]),
        BodyPart.LOWER_ARM_RIGHT: np.array([0, 1, 0]),
        BodyPart.HAND_LEFT: np.array([-1, 0, 0]),
        BodyPart.HAND_RIGHT: np.array([1, 0, 0]),
        BodyPart.LOWER_LEG_LEFT: np.array([-0.5, 1, 0]),
        BodyPart.LOWER_LEG_RIGHT: np.array([0.5, 1, 0]),
        BodyPart.FOOT_LEFT: np.array([-0.5, 1, 0]),
        BodyPart.FOOT_RIGHT: np.array([0.5, 1, 0]),
    }

    # Initialize BFS queue
    queue = deque()
    
    # Assign the torso to cluster 5 if it's not ignored
    torso_cluster = 5
    if torso_cluster in ignored_clusters:
        if verbose:
            print(f"Torso cluster {torso_cluster} is in ignored_clusters. Cannot assign TORSO.")
    elif 0 <= torso_cluster < K:
        cluster_assignments[torso_cluster] = BodyPart.TORSO
        body_part_to_cluster: Dict[BodyPart, int] = {BodyPart.TORSO: torso_cluster}
        assigned_clusters.add(torso_cluster)
        queue.append(BodyPart.TORSO)
        if verbose:
            print(f"Assigned BodyPart.TORSO to cluster {torso_cluster}.")
    else:
        if verbose:
            print(f"Error: Torso cluster index {torso_cluster} is out of bounds.")
        return cluster_assignments  # Return empty assignments
    
    # Build KDTree with the 9D features
    tree = KDTree(combined_features)
    
    while queue:
        current_body_part = queue.popleft()
        current_cluster = body_part_to_cluster[current_body_part]
        current_feature = combined_features[current_cluster]
    
        if verbose:
            print(f"\nProcessing BodyPart.{current_body_part.value} (Cluster {current_cluster})")
    
        # Get child body parts
        child_body_parts = BODY_CONNECTIONS.get(current_body_part, [])
    
        for child_part in child_body_parts:
            if child_part in body_part_to_cluster:
                if verbose:
                    print(f"Body part {child_part.value} is already assigned to cluster {body_part_to_cluster[child_part]}. Skipping.")
                continue  # Already assigned
    
            direction = BODY_PART_DIRECTIONS.get(child_part, np.array([0, 0, 0]))
            if np.linalg.norm(direction) == 0:
                if verbose:
                    print(f"No direction defined for body part {child_part.value}. Skipping assignment.")
                continue  # No direction defined
    
            # Normalize the direction vector
            direction_normalized = direction / np.linalg.norm(direction)
    
            average_distance = threshold
            expected_position = centers_mean[current_cluster] + direction_normalized * average_distance

            query_feature = combined_features[current_cluster].copy()
            query_feature[:3] = expected_position  # Update position
    
            # Query KDTree for nearest neighbors within the threshold
            indices = tree.query_radius([query_feature], r=threshold)[0]
    
            valid_candidates = []
            for idx in indices:
                if idx in assigned_clusters:
                    continue
                vec = centers_mean[idx] - centers_mean[current_cluster]
                if np.dot(vec, direction_normalized) > 0: 
                    distance = np.linalg.norm(vec)
                    valid_candidates.append((idx, distance))
    
            if not valid_candidates:
                if verbose:
                    print(f"No valid candidates found for BodyPart.{child_part.value} from Cluster {current_cluster}.")
                continue
    
            # Select the closest valid candidate
            valid_candidates.sort(key=lambda x: x[1])
            closest_idx, closest_distance = valid_candidates[0]
    
            # Assign the cluster to the body part
            cluster_assignments[closest_idx] = child_part
            body_part_to_cluster[child_part] = closest_idx
            assigned_clusters.add(closest_idx)
            queue.append(child_part)
    
            if verbose:
                print(f"Assigned BodyPart.{child_part.value} to Cluster {closest_idx} (Distance: {closest_distance:.3f}).")
    
    # After BFS, report unassigned clusters
    unassigned_clusters = set(range(K)) - assigned_clusters
    if unassigned_clusters and verbose:
        print(f"\nUnassigned Clusters: {unassigned_clusters}")
    elif verbose:
        print("\nAll clusters have been assigned.")
    
    return cluster_assignments


def validate_assignments(cluster_assignments: Dict[int, BodyPart], required_parts: List[BodyPart]) -> bool:
    """ Validates that all required body parts have been assigned. """
    assigned_parts = set(cluster_assignments.values())
    missing_parts = [part for part in required_parts if part not in assigned_parts]
    
    if missing_parts:
        print("Missing assignments for the following body parts:")
        for part in missing_parts:
            print(f"- {part.value}")
        return False
    return True

def find_skeleton_chains(clustering: Clustering, cluster_assignments: Dict[int, BodyPart]) -> List[List[int]]:
    """ Creates chains based on predefined skeleton connections. """
    chains: List[List[int]] = []
    
    # Define expected connections between body parts
    BODY_CONNECTIONS: Dict[BodyPart, List[BodyPart]] = {
        BodyPart.TORSO: [
            BodyPart.CLAVICLE_LEFT,
            BodyPart.CLAVICLE_RIGHT,
            BodyPart.HEAD,
            BodyPart.UPPER_LEG_LEFT,
            BodyPart.UPPER_LEG_RIGHT
        ],
        BodyPart.CLAVICLE_LEFT: [
            BodyPart.SHOULDERS_LEFT
        ],
        BodyPart.CLAVICLE_RIGHT: [
            BodyPart.SHOULDERS_RIGHT
        ],
        BodyPart.SHOULDERS_LEFT: [
            BodyPart.UPPER_ARM_LEFT
        ],
        BodyPart.SHOULDERS_RIGHT: [
            BodyPart.UPPER_ARM_RIGHT
        ],

        BodyPart.HEAD: [
            BodyPart.NECK
        ],
        BodyPart.NECK: [
            BodyPart.CHEST
        ],
        BodyPart.CHEST: [
            BodyPart.WAIST
        ],
        BodyPart.UPPER_ARM_LEFT: [
            BodyPart.LOWER_ARM_LEFT
        ],
        BodyPart.LOWER_ARM_LEFT: [
            BodyPart.HAND_LEFT
        ],
        BodyPart.UPPER_ARM_RIGHT: [
            BodyPart.LOWER_ARM_RIGHT
        ],
        BodyPart.LOWER_ARM_RIGHT: [
            BodyPart.HAND_RIGHT
        ],
        BodyPart.UPPER_LEG_LEFT: [
            BodyPart.LOWER_LEG_LEFT
        ],
        BodyPart.LOWER_LEG_LEFT: [
            BodyPart.FOOT_LEFT
        ],
        BodyPart.UPPER_LEG_RIGHT: [
            BodyPart.LOWER_LEG_RIGHT
        ],
        BodyPart.LOWER_LEG_RIGHT: [
            BodyPart.FOOT_RIGHT
        ],
        # Add more connections as needed
    }
    
    # Reverse mapping from BodyPart to cluster index
    body_part_to_cluster: Dict[BodyPart, int] = {v: k for k, v in cluster_assignments.items()}
    
    for parent_part, child_parts in BODY_CONNECTIONS.items():
        if parent_part not in body_part_to_cluster:
            print(f"Warning: Parent body part '{parent_part.value}' not assigned to any cluster.")
            continue  # Parent body part not assigned
        parent_cluster = body_part_to_cluster[parent_part]
        for child_part in child_parts:
            if child_part not in body_part_to_cluster:
                print(f"Warning: Child body part '{child_part.value}' not assigned to any cluster.")
                continue  # Child body part not assigned
            child_cluster = body_part_to_cluster[child_part]
            chains.append([parent_cluster, child_cluster])
            print(f"Created chain: {parent_part.value} (Cluster {parent_cluster}) -> {child_part.value} (Cluster {child_cluster})")
    
    return chains

def assign_clusters_to_body_parts_from_file(assignments_filepath: str) -> Dict[int, BodyPart]:
    """
    Loads cluster assignments from a .npz file.

    Args:
        assignments_filepath (str): Path to the .npz file containing cluster assignments.

    Returns:
        Dict[int, BodyPart]: Mapping from cluster index to BodyPart.
    """
    if not os.path.exists(assignments_filepath):
        raise FileNotFoundError(f"Cannot find {assignments_filepath}")

    assign_data = np.load(assignments_filepath)
    cluster_indices = assign_data['cluster_indices']
    body_parts = assign_data['body_parts']

    cluster_assignments = {int(idx): BodyPart(bp) for idx, bp in zip(cluster_indices, body_parts)}
    return cluster_assignments


def compute_joints(clustering: Clustering, cluster_chains: List[List[int]]) -> np.ndarray:
    """ Computes the joint positions as the midpoint between connected clusters in each chain. """
    centers_t = clustering.centers.cpu().numpy()  # Shape: (T, K, 3)
    num_timesteps = centers_t.shape[0]
    num_joints = len(cluster_chains)
    
    joints_t = np.zeros((num_timesteps, num_joints, 3))  # Shape: (T, num_joints, 3)
    
    for joint_idx, (parent_cluster, child_cluster) in enumerate(cluster_chains):
        # Compute midpoints for all timesteps
        joints_t[:, joint_idx, :] = (centers_t[:, parent_cluster, :] + centers_t[:, child_cluster, :]) / 2.0
        print(f"Computed joint {joint_idx} between Cluster {parent_cluster} and Cluster {child_cluster}.")
    
    return joints_t

#----------------------#
#   Core Functions     #
#----------------------#

@torch.no_grad()
def solve(filepath_npz: str, config: Config, save_path: str = None) -> Tuple[Dict[str, torch.Tensor], Clustering]:
    """
    The clustering algorithm (param_filepath, config) -> (scene_data, clustering). 
    """
    
    print(F"Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    params = {key : params[key] for key in Parameters.__dataclass_fields__ if key in params}
    params = Parameters(**params)

    print("Preparing Foreground Features")
    features = Features(params, config)
    fg_features = features.features[features.is_fg]
    fg_features = fg_features.cpu().numpy()

    if config.normalize_features:
        print("Normalizing Foreground Features")
        scaler = StandardScaler()
        fg_features = scaler.fit_transform(fg_features)

    print(F"Clustering Foreground Gaussians") 
    kmeans = KMeans(n_clusters=config.K)
    kmeans.fit(fg_features)
    fg_labels = kmeans.labels_

    ########
    # JOINTS STUFF
    distances = pairwise_distances(fg_features, kmeans.cluster_centers_)
    second_nearest_clusters = distances.argsort(axis=1)[:, 1] # (N,)
    ########

    clustering = Clustering(
        config.K, 
        features, 
        torch.from_numpy(fg_labels).long().to("cuda")
    )

    ### WIP: only animating the cluster centers, baking offsets
    # TODO: Transform rotation around the center instead of just translation.
    if False:
        for c in range(clustering.num_clusters):
            mask = clustering.masks[c]
            # 1) get transformation relative to cluster center, 
            # 2) rotate into cluster coordinate system, 
            # 3) mean over time
            # 4) reproject to world
            
            # positions are now in cluster-centered world-aligned coordinates
            features.pos[:,mask] = features.pos[:,mask] - clustering.centers[:,c].unsqueeze(1)
        
            for t in range(features.T):
                R_wtoc = R.from_quat(clustering.transformations[t,c,[4,5,6,3]]
                                            .cpu()
                                            .numpy()).inv()
                
                # positions are now in cluster-centered cluster-aligned coordinates
                features.pos[t,mask] = \
                    torch.from_numpy((R_wtoc
                                            .apply(features.pos[t,mask]
                                                        .cpu()
                                                        .numpy())))\
                        .float()\
                        .to("cuda")
                
                # rotations are now w.r.t. cluster axes
                features.rot[t,mask] = \
                    torch.from_numpy((R_wtoc * 
                                    (R.from_quat(features.rot[...,[1,2,3,0]][t,mask]
                                                        .reshape((-1,4))
                                                        .cpu()
                                                        .numpy())))
                                    .as_quat())\
                        .float()\
                        .to("cuda") # leave in xyzw format for now
            
            # each gaussian positions/rotations is now averaged over time w.r.t. the cluster coordinates
            features.pos[:,mask] = features.pos[:,mask].mean(dim=0, keepdim=True) 
            features.rot[:,mask] = features.rot[:,mask].mean(dim=0, keepdim=True)

            # transform back to world
            for t in range(features.T):
                R_ctow = R.from_quat(clustering.transformations[t,c,[4,5,6,3]]
                                            .cpu()
                                            .numpy())

                features.pos[t,mask] = clustering.centers[t,c] + \
                    torch.from_numpy(R_ctow.apply(features.pos[t,mask]
                                                        .reshape((-1,3))
                                                        .cpu()
                                                        .numpy()))\
                        .float()\
                        .to("cuda")

                features.rot[t,mask] = \
                    torch.from_numpy((R_ctow * 
                                    R.from_quat(features.rot[...,[1,2,3,0]][t,mask].cpu().numpy()))
                                    .as_quat())\
                        .float()\
                        .to("cuda")[...,[3,0,1,2]] # convert to wxyz format

    # Pass into Grig eventually
    # Grig(...)

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
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((features.T,-1,3))
    elif config.color_mode == "2NN":
        cluster_colors = cmap(second_nearest_clusters)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((features.T,-1,3))
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

    # Save clustering if requested
    if save_path is not None:
        save_scene_and_clustering(scene_data, clustering, save_path)

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
    """Updates the center dots visualization to timestep t."""
    centers_t = clustering.centers[t].cpu()
    
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
    """ Updates the joint dots visualization to the current time step t, projecting them closer to the camera. """
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

def find_nearest_neighbor_chains_all_timesteps(clustering: Clustering, max_distance: float = 0.5) -> List[List[int]]:
    """ 
    Find chains of clusters by considering all timesteps to determine the best adjacent cluster pairs.
    """
    centers_t = clustering.centers.cpu().numpy()       # (T, K, 3)
    dpos_t = clustering.center_dpos.cpu().numpy()            # (T, K, 3)
    drot_t = clustering.center_drot.cpu().numpy()            # (T, K, 3)
    
    T, K, _ = centers_t.shape
    
    # Aggregate features over all timesteps
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

#----------------------#
#      Main Function   #
#----------------------#

def main(filepath_npz: str, config: Config, clustering_filepath: str = "clustering.npz" , save_path: str = "clustering.npz"):
    joint = True
    if clustering_filepath is not None:
        print("Loading clustering from file.")
        scene_data, clustering = load_scene_and_clustering(clustering_filepath)
    else:
        print("Running clustering algorithm.")
        scene_data, clustering = solve(filepath_npz, config, save_path=save_path)

    # Assign clusters to body parts with a defined threshold
    cluster_assignments = assign_clusters_to_body_parts(clustering, scene_data, threshold=0.5)

    # Validate Assignments (Optional)
    required_parts = [
        BodyPart.TORSO, BodyPart.HEAD, BodyPart.NECK, BodyPart.UPPER_ARM_LEFT, BodyPart.UPPER_ARM_RIGHT,
        BodyPart.LOWER_ARM_LEFT, BodyPart.LOWER_ARM_RIGHT, BodyPart.HAND_LEFT, BodyPart.HAND_RIGHT,
        BodyPart.UPPER_LEG_LEFT, BodyPart.UPPER_LEG_RIGHT, BodyPart.LOWER_LEG_LEFT, BodyPart.LOWER_LEG_RIGHT,
        BodyPart.FOOT_LEFT, BodyPart.FOOT_RIGHT, BodyPart.CHEST, BodyPart.WAIST, BodyPart.SHOULDERS_LEFT,
        BodyPart.SHOULDERS_RIGHT, BodyPart.CLAVICLE_LEFT, BodyPart.CLAVICLE_RIGHT
    ]
    if not validate_assignments(cluster_assignments, required_parts):
        print("Warning: Not all required body parts have been assigned.")

    # Find skeleton-based chains based on assignments
    cluster_chains = find_skeleton_chains(clustering, cluster_assignments)

    # Compute joints based on skeleton chains
    joints_t = compute_joints(clustering, cluster_chains)

    # Proceed with visualization as before
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
    #     Evan's Joint     #
    #----------------------#
    # Add joint initialization
    if joint: 
        # Use cluster_chains from skeleton assignment
        # joints_t already computed
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

        update_cluster_centers(t, clustering, w2c, cluster_center_dots, vis)

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

#----------------------#
#   Execution Block    #
#----------------------#
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
