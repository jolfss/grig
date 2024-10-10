"""Clustering module."""
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

#NOTE Still using the sklearn algo, which is not the best ... Will customize if necessary :)
def plot_elbow_graph(filepath_npz, max_clusters = 10, stride= 1):
    """
    Plots the elbow graph to determine the optimal number of clusters, after removing the background.
    
    Parameters:
    - filepath_npz: Path to the .npz file containing scene data.
    - max_clusters: Maximum number of clusters to test for.
    - stride: Step size for testing different numbers of clusters.

    Returns:
    - Optimal number of clusters determined using the elbow method.
    """

    # print(f"[DEBUG]: Max_cluster: {max_clusters}, stride: {stride}")

    print(f"[grig]: Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

    pos = params["means3D"]
    rot = torch.nn.functional.normalize(params["unnorm_rotations"], dim=-1)
    dpos_dt = torch.gradient(pos, dim=0)[0]
    drot_dt = torch.gradient(rot, dim=0)[0]

    features = torch.cat((pos, dpos_dt, drot_dt), dim=-1).permute(1, 0, 2).reshape(pos.size(1), -1)
    
    # Remove background
    is_fg = params['seg_colors'][:, 0] > 0.5
    fg_features = features[is_fg]

    print("[grig]: Preprocessing foreground features for Elbow Method")
    scaler = StandardScaler()
    fg_features_scaled = scaler.fit_transform(fg_features.cpu().numpy())

    wcss = []
    for i in range(1, max_clusters + 1, stride):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(fg_features_scaled)
        wcss.append(kmeans.inertia_)
        print(f'{i} [elbow]: Clusters: WCSS = {kmeans.inertia_}')

    first_derivative = np.diff(wcss)
    second_derivative = np.diff(first_derivative)
    elbow_index = np.argmin(second_derivative) + 1

    print(f"Optimal number of clusters (Elbow Point): {elbow_index}")
    
    return elbow_index