
"""Clustering module."""
from typing import Tuple
import numpy as np
import torch

"""-----------------------------------------------------------------------------------
One instant of the dynamic gaussian scene.
All quantities below have an outer dimension of (N,) over the N gaussians.
All tensors are on the CUDA device.
--------------------------------------------------------------------------------------
scales : Tensor #[sx,sy,sz], values typically seem \in [0,1].
--------------------------------------------------------------------------------------
positions : Tensor  #[x,y,z], y-vertical
--------------------------------------------------------------------------------------
rotations : Tensor #quaternion? [x,y,z,w]?, TODO: not sure about convention here
--------------------------------------------------------------------------------------
colors : Tensor # [r,g,b] NOTE: values \in (-inf?,inf?) but "intended" values are [0,1]
                # values < 0 are black
                # value == 1 has peak intensity in the center of the gaussian
                # values >> 1 limits to a solid ellipse of peak intensity
                # (values seen ranging in practice from ~250 to ~250).
--------------------------------------------------------------------------------------
opacities : Tensor # opacities \in [0,1]
-----------------------------------------------------------------------------------"""

#NOTE modified from [load_scene_data] in ./Dynamic3DGaussians/visualize.py line 49
@torch.no_grad()
def load(seq, exp):
    params = dict(np.load(F"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

    features = {
        "position" : params["means3D"],
        "position_dt": torch.gradient(params["means3D"], dim=0)[0],
        "rotation_dt": torch.gradient(torch.nn.functional.normalize(params["unnorm_rotations"]), dim=0)[0]
    }

    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        scene_data.append(rendervar)
    return scene_data, features

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def prepare_data(features, t=70):
    # Extract derivatives

    position = torch.flatten(features['position'][::150], start_dim=0, end_dim=1)
    position_dt = torch.flatten(features['position_dt'][::150], start_dim=0, end_dim=1)
    rotation_dt = torch.flatten(features['rotation_dt'][::150], start_dim=0, end_dim=1)

    # Concatenate to form the 10-dimensional data
    data = torch.cat([position, position_dt, rotation_dt], dim=1)
    return data


# Function to plot the elbow graph
def plot_elbow_graph(seq, exp, max_clusters=10, stride=1):
    _, dt = load(seq, exp)
    data = prepare_data(dt)
    # Convert to NumPy array
    data_np = data.cpu().numpy()

    # Step 2: Preprocess the Data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)

    # Step 3: Elbow Method for optimal K
    wcss = []
    for i in range(1, max_clusters + 1, stride):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
        print(f'{i} Clusters: WCSS = {kmeans.inertia_}')

    # Step 4: Automatically detect the elbow using the second derivative
    first_derivative = np.diff(wcss)
    second_derivative = np.diff(first_derivative)
    elbow_index = np.argmin(second_derivative) + 1

    # Step 5: Plot the elbow graph with the detected elbow point
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, max_clusters + 1, stride), wcss, marker='o', label='WCSS')
    # plt.axvline(x=elbow_index, color='red', linestyle='--', label='Elbow Point')
    # plt.title('Elbow Method for Optimal K')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS (Inertia)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    print(f"Optimal number of clusters (Elbow Point): {elbow_index}")
    return elbow_index




# Modified get_colors function to include elbow plotting
def get_colors(seq, exp, num_clusters = 30):
    _, dt = load(seq, exp)
    data = prepare_data(dt)

    # Convert to NumPy array
    data_np = data.cpu().numpy()

    # Step 2: Preprocess the Data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)

    # Step 3: Clustering
    num_clusters = num_clusters  # Adjust based on elbow method result if desired
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Generate colors based on clusters
    import matplotlib.cm as cm

    # Get a colormap (e.g., 'viridis', 'plasma', 'tab10')
    cmap = cm.get_cmap('tab20', num_clusters)

    # Map cluster labels to colors
    colors = cmap(clusters)  # Returns an array of RGBA values with shape (N, 4)

    # Discard the alpha channel to get RGB
    colors_rgb = colors[:, :3]  # Shape (N, 3)

    # Convert to Tensor
    colors_tensor = torch.from_numpy(colors_rgb).float().to("cuda")

    return colors_tensor