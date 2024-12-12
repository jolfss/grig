import numpy as np
import torch
from matplotlib.colors import Colormap
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances
from clustering import Clustering
from features import Features


def indecision(
        features : Features, 
        clustering : Clustering, 
        colormap : Colormap = cm.get_cmap("inferno")
    ):

    distances = pairwise_distances(features[features.is_fg], kmeans.cluster_centers_)
    distances_indices = distances.argsort(axis=1)

    # Compute initial minimum indices before applying the penalty
    initial_min_indices = distances.argsort(axis=1)[:, 0]

    # Calculate the initial minimum and second minimum distances
    sorted_distances = distances.copy()
    sorted_indices = distances_indices.copy()

    # The initial minimum values and indices
    initial_min_values = sorted_distances[np.arange(distances.shape[0]), sorted_indices[:, 0]]
    initial_second_min_values = sorted_distances[np.arange(distances.shape[0]), sorted_indices[:, 1]]

    # Compute the factor that would cause a flip
    flip_factors = initial_second_min_values / initial_min_values

    percentile_5 = np.percentile(flip_factors, 5)
    percentile_95 = np.percentile(flip_factors, 95)

    # Clip the values to the 5th and 95th percentiles
    flip_factors_clipped = np.clip(flip_factors, percentile_5, percentile_95)

    # Normalize to the range [0, 1] and then reverse and cube
    flip_factors_normalized = (1 - ((flip_factors_clipped - percentile_5) / (percentile_95 - percentile_5))) ** 3

    # Apply the colormap using the normalized values
    _feature_colors = colormap(flip_factors_normalized)[:, :3]  
    feature_colors = torch.from_numpy(_feature_colors).float().to("cuda").expand((features.T, -1, 3))