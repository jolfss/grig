"""Encapsulation of a clustering over a gaussian splat."""
from features import Features
import torch

class Clustering:
    num_clusters : int
    """The number of clusters in this clustering."""
    
    features : Features
    """The features of the clustered gaussians."""

    labels : torch.Tensor
    """The cluster ids/labels of the gaussians.
    [0,num_clusters) if part of a cluster, -1 if background
    `(N,)@cuda long`"""

    masks : torch.Tensor
    """Boolean masks for each of the clusters.
    `(num_clusters,N)@cuda bool`"""

    centers : torch.Tensor
    """The (x,y,z) centers for each of the clusters for each timestep.
    `(T,num_clusters,3)@cpu float`"""

    transformations : torch.Tensor
    """The transformations of each cluster represented as [x,y,z,qw,qx,qy,qz].
    `(T,num_clusters,7)@cuda float`"""

    def __init__(self, num_clusters:int, features:Features, labels:torch.Tensor):
        """
        TODO: docs
        """
        self.num_clusters = num_clusters
        self.features = features
        
        self.labels = torch.zeros((features.N),device="cuda").long() - 1
        self.labels[features.is_fg] = labels

        self.masks = torch.zeros((num_clusters, features.N), device="cuda").long()
        self.centers = torch.zeros((features.T, num_clusters, 3), device="cpu")
        self.transformations = torch.zeros((features.T,num_clusters,7))
        for c in range(num_clusters):
            mask = (self.labels==c)
            self.masks[c,:] = mask
            # NOTE: assumes no empty clusters
            self.centers[:,c,:] = (features.pos * mask.unsqueeze(-1)).sum(dim=1) / mask.sum() 
            self.transformations[:,c,:] = (torch.cat((features.pos,features.rot),dim=-1) * mask.unsqueeze(-1)).sum(dim=1) / mask.sum() 

        
