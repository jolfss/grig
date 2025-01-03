"""Encapsulation of a clustering over a gaussian splat."""
from features import Features
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class Clustering:
    num_clusters : int
    """The number of clusters in this clustering."""

    labels : torch.Tensor
    """The cluster ids/labels of the gaussians; requires indices are assigned starting with the most populated cluster, so index
    0 is the most populated and K-1 is the least. Labels are `[0,num_clusters)` if in a cluster or `-1` if background.
    `(N,)@cuda long`"""

    masks : torch.Tensor
    """Boolean masks for each of the clusters.
    `(num_clusters,N)@cuda bool`"""

    center_pos : torch.Tensor
    """The (x,y,z) centers for each of the clusters for each timestep.
    `(T,num_clusters,3)@cuda float`"""

    center_dpos : torch.Tensor
    """The (dx,dy,dz) center displacements for each of the clusters for each timestep.
    `(T,num_clusters,3)@cuda float`"""

    center_drot : torch.Tensor
    """The (dx,dy,dz) center displacements for each of the clusters for each timestep.
    `(T,num_clusters,3)@cuda float`"""

    transformations : torch.Tensor
    """The transformations of each cluster represented as [x,y,z,qw,qx,qy,qz].
    `(T,num_clusters,7)@cuda float`"""

    @torch.no_grad()
    def __init__(self, num_clusters:int, features:Features, labels:torch.Tensor):
        """
        TODO: docs
        """

        self.num_clusters = num_clusters
        
        self.labels = torch.zeros((features.N),device="cuda").long() - 1
        self.labels[features.is_fg] = labels.to("cuda")

        self.masks = torch.zeros((num_clusters, features.N), device="cuda", dtype=torch.bool)
        self.center_pos = torch.zeros((features.T, num_clusters, 3), device="cuda")
        self.transformations = torch.zeros((features.T,num_clusters,7), device="cuda")
        for c in range(num_clusters):
            mask = (self.labels==c)
            self.masks[c,:] = mask
            self.center_pos[:,c,:] = (features.pos * mask.unsqueeze(-1)).sum(dim=1) / max(mask.sum(),1) 
            self.transformations[:,c,:] = (torch.cat((features.pos,features.rot),dim=-1) * mask.unsqueeze(-1)).sum(dim=1) / mask.sum() 
        
        # assert masks are disjoint and cover
        total = 0
        for i in range(num_clusters):
            for j in range(num_clusters):
                if i==j: continue
                assert (self.masks[i] & self.masks[j]).sum() ==0
            total += self.masks[i].sum() 
        assert total == features.is_fg.sum()

        self.center_dpos = self.compute_velocity(delta_t=1.0)
        self.center_drot = self.compute_angular_velocity(delta_t=1.0)

    def compute_velocity(self, delta_t: float = 1.0) -> torch.Tensor:
        """ Compute velocity (dpos_dt) for cluster centers using finite differences. """
        # Assuming centers has shape (T, K, 3)
        centers_np = self.center_pos.cpu().numpy()  # (T, K, 3)
        velocity_np = np.zeros_like(centers_np)
        velocity_np[:-1] = (centers_np[1:] - centers_np[:-1]) / delta_t
        velocity_np[-1] = velocity_np[-2]  # Replicate last velocity
        velocity_tensor = torch.from_numpy(velocity_np).float().to(self.center_pos.device)
        return velocity_tensor

    def compute_angular_velocity(self, delta_t: float = 1.0) -> torch.Tensor:
        """ Compute angular velocity (drot_dt) for cluster centers using quaternion differences. """
        transformations_np = self.transformations.cpu().numpy() 
        drot_np = np.zeros((transformations_np.shape[0], transformations_np.shape[1], 3))
        
        for t in range(1, transformations_np.shape[0]):
            for k in range(transformations_np.shape[1]):
                q_prev = transformations_np[t-1, k, 3:7]  # [qw, qx, qy, qz]
                q_curr = transformations_np[t, k, 3:7]

                r_prev = R.from_quat(q_prev[[1, 2, 3, 0]])  # [qx, qy, qz, qw]
                r_curr = R.from_quat(q_curr[[1, 2, 3, 0]])
                
                r_delta = r_prev.inv() * r_curr

                axis_angle = r_delta.as_rotvec()  # (3,)

                omega = axis_angle / delta_t
                
                drot_np[t, k, :] = omega

        drot_np[0, :, :] = drot_np[1, :, :]
        
        drot_tensor = torch.from_numpy(drot_np).float().to(self.center_pos.device)
        return drot_tensor
