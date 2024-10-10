import torch
from dataclasses import dataclass

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
