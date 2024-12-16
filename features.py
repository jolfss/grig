from typing import List, Optional
import torch
import torch.nn.functional as F
from config import Config
from parameters import Parameters

def smooth(tensor: torch.Tensor, timestride: int) -> torch.Tensor:
    """
    Smooths the input tensor by averaging every `timestride` steps along the time dimension.
    The resulting tensor will have a size reduced by `timestride` along the time dimension.

    Args:
        tensor (torch.Tensor): The input tensor to be smoothed, shape should be (T, N, D).
        timestride (int): The factor by which to downsample the tensor along the time dimension.

    Returns:
        torch.Tensor: The smoothed tensor, shape will be (T//timestride, N, D).
    """
    T, N, D = tensor.shape
    # Reshape the tensor to average every timestride steps
    new_T = T // timestride
    tensor = tensor[:new_T * timestride]  # Ensure the length is a multiple of timestride
    tensor = tensor.view(new_T, timestride, N, D)
    smoothed_tensor = tensor.mean(dim=1)
    return smoothed_tensor

class Features:
    """
    The set of features in our working set for clustering/otherwise.
    """

    features : torch.Tensor
    """The output of some function on the input parameters.\n
    `(N,D*T//config.timestride)@cuda float`"""
    
    is_fg : torch.Tensor
    """A boolean mask along gaussians specifying which are foreground and background.\n
    `(N,)@cuda bool`"""
    
    pos : torch.Tensor
    """The centers of the gaussians (x,y,z).
    `(T,N,3)@cuda float`"""
    pos_slice : Optional[List[int]] = None
    """Slice of the `pos` feature in the feature vector or `None` if the `pos` feature is excluded `(config.POS=0)`."""
    
    rot : torch.Tensor
    """The orientations of the gaussians as normalized quaternions (qw,qx,qy,qz)\n
    TODO: Have not fully considered the implications of the double cover of `SO(3)`.\n
    `(T,N,4)@cuda float`"""
    rot_slice : Optional[List[int]] = None
    """Slice of the `rot` feature in the feature vector or `None` if the `rot` feature is excluded `(config.ROT=0)`."""

    dpos_dt : torch.Tensor
    """The time derivative `(/s)` of `pos`.\n
    `(T,N,3)@cuda float`"""
    dpos_slice : Optional[List[int]] = None
    """Slice of the `dpos` feature in the feature vector or `None` if `dpos` feature is excluded `(config.DPOS=0)`."""
    
    drot_dt : torch.Tensor
    """The time derivative `(/s)` of `rot`.\n
    `(T,N,4)@cuda float`"""
    drot_slice : Optional[List[int]] = None
    """Slice of the `drot` feature in the feature vector or `None` if `drot` feature is excluded `(config.DROT=0)`."""

    T : int
    "The number of timesteps."
    N : int
    "The number of gaussians."
    D : int
    "The size of the feature dimension (at each timestep)."


    @torch.no_grad()
    def __init__(self, params : Parameters, config : Config):
        # initialize  
        self.pos = params.means3D
        self.rot = torch.nn.functional.normalize(params.unnorm_rotations, dim=-1)
        self.dpos_dt = torch.gradient(self.pos, spacing=30*config.timestride, dim=0)[0] # NOTE: Spacing is adjustment for 30fps input
        self.drot_dt = torch.gradient(self.rot, spacing=30*config.timestride, dim=0)[0] # Normalize?
        
        # TODO: stress tensor for each gaussian

        # size checks
        T_pos, N_pos, D_pos = self.pos.size()
        T_posdt, N_posdt, D_posdt = self.dpos_dt.size()
        T_rot, N_rot, D_rot = self.rot.size()
        T_rotdt, N_rotdt, D_rotdt = self.drot_dt.size()
        assert T_pos == T_posdt == T_rot == T_rotdt >= 2
        assert N_pos == N_posdt == N_rot == N_rotdt >= 2
        assert D_pos == D_posdt == 3
        assert D_rot == D_rotdt == 4

        self.N = N_pos
        self.T = T_pos
        self.D = 0


        self.features = [] 

        def normalize(t:torch.Tensor):
            return (t - t.mean(dim=0, keepdim=True))/t.std(dim=0,keepdim=True)
        
        if config.POS != 0:
            pos = self.pos
            self.pos_slice = slice(self.D, self.D+self.pos.size(-1))
            self.D += self.pos.size(-1)
            pos = smooth(self.pos, config.timestride) #pos = self.pos[::config.timestride]  
            if config.pre_normalize: 
                pos = normalize(pos)
            pos *= config.POS
            self.features.append(pos)

        if config.ROT != 0:
            rot = self.rot
            self.rot_slice = slice(self.D, self.D+self.rot.size(-1))
            self.D += self.rot.size(-1)
            rot = smooth(self.rot,config.timestride) #self.rot[::config.timestride]
            if config.pre_normalize: 
                rot = normalize(rot)
            rot *= config.ROT
            self.features.append(rot)   

        if config.DPOS != 0:
            dpos_dt = self.dpos_dt
            self.dpos_slice = slice(self.D, self.D+self.dpos_dt.size(-1))
            self.D += self.dpos_dt.size(-1)
            dpos_dt = smooth(self.dpos_dt,config.timestride) #self.dpos_dt[::config.timestride] 
            if config.pre_normalize: 
                dpos_dt = normalize(dpos_dt)
            dpos_dt *= config.DPOS
            self.features.append(dpos_dt)   

        if config.DROT != 0:
            drot_dt = self.drot_dt
            self.drot_slice = slice(self.D, self.D+self.drot_dt.size(-1))
            self.D = self.drot_dt.size(-1)
            drot_dt = smooth(self.drot_dt,config.timestride) #self.drot_dt[::config.timestride]
            if config.pre_normalize: 
                drot_dt = normalize(drot_dt)
            drot_dt *= config.DROT
            self.features.append(drot_dt)

        # prepare raw feature vector 
        self.features = torch.cat(self.features,dim=-1).permute(1, 0, 2).reshape((self.N, -1))  # -1 = feature_dim*T//timestride

        # foreground mask
        self.is_fg=params.seg_colors[:,0] > 0.5
