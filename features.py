import torch
from config import Config
from parameters import Parameters

class Features:
    """
    Representation of the features extracted from the parameter file.
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
    
    rot : torch.Tensor
    """The orientations of the gaussians as normalized quaternions (qw,qx,qy,qz)\n
    TODO: Have not fully considered the implications of the double cover of `SO(3)`.\n
    `(T,N,4)@cuda float`"""

    dpos_dt : torch.Tensor
    """The time derivative `(/s)` of `pos`.\n
    `(T,N,3)@cuda float`"""
    
    drot_dt : torch.Tensor
    """The time derivative `(/s)` of `rot`.\n
    `(T,N,4)@cuda float`"""

    T : int
    "The number of timesteps."
    N : int
    "The number of gaussians."
    D : int
    "The size of the feature dimension (at each timestep)."

    def __init__(self, params : Parameters, config : Config):
        # initialize  
        self.pos = params.means3D
        self.rot = torch.nn.functional.normalize(params.unnorm_rotations, dim=-1)
        self.dpos_dt = torch.gradient(self.pos, spacing=30*config.timestride, dim=0)[0] # NOTE: Spacing is adjustment for 30fps input
        self.drot_dt = torch.gradient(self.rot, spacing=30*config.timestride, dim=0)[0] 

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
        self.D = self.pos.size(-1) + self.dpos_dt.size(-1) + self.drot_dt.size(-1) # TODO: adjust for feature ablation/mod/aug

        # prepare raw feature vector 
        self.features = torch.cat((
            config.POS*self.pos[::config.timestride], 
            config.DPOS*self.dpos_dt[::config.timestride], 
            config.DROT*self.drot_dt[::config.timestride]), dim=-1).permute(1, 0, 2).reshape((self.N, -1))  # -1 = feature_dim*T//timestride

        # foreground mask
        self.is_fg=params.seg_colors[:,0] > 0.5