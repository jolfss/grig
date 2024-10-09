#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

@dataclass
class Config:
    K:int
    "The number of means to cluster over."
    timestride:int
    "How many time frames should be skipped over."
    POS:float
    "The weighting of the `POS` feature."
    DPOS:float
    "The weighting of the `DPOS` feature."
    DROT:float
    "The weighting of the `DROT` feature."
    remove_bg:bool
    "Whether to render the background gaussians (with their default rbg colors)."
    normalize_features:bool
    "Whether each feature component should be normalized.."
    color_mode:str
    """Which color mode to use for rendering.
    Options:
        CLUSTERS
        POS
        DPOS
        ROT
        DROT
    """

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

class Features:
    """
    Representation of the features extracted from the parameter file.
    """
    features : torch.Tensor
    """The output of some function on the input parameters.\n
    `(N, D*T//config.timestride)@cuda float`"""
    is_fg : torch.Tensor
    """A boolean mask along gaussians specifying which are foreground and background.\n
    `(N, )@cuda bool`"""
    pos : torch.Tensor
    """The centers of the gaussians (x,y,z).
    `(T,N,3)@cuda float`"""
    rot : torch.Tensor
    """The orientations of the gaussians as normalized quaternions (qx,qy,qz,qw)\n
    NOTE: Have not fully considered the implications of the double cover of `SO(3)`.\n
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
        self.is_fg=params.seg_colors[:,0] > 0.5,

    def __getattr__(self, name):
        """Called when the requested attribute or method isn't found in the object."""
        return getattr(self.features, name)
    
    def __getitem__(self, key):
        """Forward item access (slicing or indexing)."""
        return self.features[key]
    
    def __setitem__(self, key, value):
        """Forward item assignment."""
        self.features[key] = value
    
    def __delitem__(self, key):
        """Forward item deletion."""
        del self.features[key]


#NOTE modified from [load_scene_data] in visualize.py 
@torch.no_grad()
def cluster(filepath_npz:str, config:Config) -> Tuple[Dict[str, torch.Tensor], Features, torch.Tensor, torch.Tensor]:
    """
    The primary clustering algorithm `(param_filepath, config) -> (scene_data, features, labels, centers)`. 
    """

    print(F"Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    params = {key : params[key] for key in Parameters.__dataclass_fields__ if key in params}
    params = Parameters(**params)

    print("Preparing Foreground Features")
    features = Features(params, config)
    fg_features = features[features.is_fg]
    fg_features = fg_features.cpu().numpy()

    if config.normalize_features:
        print("Normalizing Foreground Features")
        scaler = StandardScaler()
        fg_features = scaler.fit_transform(fg_features)

    print(F"Clustering Foreground Gaussians")
    kmeans = KMeans(n_clusters=config.K)
    kmeans.fit(fg_features)
    fg_labels = kmeans.labels_

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
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((150,-1,3))
    else: #RGB
        pass

    colors = params.rgb_colors    #NOTE: color is constant w.r.t. time
    if feature_colors is not None:
        colors[:,features.is_fg] = feature_colors 

    print(F"Calculating Centers")
    cluster_labels = torch.zeros((features.N, 1), device="cuda").long() - 1
    cluster_labels[features.is_fg] = torch.from_numpy(fg_labels).long().to("cuda").unsqueeze(-1)
    cluster_centers = torch.zeros((features.T, config.K, 3))  # T time steps, K clusters, 3 dimensions for position
    
    for c in range(config.K):
        mask = (cluster_labels==c)
        cluster_centers[:, c, :] = (features.pos * mask).sum(dim=1) / mask.sum() # NOTE: Assumes no empty clusters
    
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

    return scene_data, features, cluster_labels, cluster_centers
    
if __name__ == "__main__":
    ()
