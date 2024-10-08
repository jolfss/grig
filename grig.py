#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

class ColorMode():
    RGB = 0
    POS = 1
    DPOS = 2
    DROT = 3
    CLUSTERS = 4

@dataclass
class Config:
    K:24
    timestride:1
    POS:1
    DPOS:1
    DROT:1
    remove_bg:True
    normalize_features:False
    color_mode=ColorMode.CLUSTERS

@dataclass
class RenderVar:
    """
    A protocol representing the 'render variables' which dynamic 3d gaussians
    takes as input; this is the representation of the scene. The dimensions referenced
    in the tensor sizes are as follows:\n
        T: timestep
        N: number of gaussians\n
    ? => Unconfirmed statement.
    """
    means3D:torch.Tensor
    """The centers (x,y,z) of the gaussians; y is vertical.\n
    (T,N,3)@cuda float"""
    unnorm_rotations : torch.Tensor 
    """The unnormalized quaternions (qx,qy,qz,qw)? representation of the gaussians'
    rotations.\n
    (T,N,4)@cuda float"""
    log_scales : torch.Tensor 
    """The log scales/multivariances of the gaussians (sx, sy, sz)?\n
    (N,3)?@cuda float"""
    rgb_colors : torch.Tensor 
    """The colors of the gaussians; (-inf,inf) but intended range [0,1].\n
    (<0 black, 1 is peak intensity, >1 limits to solid ellipse of peak intensity)\n
    (N,3)@cuda float"""
    seg_colors : torch.Tensor
    """The segmentation color of the gaussians; used to segment foreground and background natively.
    (N,3)@cuda float?"""
    logit_opacities : torch.Tensor
    """The logits representing the opacities of the gaussians.\n
    (N)?@cuda float""" 

@dataclass
class Features:
    features : torch.Tensor
    is_fg : torch.Tensor
    pos : torch.Tensor
    rot : torch.Tensor
    dpos_dt : torch.Tensor
    drot_dt : torch.Tensor
    T : int
    N : int
    D : int
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


def prepare_and_clean_features(rendervars : RenderVar, config : Config) -> Features:
    pos = rendervars.means3D

    rot = torch.nn.functional.normalize(rendervars.unnorm_rotations, dim=-1)
    dpos_dt = torch.gradient(pos, dim=0)[0] * 30 # NOTE: Adjustment for dt, 30fps
    drot_dt = torch.gradient(rot, dim=0)[0] * 30

    # size checks
    T_pos, N_pos, D_pos = pos.size()
    T_posdt, N_posdt, D_posdt = dpos_dt.size()
    T_rot, N_rot, D_rot = rot.size()
    T_rotdt, N_rotdt, D_rotdt = drot_dt.size()
    assert T_pos == T_posdt == T_rot == T_rotdt >= 2
    assert N_pos == N_posdt == N_rot == N_rotdt >= 2
    assert D_pos == D_posdt == 3
    assert D_rot == D_rotdt == 4
    N = N_pos
    T = T_pos
    D = pos.size(-1) + dpos_dt.size(-1) + drot_dt.size(-1) # TODO: adjust for feature ablation/mod/aug
    
    feature_vec = torch.cat((
        config.POS*pos[::config.timestride], 
        config.DPOS*dpos_dt[::config.timestride], 
        config.DROT*drot_dt[::config.timestride]), dim=-1).permute(1, 0, 2).reshape((N, -1))  # -1 = feature_dim*T//timestride
    
    features = Features(
        features=feature_vec,
        is_fg=rendervars.seg_colors[:,0] > 0.5,
        pos=pos,
        rot=rot,
        dpos_dt=dpos_dt,
        drot_dt=drot_dt,
        T=T,
        N=N,
        D=D
    )

    return features

#NOTE modified from [load_scene_data] in visualize.py 
@torch.no_grad()
def grig_cluster(filepath_npz:str, config:Config) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Requires (from .npz)
        means3D (x,y,z) --note y is vertical
        unnorm_rotations (quaternions qx, qy, qz, qw) --not sure if in this convention
        log_scales (-inf,inf)
        rgb_colors (-inf,inf) but [0,1] intended range (<0 black, 1 is peak intensity, >1 limits to solid ellipse of peak intensity)
        logit_opacities (0,inf)
    Returns
        features: (N, T*10) Tensor@cuda [x, y, z, dxdt, dydt, dzdt, dqxdt, dqydt, dqzdt, dqwdt]:w
        scene_data: Dict[str,Tensor@cuda]
    """
    print(F"Opening {filepath_npz}")
    rendervars = dict(np.load(filepath_npz))
    rendervars = {k: torch.tensor(v).cuda().float() for k, v in rendervars.items()}
    rendervars = {key : rendervars[key] for key in RenderVar.__dataclass_fields__ if key in rendervars}
    rendervars : RenderVar = RenderVar(**rendervars)

    print("Preparing Features")
    features = prepare_and_clean_features(rendervars, config)
    fg_features = features[features.is_fg]

    print(F"Preprocessing Foreground Features")
    scaler = StandardScaler()
    fg_features = fg_features.cpu().numpy()
    if config.normalize_features:
        fg_features = scaler.fit_transform(fg_features)

    print(F"Clustering Foreground Gaussians")
    kmeans = KMeans(n_clusters=config.K)
    kmeans.fit(fg_features)
    fg_labels = kmeans.labels_

    print(F"Calculating Colors")
    cmap = cm.get_cmap('turbo', config.K)
    feature_colors = None
    if config.color_mode == ColorMode.POS:
        feature_colors = features.pos[:,:,:3][:,features.is_fg] #(T,N[is_fg],3)
    elif config.color_mode == ColorMode.DPOS:
        feature_colors = features.dpos_dt[:,:,:3][:,features.is_fg] #(T,N[is_fg],3)
    elif config.color_mode == ColorMode.DROT:
        feature_colors = features.drot_dt[:,:,:3][:,features.is_fg] #(T,N[is_fg],3)
    elif config.color_mode == ColorMode.CLUSTERS:
        cluster_colors = cmap(fg_labels)[:, :3]  
        feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((150,-1,3))
    else: #RGB
        pass

    colors = rendervars.rgb_colors    #NOTE: color is constant w.r.t. time
    if feature_colors is not None:
        colors[:,features.is_fg] = feature_colors 

    print(F"Calculating Centers")
    cluster_labels = torch.zeros((features.N, 1), device="cuda").long() - 1
    cluster_labels[features.is_fg] = torch.from_numpy(fg_labels).long().to("cuda").unsqueeze(-1)
    cluster_centers = torch.zeros((features.T, config.K, 3))  # T time steps, K clusters, 3 dimensions for position

    for c in range(config.K):
        mask = (cluster_labels==c)
        cluster_centers[:, c, :] = (features.pos * mask).sum(dim=1) / mask.sum()
    
    print(F"Preparing Scene Data")
    scene_data = []
    for t in range(len(features.pos)):
        rendervar = {
            'means3D': features.pos[t],
            'colors_precomp': colors[t], 
            'rotations': features.rot[t],
            'opacities': torch.sigmoid(rendervars.logit_opacities),
            'scales': torch.exp(rendervars.log_scales),
            'means2D': torch.zeros_like(rendervars.means3D[0], device="cuda")
        }
        if config.remove_bg:
            rendervar = {k: v[features.is_fg] for k, v in rendervar.items()}  
        scene_data.append(rendervar)

    return scene_data, features, cluster_labels, cluster_centers
    
if __name__ == "__main__":
    ()
