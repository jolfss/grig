"""Clustering module."""
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm


default_params = {
    "K":24,
    "timestride":1,
    "POS":1,
    "DPOS":1,
    "DROT":1,
    "remove_bg":True,
    "normalize_features":False
}

#NOTE modified from [load_scene_data] in visualize.py 
@torch.no_grad()
def cluster(filepath_npz, K:int=26, timestride:int=1, POS:float=1.0, DPOS:float=1.0, DROT:float=1.0, remove_bg:bool=True, normalize_features:bool=True) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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
    print(F"[grig]: Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))

    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    print(F"""\
[grig]: Experiment Parameters:
    K = {K}
    timestride = {timestride}
    POS = {POS}
    DPOS = {DPOS}
    DROT = {DROT} 
    remove_bg = {remove_bg}\
""")

    print(F"[grig]: Preparing Features")
    # prepare features
    pos = params["means3D"]
    rot = torch.nn.functional.normalize(params["unnorm_rotations"], dim=-1)
    dpos_dt = torch.gradient(pos, dim=0)[0] * 30 # NOTE: Adjustment for dt
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
    
    features = torch.cat((
        POS*pos[::timestride], 
        DPOS*dpos_dt[::timestride], 
        DROT*drot_dt[::timestride]), dim=-1).permute(1, 0, 2).reshape((N, -1))  # -1 = feature_dim*T//timestride

    # cull bg
    is_fg = params['seg_colors'][:, 0] > 0.5  
    fg_features = features[is_fg]

    print(F"[grig]: Preprocessing Foreground Features")
    scaler = StandardScaler()
    fg_features = fg_features.cpu().numpy()
    if normalize_features:
        fg_features = scaler.fit_transform(fg_features)

    # cluster
    print(F"[grig]: Clustering Foreground Gaussians")
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(fg_features)
    fg_labels = kmeans.labels_

    # color modes
    cmap = cm.get_cmap('turbo', K)
    #CLUSTER COLORS
    cluster_colors = cmap(fg_labels)[:, :3]  
    feature_colors = torch.from_numpy(cluster_colors).float().to("cuda").expand((150,-1,3))
    #DROTATION COLORS
    #feature_colors = drot_dt[:,:,:3][:,is_fg] #(T,N[is_fg],3)
    #DPOSITION COLORS
    #feature_colors = torch.abs(dpos_dt)[:,:,:3][:,is_fg] #(T,N[is_fg],3)
    #POSITION COLORS
    # feature_colors = torch.abs(pos)[:,:,:3][:,is_fg] #(T,N[is_fg],3)


    colors = params["rgb_colors"]       #NOTE: color is constant w.r.t. time 
    colors[:,is_fg] = feature_colors    

    cluster_labels = torch.zeros((N, 1), device="cuda").long() - 1
    cluster_labels[is_fg] = torch.from_numpy(fg_labels).long().to("cuda").unsqueeze(-1)

    cluster_centers = torch.zeros((T, K, 3))  # T time steps, K clusters, 3 dimensions for position

    for c in range(K):
        mask = (cluster_labels==c)
        cluster_centers[:, c, :] = (pos * mask).sum(dim=1) / mask.sum()
    
    # prepare scene data
    print(F"[grig]: Preparing Scene Data")
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': colors[t], 
            'rotations': rot[t],
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if remove_bg:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}  
        scene_data.append(rendervar)

    return scene_data, features, cluster_labels, cluster_centers


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

#NOTE modified from [load_scene_data] in visualize.py
@torch.no_grad()
def __deprecated__clusterer(filepath_npz:str, K:int, timestride:int) -> Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
    """
    Requires (from .npz)
        means3D (x,y,z) --note y is vertical
        unnorm_rotations (quaternions qx, qy, qz, qw) --not sure if in this convention
        log_scales (-inf,inf)
        rgb_colors (-inf,inf) but [0,1] intended range (<0 black, 1 is peak intensity, >1 limits to solid ellipse of peak intensity)
        logit_opacities (0,inf)
    Returns
        features: (N, T*10) Tensor@cuda [x, y, z, dxdt, dydt, dzdt, dqxdt, dqydt, dqzdt, dqwdt]
        scene_data: Dict[str,Tensor@cuda]
    """
    print(F"[grig]: Opening {filepath_npz}")
    params = dict(np.load(filepath_npz))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

    print(F"[grig]: Preparing Features")


    # prepare features
    pos = params["means3D"]
    rot = torch.nn.functional.normalize(params["unnorm_rotations"],dim=-1)
    dpos_dt = torch.gradient(pos, dim=0)[0]
    drot_dt = torch.gradient(rot ,dim=0)[0]

    # check sizes
    T_pos,  N_pos,  D_pos   = pos.size() 
    T_posdt,N_posdt,D_posdt = dpos_dt.size()
    T_rot,  N_rot,  D_rot   = rot.size()
    T_rotdt,N_rotdt,D_rotdt = drot_dt.size()
    assert T_pos == T_posdt == T_rot == T_rotdt >= 2
    assert N_pos == N_posdt == N_rot == N_rotdt >= 2
    assert D_pos == D_posdt == 3
    assert D_rot == D_rotdt == 4
    N=N_pos
    T=T_pos 
    features = torch.cat((pos[::timestride], dpos_dt[::timestride], drot_dt[::timestride]),dim=-1).permute(1,0,2).reshape((N,-1)) # -1 = feature_dim*T//timestride

    # cluster
    print(F"[grig]: Preprocessing Features")
    scaler = StandardScaler()
    features  = scaler.fit_transform(features.cpu().numpy())

    print(F"[grig]: Clustering")
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(features)

    labels = kmeans.labels_
    
    cmap = cm.get_cmap('turbo', K)
    colors = cmap(labels)[:,:3]
    colors = torch.from_numpy(colors).float().to("cuda")

    # prepare scene data
    print(F"[grig]: Preparing Scene Data")
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D':        params['means3D'][t],
            'colors_precomp': colors,
            'rotations':      rot[t],
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales':    torch.exp(params['log_scales']),
            'means2D':   torch.zeros_like(params['means3D'][0], device="cuda")
        }
        rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)

    return features, scene_data