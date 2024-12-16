#!/home/pcgta/mambaforge/envs/dynamic_gaussians/bin/python
"""This modulescript encapsulates the core functionality of Grig."""
import numpy as np
from run import capture_timestep_offscreen
from config import KMeansConfig, KMedoidsConfig
from tqdm import tqdm

def barycentric_sampler(n):
    list_x = []
    list_y = []
    list_l1 = []
    list_l2 = []
    list_l3 = []

    def pair_to_bary(x, y):
        Ax, Ay = (0, 1)
        Bx, By = (-np.sqrt(3) / 2, -1 / 2)
        Cx, Cy = (np.sqrt(3) / 2, -1 / 2)

        denom = (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)

        l1 = ((Bx - x) * (Cy - y) - (By - y) * (Cx - x)) / denom
        l2 = ((Cx - x) * (Ay - y) - (Cy - y) * (Ax - x)) / denom
        l3 = ((Ax - x) * (By - y) - (Ay - y) * (Bx - x)) / denom

        return l1, l2, l3

    for i in range(n, -1, -1):
        for j in range(i, -1, -1):
            x = np.sqrt(3) * (i - (j / 2) - (n / 2)) / n
            y = (3 / 2) * (j / n - 1 / 3)
            
            l1, l2, l3 = pair_to_bary(x, y)
            list_x.append(x)
            list_y.append(y)
            list_l1.append(l1)
            list_l2.append(l2)
            list_l3.append(l3)

    return zip(list_x, list_y, list_l1, list_l2, list_l3)

if __name__ == "__main__":
    K=28
    for x,y, POS,DPOS,DROT in tqdm(barycentric_sampler(10)):

        base_config = \
        {
            "timestride":1,
            "POS" :  POS,      
            "DPOS":  DPOS,
            "DROT":  DROT,
            "ROT" :  0, # ROT seems like a bad feature even though it contains some information
            "pre_normalize":True,
            "remove_bg":True,
            "color_mode":"CLUSTERS",
            "arborescence":False,
            "show_adjacency":True,
            "use_cluster_transforms":False,
        }

        for config in \
        [
            KMeansConfig(K=K,**base_config),
            #KMedoidsConfig(K=K, sample_size=2000,samplings=10,**base_config)
        ]:\
        {
            capture_timestep_offscreen(F"./output/pretrained/softball/params.npz", config,F"{x}_{y}_POS-{POS}_DPOS-{DPOS}_DROT-{DROT}", timestep=60)
            
        }
            

    

