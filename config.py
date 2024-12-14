from typing import List
from dataclasses import dataclass
from typing import Union

@dataclass
class BaseConfig:
    color_mode:str
    """Which color mode to use for rendering.
    Options: CLUSTER POS DPOS ROT DROT"""

    remove_bg:bool
    "Whether to render the background gaussians (with their default rbg colors)."

    timestride:int
    "How many timesteps to stride over when gathering features."

    pre_normalize:bool
    "Whether or not to normalize (over each component of the input features)."

    show_adjacency:bool
    "Whether or not to spawn a thread to view the adjacency matrix or not."

    arborescence:bool
    "If `true`, solves for the minimum arborescence instead of the minimum spanning tree of the adjacency weight matrix."

    POS:float
    "The weighting of the `POS` feature."

    ROT:float
    "The weighting of the `ROT` feature."

    DPOS:float
    "The weighting of the `DPOS` feature."

    DROT:float
    "The weighting of the `DROT` feature."


@dataclass
class KMeansConfig(BaseConfig):
    K:int
    "The number of means to cluster over."

@dataclass
class KMedoidsConfig(BaseConfig):
    K:int
    "The number of medoids to cluster over."
    sample_size:int
    "The number of gaussians to draw when clustering (?)."
    samplings:int
    "The number of times to recluster (?)."

Config = Union[KMeansConfig,KMedoidsConfig]