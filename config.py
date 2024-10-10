from dataclasses import dataclass
from typing import Union

@dataclass
class BaseConfig:
    color_mode:str
    """Which color mode to use for rendering.
    Options: CLUSTER POS DPOS ROT DROT"""

    remove_bg:bool
    "Whether to render the background gaussians (with their default rbg colors)."

    normalize_features:bool
    "Whether each feature component should be normalized."

    timestride:int
    "How many timesteps to stride over when gathering features."

    POS:float
    "The weighting of the `POS` feature."

    DPOS:float
    "The weighting of the `DPOS` feature."

    DROT:float
    "The weighting of the `DROT` feature."

@dataclass
class KMeansConfig(BaseConfig):
    K:int
    "The number of means to cluster over."

Config = Union[KMeansConfig,]