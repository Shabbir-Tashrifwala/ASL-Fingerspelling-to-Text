
from dataclasses import dataclass
@dataclass
class RTConfig:
    window: int = 64
    smooth_k: int = 5
    commit_thresh: float = 0.60
    commit_patience: int = 3
    mp_max_hands: int = 2
    mp_min_det_conf: float = 0.5
    mp_min_track_conf: float = 0.5
    target_fps: int = 20
