from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    frame_bgr: np.ndarray
    mask: np.ndarray