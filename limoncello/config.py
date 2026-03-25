# config.py

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NucleiConfig:
    tophat_radius: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    spot_sigma: float = 5.0
    outline_sigma: float = 1.0


@dataclass
class NeuriteConfig:
    min_size: int = 50
    spot_sigma: float = 10.0
    outline_sigma: float = 1.0


@dataclass
class BasalBodyConfig:
    spot_sigma: float = 2.0
    outline_sigma: float = 2.0
    gaussian_sigma: Tuple[float, float, float] = (1.0, 1.0, 0.0)

@dataclass
class CiliaConfig:
    classifier_path: str
    min_size: int = 20