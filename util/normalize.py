import math
import torch
from dataclasses import dataclass


@dataclass
class FixedRanges:
    pos_lo: float = -200.0
    pos_hi: float = 200.0
    vel_lo: float = -40.0
    vel_hi: float = 40.0
    speed_hi: float = 40.0  # m/s
    dist_hi: float = 80.0  # m
    size_hi: float = 5.0  # m
    lane_w_lo: float = 2.5  # m
    lane_w_hi: float = 6.0  # m


R = FixedRanges()


def clip01(x, lo: float, hi: float):
    """
    将 [lo, hi] 映射到 [0,1]
    """
    if x is None:
        return 0.0

    if isinstance(x, torch.Tensor):
        x = x.clamp(lo, hi)
        return (x - lo) / (hi - lo + 1e-8)
    else:
        x = float(x)
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        return (x - lo) / (hi - lo + 1e-8)


def clip11(x, lo: float, hi: float):
    """
    将 [lo, hi] 映射到 [-1,1]
    """
    if x is None:
        return 0.0
    return 2.0 * clip01(x, lo, hi) - 1.0


def sincos_deg(deg_or_rad: float):
    """
    输入角度(度)或弧度，返回 sin, cos
    """
    if deg_or_rad is None:
        return 0.0, 1.0
    r = deg_or_rad
    if abs(r) > 2 * math.pi:
        r = math.radians(r)
    return math.sin(r), math.cos(r)
