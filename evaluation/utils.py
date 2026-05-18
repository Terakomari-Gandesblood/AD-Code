import math
from typing import Tuple
import torch
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState, RawState


def estimate_errors(env: CarlaEnv, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rs = env.raw_state

    yaw_diff_deg = 0.0
    lane_info = getattr(rs, "lane_info", None)
    if lane_info is not None and hasattr(lane_info, "yaw_diff"):
        yaw_diff_deg = float(lane_info.yaw_diff or 0.0)

    yaw_err = torch.tensor([[math.radians(yaw_diff_deg)]], dtype=torch.float32, device=device)

    v_des = float(env.desired_speed()) if hasattr(env, "desired_speed") else 0.0
    ego_info = getattr(rs, "ego_info", None)
    v_cur = float(getattr(ego_info, "speed", 0.0) or 0.0)
    v_err = torch.tensor([[v_des - v_cur]], dtype=torch.float32, device=device)

    return yaw_err, v_err


def make_low_obs_from_raw(es: ExtractedState, device: torch.device) -> torch.Tensor:
    parts = [
        es.ego_vector,
        es.gnss_vector,
        # es.imu_vector,
        es.tl_vector,
        es.lane_vector,
        es.nearest_vector,
    ]
    obs = torch.cat(parts, dim=-1).to(device)  # [1, low_input_dim]
    obs = obs.view(1, -1)
    return obs


def compute_errors_from_raw(raw: RawState, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    lane_info = raw.lane_info
    ego_info = raw.ego_info

    yaw_diff_deg = float(getattr(lane_info, "yaw_diff", 0.0) or 0.0)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)

    v_des = float(getattr(lane_info, "speed_limit", 0.0) or 0.0)
    v_cur = float(getattr(ego_info, "speed", 0.0) or 0.0)
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=device)

    return yaw_err, v_err
