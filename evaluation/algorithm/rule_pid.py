import time
from typing import Tuple, Optional

import torch

from baseline.rule_based.rule import PIDRulePolicy
from config import config
from environment.carla_env import CarlaEnv
from environment.state import RawState
from evaluation.utils import estimate_errors, compute_errors_from_raw


class RulePIDEvalPolicy:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pid = PIDRulePolicy()

    def reset(self):
        self.pid.reset()

    def act(self, env: CarlaEnv) -> Tuple[float, float, float]:
        t0 = time.perf_counter()
        yaw_err, v_err = estimate_errors(env, self.device)
        steer, throttle, brake = self.pid.act(yaw_err, v_err, dt=1.0 / config.default_fps)
        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)
        return steer, throttle, brake

    def predict(self, raw_state: RawState) -> Tuple[float, float, float]:

        t0 = time.perf_counter()

        yaw_err, v_err = compute_errors_from_raw(raw_state, self.device)

        # 兼容：PID 若要标量
        yaw_in = float(yaw_err.item()) if isinstance(yaw_err, torch.Tensor) and yaw_err.numel() == 1 else yaw_err
        v_in = float(v_err.item()) if isinstance(v_err, torch.Tensor) and v_err.numel() == 1 else v_err

        steer, throttle, brake = self.pid.act(yaw_in, v_in, dt=1.0 / config.default_fps)

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        return float(steer), float(throttle), float(brake)
