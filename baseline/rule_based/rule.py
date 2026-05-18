from typing import Tuple

import torch

from config import config


class SimplePID:

    def __init__(self, kp=0.6, ki=0.0, kd=0.1, clamp=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ei = 0.0          # 积分项
        self.prev = None       # 上一时刻误差
        self.clamp = clamp

    def reset(self):
        self.ei = 0.0
        self.prev = None

    def step(self, e: float, dt: float) -> float:
        ed = 0.0 if self.prev is None else (e - self.prev) / max(dt, 1e-3)
        self.ei += e * dt
        u = self.kp * e + self.ki * self.ei + self.kd * ed
        self.prev = e
        lo, hi = self.clamp
        return max(lo, min(hi, u))


class PIDRulePolicy:

    def __init__(self,
                 steer_pid_kp=0.8, steer_pid_ki=0.0, steer_pid_kd=0.1,
                 speed_pid_kp=0.4, speed_pid_ki=0.05, speed_pid_kd=0.0,
                 action_clamp_steer=None,
                 action_clamp_throttle=None,
                 action_clamp_brake=None):
        self.steer_min, self.steer_max = (action_clamp_steer
                                          if action_clamp_steer is not None
                                          else config.action_clamp_steer)
        self.thr_min, self.thr_max = (action_clamp_throttle
                                      if action_clamp_throttle is not None
                                      else config.action_clamp_throttle)
        self.brk_min, self.brk_max = (action_clamp_brake
                                      if action_clamp_brake is not None
                                      else config.action_clamp_brake)

        # 两个 PID 控制器
        self._steer_pid = SimplePID(
            kp=steer_pid_kp, ki=steer_pid_ki, kd=steer_pid_kd,
            clamp=(-1.0, 1.0)
        )
        self._speed_pid = SimplePID(
            kp=speed_pid_kp, ki=speed_pid_ki, kd=speed_pid_kd,
            clamp=(-1.0, 1.0)
        )

    def reset(self):
        self._steer_pid.reset()
        self._speed_pid.reset()

    def act(self,
            yaw_err: torch.Tensor,
            v_err: torch.Tensor,
            dt: float = 1.0 / config.default_fps) -> Tuple[float, float, float]:
        # 转成标量
        if torch.is_tensor(yaw_err):
            yaw_e = float(yaw_err.view(-1)[0].item())
        else:
            yaw_e = float(yaw_err)

        if torch.is_tensor(v_err):
            v_e = float(v_err.view(-1)[0].item())
        else:
            v_e = float(v_err)

        # 方向控制
        steer_t = self._steer_pid.step(yaw_e, dt)
        steer_t = max(self.steer_min, min(self.steer_max, steer_t))

        # 速度控制
        accel_t = self._speed_pid.step(v_e, dt)
        accel_t = max(-1.0, min(1.0, accel_t))

        throttle_t = max(0.0, accel_t)
        brake_t = max(0.0, -accel_t)

        throttle_t = max(self.thr_min, min(self.thr_max, throttle_t))
        brake_t = max(self.brk_min, min(self.brk_max, brake_t))

        return float(steer_t), float(throttle_t), float(brake_t)
