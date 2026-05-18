from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from config import config


class SimplePID:

    def __init__(self, kp=0.6, ki=0.0, kd=0.1, clamp=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ei = 0.0
        self.prev = None
        self.clamp = clamp

    def step(self, e: float, dt: float):
        ed = 0.0 if self.prev is None else (e - self.prev) / max(dt, 1e-3)
        self.ei += e * dt
        u = self.kp * e + self.ki * self.ei + self.kd * ed
        self.prev = e
        lo, hi = self.clamp
        return max(lo, min(hi, u))


class PureActorNetwork(nn.Module):

    def __init__(self,
                 input_dim: int = 6 * config.node_feature_dim,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 use_exploration: bool = True,
                 soft_exclusive: bool = True,
                 alpha: float = config.alpha,
                 cone_ratio: float = config.cone_ratio,
                 residual_scale=tuple(config.residual_scale),
                 action_clamp_steer=tuple(config.action_clamp_steer),
                 action_clamp_throttle=tuple(config.action_clamp_throttle),
                 action_clamp_brake=tuple(config.action_clamp_brake),
                 init_log_std: float = -0.5,
                 ):
        super().__init__()
        self.use_exploration = use_exploration
        self.soft_exclusive = soft_exclusive

        self.steer_min, self.steer_max = action_clamp_steer
        self.thr_min, self.thr_max = action_clamp_throttle
        self.brk_min, self.brk_max = action_clamp_brake

        self.alpha = alpha
        self.cone_ratio = cone_ratio
        self.register_buffer("residual_scale",
                             torch.tensor(residual_scale, dtype=torch.float32).view(1, 3))

        num_layers = max(num_layers, 2)
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_dim, 3)

        self.log_std = nn.Parameter(torch.full((3,), init_log_std, dtype=torch.float32))

        self._steer_pid = SimplePID(kp=0.8, ki=0.0, kd=0.1, clamp=(-1.0, 1.0))
        self._speed_pid = SimplePID(kp=0.4, ki=0.05, kd=0.0, clamp=(-1.0, 1.0))

    @staticmethod
    def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def log_prob_squashed(self, dist: Normal, a_squash: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        u = self._atanh(a_squash, eps=eps)
        log_prob_u = dist.log_prob(u).sum(dim=-1)  # (B,)
        log_det = torch.log(1 - a_squash.pow(2) + eps).sum(dim=-1)
        out = log_prob_u - log_det
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _soft_exclusive_tb(self, act: torch.Tensor) -> torch.Tensor:
        if not self.soft_exclusive:
            return act
        steer, thr, brk = act.unbind(dim=-1)
        brk = brk * (1.0 - 0.8 * (thr > 0.1).float())
        thr = thr * (1.0 - 0.8 * (brk > 0.1).float())
        return torch.stack([steer, thr, brk], dim=-1)

    def _cone_project(self, act: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        if self.cone_ratio is None:
            return act

        steer, thr, brk = act.unbind(dim=-1)
        steer_t = template[:, 0]
        cone = torch.clamp(torch.abs(steer_t) * self.cone_ratio, 0.0, 1.0)

        steer_adj = torch.where(
            steer_t >= 0.0,
            torch.maximum(steer, -cone),
            torch.minimum(steer, cone),
        )
        return torch.stack([steer_adj, thr, brk], dim=-1)

    @torch.no_grad()
    def template_action(self,
                        yaw_err: torch.Tensor,
                        v_err: torch.Tensor,
                        dt: float) -> torch.Tensor:

        yaw_err = yaw_err.view(-1)
        v_err = v_err.view(-1)
        B = yaw_err.shape[0]

        steer_t_list = []
        accel_t_list = []
        for b in range(B):
            st = self._steer_pid.step(float(yaw_err[b].item()), dt)
            ac = self._speed_pid.step(float(v_err[b].item()), dt)
            steer_t_list.append(st)
            accel_t_list.append(ac)

        device = yaw_err.device
        steer_t = torch.tensor(steer_t_list, device=device).clamp(self.steer_min, self.steer_max)
        accel_t = torch.tensor(accel_t_list, device=device).clamp(-1.0, 1.0)

        throttle_t = torch.clamp(accel_t, 0.0, 1.0)
        brake_t = torch.clamp(-accel_t, 0.0, 1.0)

        return torch.stack([steer_t, throttle_t, brake_t], dim=-1)

    def forward(self,
                global_feat: torch.Tensor,
                yaw_err: Optional[torch.Tensor] = None,
                v_err: Optional[torch.Tensor] = None,
                dt: float = 1 / config.default_fps,
                ):
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

        B = global_feat.shape[0]

        h = self.trunk(global_feat)
        mu = self.mu_head(h)

        log_std = self.log_std.clamp(-5.0, 2.0)
        std = log_std.exp().unsqueeze(0).expand(B, -1)

        dist = Normal(mu, std)

        if self.use_exploration:
            u = dist.sample()
        else:
            u = mu
        residual_squash = torch.tanh(u)

        if (yaw_err is None) or (v_err is None):
            steer = residual_squash[:, 0].clamp(self.steer_min, self.steer_max)
            throttle = ((residual_squash[:, 1] + 1.0) / 2.0).clamp(self.thr_min, self.thr_max)
            brake = ((residual_squash[:, 2] + 1.0) / 2.0).clamp(self.brk_min, self.brk_max)
            act = torch.stack([steer, throttle, brake], dim=-1)
            if self.soft_exclusive:
                act = self._soft_exclusive_tb(act)
        else:

            template = self.template_action(yaw_err, v_err, dt)  # (B,3)
            res_scaled = residual_squash * self.residual_scale  # (B,3)
            act = template + self.alpha * res_scaled  # (B,3)

            steer = act[:, 0].clamp(self.steer_min, self.steer_max)
            throttle = act[:, 1].clamp(self.thr_min, self.thr_max)
            brake = act[:, 2].clamp(self.brk_min, self.brk_max)
            act = torch.stack([steer, throttle, brake], dim=-1)

            if self.soft_exclusive:
                act = self._soft_exclusive_tb(act)
            act = self._cone_project(act, template)

        info = {
            "dist": dist,
            "squashed": residual_squash,
        }
        return act, info

    def evaluate(self,
                 global_feat: torch.Tensor,
                 squashed: torch.Tensor,
                 ):

        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

        B = global_feat.shape[0]
        h = self.trunk(global_feat)
        mu = self.mu_head(h)

        log_std = self.log_std.clamp(-5.0, 2.0)
        std = log_std.exp().unsqueeze(0).expand(B, -1)

        dist = Normal(mu, std)
        log_prob = self.log_prob_squashed(dist, squashed)
        entropy = dist.entropy().sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

        return log_prob, entropy

    @torch.no_grad()
    def act(self,
            global_feat: torch.Tensor,
            yaw_err: Optional[torch.Tensor] = None,
            v_err: Optional[torch.Tensor] = None,
            dt: float = 1 / config.default_fps):
        self.eval()
        act, info = self.forward(global_feat, yaw_err=yaw_err, v_err=v_err, dt=dt)
        a = act.squeeze(0).cpu().numpy()
        return tuple(map(float, a)), info

    @torch.no_grad()
    def copy_parameter(self,
                       another: 'PureActorNetwork',
                       strict: bool = True,
                       device: Optional[torch.device] = None) -> None:
        if device is not None:
            self.to(device)
        self.load_state_dict(another.state_dict(), strict=strict)

