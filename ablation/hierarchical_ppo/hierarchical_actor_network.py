from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from config import config


class HighLevelNet(nn.Module):

    def __init__(self,
                 layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 ):
        super().__init__()
        layers = layers if layers > 1 else 2
        self.layers = layers

        self.top_net = nn.ModuleList()
        self.top_net.append(nn.Linear(input_dim, hidden_dim))
        self.top_net.append(nn.LayerNorm(hidden_dim))
        for _ in range(layers - 2):
            self.top_net.append(nn.Linear(hidden_dim, hidden_dim))
            self.top_net.append(nn.LayerNorm(hidden_dim))

        self.dir_head = nn.Linear(hidden_dim, 3)
        self.spd_head = nn.Linear(hidden_dim, 3)

    def compute_logits(self, global_feat: torch.Tensor):
        x = global_feat
        for layer in range(0, len(self.top_net), 2):
            x = self.top_net[layer](x)
            x = self.top_net[layer + 1](x)
            x = nn.functional.relu(x)

        dir_logits = self.dir_head(x)
        spd_logits = self.spd_head(x)

        return dir_logits, spd_logits

    def forward(self, global_feat: torch.Tensor):
        dir_logits, spd_logits = self.compute_logits(global_feat)

        dir_idx = Categorical(logits=dir_logits).sample()
        spd_idx = Categorical(logits=spd_logits).sample()

        return dir_idx, spd_idx

    def evaluate(self,
                 global_feat: torch.Tensor,
                 dir_idx: torch.Tensor,
                 spd_idx: torch.Tensor):
        dir_logits, spd_logits = self.compute_logits(global_feat)

        dir_dist = Categorical(logits=dir_logits)
        spd_dist = Categorical(logits=spd_logits)

        log_probs = dir_dist.log_prob(dir_idx) + spd_dist.log_prob(spd_idx)
        entropy = (dir_dist.entropy() + spd_dist.entropy())
        return log_probs, entropy


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


class LowLevelNet(nn.Module):

    def __init__(self,
                 layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 use_condition: bool = True,
                 use_residual: bool = True,
                 use_exploration: bool = True,
                 soft_exclusive: bool = True,
                 alpha: float = 0.5,
                 cone_ratio: float = 0.3,
                 cond_dim: int = 16,
                 residual_scale=(0.8, 0.2, 0.2),
                 action_clamp_steer=(-1.0, 1.0),
                 action_clamp_throttle=(0.0, 1.0),
                 action_clamp_brake=(0.0, 1.0),
                 ):
        super().__init__()

        self.use_exploration = use_exploration
        self.use_condition = use_condition
        self.use_residual = use_residual

        self.dir_embed = nn.Embedding(3, cond_dim)
        self.spd_embed = nn.Embedding(3, cond_dim)
        all_cond_dim = cond_dim * 2 if use_condition else 0

        layers = max(layers, 2)
        fcs = [nn.Linear(input_dim + all_cond_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 2):
            fcs.append(nn.Linear(hidden_dim, hidden_dim))
            fcs.append(nn.ReLU())
        self.trunk = nn.Sequential(*fcs)

        self.mu_head = nn.Linear(hidden_dim, 3)
        self.log_std = nn.Parameter(torch.full((3,), -0.5))

        self.register_buffer("residual_scale", torch.tensor(residual_scale).view(1, 3))
        self.register_buffer("action_clamp_steer", torch.tensor(action_clamp_steer))
        self.register_buffer("action_clamp_throttle", torch.tensor(action_clamp_throttle))
        self.register_buffer("action_clamp_brake", torch.tensor(action_clamp_brake))
        self.alpha = alpha
        self.soft_exclusive = soft_exclusive
        self.cone_ratio = cone_ratio
        self.steer_min, self.steer_max = action_clamp_steer
        self.thr_min, self.thr_max = action_clamp_throttle
        self.brk_min, self.brk_max = action_clamp_brake

        self._steer_pid = SimplePID(kp=0.8, ki=0.0, kd=0.1, clamp=(-1.0, 1.0))
        self._speed_pid = SimplePID(kp=0.4, ki=0.05, kd=0.0, clamp=(-1.0, 1.0))

    @staticmethod
    def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _log_prob_squashed(self, dist: Normal, a_squash: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        u = self._atanh(a_squash, eps=eps)
        log_prob_u = dist.log_prob(u).sum(dim=-1)
        log_det = torch.log(1 - a_squash.pow(2) + eps).sum(dim=-1)
        out = log_prob_u - log_det
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @torch.no_grad()
    def template_action(self,
                        yaw_err: torch.Tensor,
                        v_err: torch.Tensor,
                        dt: float) -> torch.Tensor:
        B = yaw_err.shape[0]
        steer_t = []
        accel_t = []
        for b in range(B):
            st = self._steer_pid.step(float(yaw_err[b].item()), dt)
            ac = self._speed_pid.step(float(v_err[b].item()), dt)
            steer_t.append(st)
            accel_t.append(ac)
        steer_t = torch.tensor(steer_t, device=yaw_err.device).clamp(self.steer_min, self.steer_max)
        accel_t = torch.tensor(accel_t, device=yaw_err.device).clamp(-1.0, 1.0)

        throttle_t = torch.clamp(accel_t, 0.0, 1.0)
        brake_t = torch.clamp(-accel_t, 0.0, 1.0)
        return torch.stack([steer_t, throttle_t, brake_t], dim=-1)

    def _cond_vec(self, dir_idx: torch.Tensor, spd_idx: torch.Tensor):
        dir_vec = self.dir_embed(dir_idx.long())
        spd_vec = self.spd_embed(spd_idx.long())
        return torch.cat([dir_vec, spd_vec], dim=-1)

    def _soft_exclusive_tb(self, act: torch.Tensor):
        # 如果油门较大, 轻抑制刹车, 反之亦然
        if not self.soft_exclusive:
            return act
        steer, thr, brk = act.unbind(dim=-1)
        brk = brk * (1.0 - 0.8 * (thr > 0.1).float())
        thr = thr * (1.0 - 0.8 * (brk > 0.1).float())
        return torch.stack([steer, thr, brk], dim=-1)

    def _cone_project(self, act: torch.Tensor, template: torch.Tensor):
        if self.cone_ratio is None:
            return act
        steer, thr, brk = act.unbind(dim=-1)
        steer_t = template[:, 0]
        cone = torch.clamp(torch.abs(steer_t) * self.cone_ratio, 0.0, 1.0)
        steer_adj = torch.where(
            steer_t >= 0.0,
            torch.maximum(steer, -cone),
            torch.minimum(steer, cone)
        )
        return torch.stack([steer_adj, thr, brk], dim=-1)

    def forward(self,
                obs: torch.Tensor,
                dir_idx: torch.Tensor = None,
                spd_idx: torch.Tensor = None,
                yaw_err: torch.Tensor = None,
                v_err: torch.Tensor = None,
                dt: float = 1 / config.default_fps):

        B = obs.shape[0]
        x = obs

        if self.use_condition:
            assert dir_idx is not None and spd_idx is not None, "use_condition=True 需要 dir_idx/spd_idx"
            x = torch.cat([x, self._cond_vec(dir_idx, spd_idx)], dim=-1)

        h = self.trunk(x)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(min=-5.0, max=2.0)
        std = log_std.exp().unsqueeze(0).expand(B, -1)
        dist = Normal(mu, std)

        u = dist.sample() if self.use_exploration else mu
        squashed_act = torch.tanh(u)

        if not self.use_residual:
            steer = squashed_act[:, 0].clamp(self.steer_min, self.steer_max)
            throttle = ((squashed_act[:, 1] + 1.0) / 2.0).clamp(self.thr_min, self.thr_max)
            brake = ((squashed_act[:, 2] + 1.0) / 2.0).clamp(self.brk_min, self.brk_max)
            act = torch.stack([steer, throttle, brake], dim=-1)
            if self.soft_exclusive:
                act = self._soft_exclusive_tb(act)
            return act, dist, squashed_act.detach()

        assert yaw_err is not None and v_err is not None, "use_residual=True 需要提供 yaw_err/v_err"
        template = self.template_action(yaw_err, v_err, dt)

        residual_squash = squashed_act
        res_scaled = residual_squash * self.residual_scale

        act = template + self.alpha * res_scaled

        steer = act[:, 0].clamp(self.steer_min, self.steer_max)
        throttle = act[:, 1].clamp(self.thr_min, self.thr_max)
        brake = act[:, 2].clamp(self.brk_min, self.brk_max)
        act = torch.stack([steer, throttle, brake], dim=-1)

        if self.soft_exclusive:
            act = self._soft_exclusive_tb(act)
        act = self._cone_project(act, template)

        return act, dist, residual_squash.detach()

    def evaluate(self,
                 obs: torch.Tensor,
                 dir_idx: torch.Tensor = None,
                 spd_idx: torch.Tensor = None,
                 low_squashed: torch.Tensor = None
                 ):
        B = obs.shape[0]
        x = obs

        if self.use_condition:
            assert dir_idx is not None and spd_idx is not None, "use_condition=True 需要 dir_idx/spd_idx"
            cond = self._cond_vec(dir_idx, spd_idx)
            x = torch.cat([x, cond], dim=-1)

        h = self.trunk(x)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(min=-5.0, max=2.0)
        std = log_std.exp().unsqueeze(0).expand(B, -1)
        dist = Normal(mu, std)

        assert low_squashed is not None, "evaluate需要残差或最终动作的tanh变量"
        log_prob = self._log_prob_squashed(dist, low_squashed)
        entropy = dist.entropy().sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob, entropy


class HierarchicalActorNetwork(nn.Module):

    def __init__(self,
                 high_layers: int = config.high_layers,
                 high_input_dim: int = 6 * config.node_feature_dim,
                 high_hidden_dim: int = config.high_hidden_dim,
                 low_layers: int = config.low_layers,
                 low_input_dim: int = config.low_input_dim,
                 low_hidden_dim: int = config.low_hidden_dim,
                 use_condition: bool = config.use_condition,
                 use_residual: bool = config.use_residual,
                 use_exploration: bool = config.use_exploration,
                 alpha: float = config.alpha,
                 cone_ratio: float = config.cone_ratio,
                 residual_scale=tuple(config.residual_scale),
                 action_clamp_steer=tuple(config.action_clamp_steer),
                 action_clamp_throttle=tuple(config.action_clamp_throttle),
                 action_clamp_brake=tuple(config.action_clamp_brake),
                 ):
        super().__init__()
        self.high = HighLevelNet(
            layers=high_layers,
            input_dim=high_input_dim,
            hidden_dim=high_hidden_dim
        )
        self.low = LowLevelNet(
            layers=low_layers,
            input_dim=low_input_dim,
            hidden_dim=low_hidden_dim,
            use_condition=use_condition,
            use_residual=use_residual,
            use_exploration=use_exploration,
            soft_exclusive=config.soft_exclusive,
            alpha=alpha,
            cone_ratio=cone_ratio,
            cond_dim=config.cond_dim,
            residual_scale=residual_scale,
            action_clamp_steer=action_clamp_steer,
            action_clamp_throttle=action_clamp_throttle,
            action_clamp_brake=action_clamp_brake,
        )
        self.use_condition = use_condition
        self.use_residual = use_residual
        self.use_exploration = use_exploration

    @torch.no_grad()
    def act(self,
            global_feat: torch.Tensor,
            obs: torch.Tensor,
            yaw_err: torch.Tensor = None,
            v_err: torch.Tensor = None,
            dt: float = 1 / config.default_fps):
        act, _ = self.forward(global_feat, obs, yaw_err=yaw_err, v_err=v_err, dt=dt)
        return act

    def forward(self,
                global_feat: torch.Tensor,
                obs: torch.Tensor,
                yaw_err: torch.Tensor = None,
                v_err: torch.Tensor = None,
                dt: float = 1 / config.default_fps):
        B = global_feat.shape[0]
        dir_idx, spd_idx = self.high(global_feat)

        if self.use_residual:
            if yaw_err is None:
                yaw_err = torch.zeros(B, device=global_feat.device, dtype=global_feat.dtype)
            if v_err is None:
                v_err = torch.zeros(B, device=global_feat.device, dtype=global_feat.dtype)
            act, low_dist, low_squashed = self.low(
                obs=obs,
                dir_idx=dir_idx,
                spd_idx=spd_idx,
                yaw_err=yaw_err,
                v_err=v_err,
                dt=dt,
            )
        else:
            act, low_dist, low_squashed = self.low(
                obs=obs,
                dir_idx=dir_idx if self.use_condition else None,
                spd_idx=spd_idx if self.use_condition else None,
                dt=dt,
            )

        info = {
            "dir_idx": dir_idx,
            "spd_idx": spd_idx,
            "low_dist": low_dist,
            "low_squashed": low_squashed,
        }
        return act, info

    def evaluate(self,
                 global_feat: torch.Tensor,
                 obs: torch.Tensor,
                 dir_idx: torch.Tensor,
                 spd_idx: torch.Tensor,
                 low_squashed: torch.Tensor
                 ):
        log_p_high, entropy_high = self.high.evaluate(global_feat, dir_idx, spd_idx)

        if self.use_condition:
            x_dir, x_spd = dir_idx, spd_idx
        else:
            x_dir, x_spd = None, None

        log_p_low, entropy_low = self.low.evaluate(
            obs=obs, dir_idx=x_dir, spd_idx=x_spd, low_squashed=low_squashed
        )
        return log_p_high, entropy_high, log_p_low, entropy_low

    @torch.no_grad()
    def copy_parameter(self,
                       another_net: 'HierarchicalActorNetwork',
                       strict: bool = True,
                       device: Optional[torch.device] = None) -> None:
        if device is not None:
            self.to(device)

        self.load_state_dict(another_net.state_dict(), strict=strict)

        self.use_condition = another_net.use_condition
        self.use_residual = another_net.use_residual
        self.use_exploration = another_net.use_exploration

        self.low.use_condition = another_net.low.use_condition
        self.low.use_residual = another_net.low.use_residual
        self.low.use_exploration = another_net.low.use_exploration

        self.low.alpha = another_net.low.alpha
        self.low.cone_ratio = another_net.low.cone_ratio

    @classmethod
    def create_from_existing(cls, another_net: 'HierarchicalActorNetwork') -> 'HierarchicalActorNetwork':
        high_first_linear = another_net.high.top_net[0]
        assert isinstance(high_first_linear, nn.Linear)
        high_input_dim = high_first_linear.in_features
        high_hidden_dim = high_first_linear.out_features
        # len(top_net) = 2*(layers-1)  =>  layers = len/2 + 1
        high_layers = len(another_net.high.top_net) // 2 + 1

        low_first_linear = None
        for m in another_net.low.trunk:
            if isinstance(m, nn.Linear):
                low_first_linear = m
                break
        assert low_first_linear is not None

        low_hidden_dim = low_first_linear.out_features
        # cond_dim_eff = config.cond_dim * 2 if another_net.low.use_condition else 0
        # low_input_dim = low_first_linear.in_features - cond_dim_eff
        cond_dim = another_net.low.dir_embed.embedding_dim
        cond_dim_eff = cond_dim * 2 if another_net.low.use_condition else 0
        low_input_dim = low_first_linear.in_features - cond_dim_eff

        low_linear_count = sum(isinstance(m, nn.Linear) for m in another_net.low.trunk)
        low_layers = low_linear_count + 1

        use_condition = another_net.use_condition
        use_residual = another_net.use_residual
        use_exploration = another_net.use_exploration

        alpha = another_net.low.alpha
        cone_ratio = another_net.low.cone_ratio

        residual_scale = tuple(another_net.low.residual_scale.squeeze(0).tolist())
        action_clamp_steer = tuple(another_net.low.action_clamp_steer.tolist())
        action_clamp_throttle = tuple(another_net.low.action_clamp_throttle.tolist())
        action_clamp_brake = tuple(another_net.low.action_clamp_brake.tolist())

        new_net = cls(
            high_layers=high_layers,
            high_input_dim=high_input_dim,
            high_hidden_dim=high_hidden_dim,
            low_layers=low_layers,
            low_input_dim=low_input_dim,
            low_hidden_dim=low_hidden_dim,
            use_condition=use_condition,
            use_residual=use_residual,
            use_exploration=use_exploration,
            alpha=alpha,
            cone_ratio=cone_ratio,
            residual_scale=residual_scale,
            action_clamp_steer=action_clamp_steer,
            action_clamp_throttle=action_clamp_throttle,
            action_clamp_brake=action_clamp_brake,
        )

        new_net.copy_parameter(another_net, strict=True)

        return new_net
