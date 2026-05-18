import torch
import torch.nn as nn

from config import config


class SimplePID:
    def __init__(self, kp=0.6, ki=0.0, kd=0.1, clamp=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.clamp = clamp

    def step_once(self, e: float):
        u = self.kp * e
        lo, hi = self.clamp
        return max(lo, min(hi, u))


class ResidualBCActor(nn.Module):

    def __init__(self,
                 input_dim: int = 6 * config.node_feature_dim,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 alpha: float = config.alpha,
                 residual_scale=tuple(config.residual_scale),
                 action_clamp_steer=tuple(config.action_clamp_steer),
                 action_clamp_throttle=tuple(config.action_clamp_throttle),
                 action_clamp_brake=tuple(config.action_clamp_brake),
                 ):
        super().__init__()

        self.steer_min, self.steer_max = action_clamp_steer
        self.thr_min, self.thr_max = action_clamp_throttle
        self.brk_min, self.brk_max = action_clamp_brake

        self.alpha = alpha
        self.register_buffer("residual_scale",
                             torch.tensor(residual_scale, dtype=torch.float32).view(1, 3))

        num_layers = max(num_layers, 2)
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        self.residual_head = nn.Linear(hidden_dim, 3)

        self._steer_pid = SimplePID(kp=0.8, ki=0.0, kd=0.1, clamp=(-1.0, 1.0))
        self._speed_pid = SimplePID(kp=0.4, ki=0.0, kd=0.0, clamp=(-1.0, 1.0))

    @torch.no_grad()
    def template_action(self,
                        yaw_err: torch.Tensor,
                        v_err: torch.Tensor):
        yaw_err = yaw_err.view(-1)
        v_err = v_err.view(-1)
        B = yaw_err.shape[0]

        steer_t_list = []
        accel_t_list = []
        for b in range(B):
            st = self._steer_pid.step_once(float(yaw_err[b].item()))
            ac = self._speed_pid.step_once(float(v_err[b].item()))
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
                yaw_err: torch.Tensor,
                v_err: torch.Tensor):
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

        B = global_feat.shape[0]

        h = self.trunk(global_feat)
        r_raw = self.residual_head(h)
        r = torch.tanh(r_raw)

        # 模板动作
        template = self.template_action(yaw_err, v_err)

        # 残差缩放并叠加
        res_scaled = r * self.residual_scale
        act = template + self.alpha * res_scaled

        steer = act[:, 0].clamp(self.steer_min, self.steer_max)
        throttle = act[:, 1].clamp(self.thr_min, self.thr_max)
        brake = act[:, 2].clamp(self.brk_min, self.brk_max)
        act = torch.stack([steer, throttle, brake], dim=-1)

        return act, r
