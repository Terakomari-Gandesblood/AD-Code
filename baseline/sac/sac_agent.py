import os
from itertools import chain
from typing import TypedDict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from baseline.sac.sac_actor import SacActorNetwork
from baseline.sac.sac_buffer import ReplayBuffer
from baseline.sac.sac_q import QNetwork
from config import config
from model.encoder_net import EncodingNetwork


class EncodedFeatureDict(TypedDict):
    ego: torch.Tensor
    gnss: torch.Tensor
    traffic_light: torch.Tensor
    lane: torch.Tensor
    nearest_vehicle: torch.Tensor
    vehicles: Optional[torch.Tensor]


class SacAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ego_encoder = EncodingNetwork(
            config.encoding_layers, config.ego_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.gnss_encoder = EncodingNetwork(
            config.encoding_layers, config.gnss_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.vehicle_encoder = EncodingNetwork(
            config.encoding_layers, config.vehicle_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.tl_encoder = EncodingNetwork(
            config.encoding_layers, config.tl_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.lane_encoder = EncodingNetwork(
            config.encoding_layers, config.lane_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.actor = SacActorNetwork(
            input_dim=6 * config.node_feature_dim,
            use_exploration=True,
        ).to(self.device)

        state_dim = 6 * config.node_feature_dim
        action_dim = 3

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim=config.sac_hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim=config.sac_hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim=config.sac_hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim=config.sac_hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_params = list(chain(
            self.ego_encoder.parameters(),
            self.gnss_encoder.parameters(),
            self.vehicle_encoder.parameters(),
            self.tl_encoder.parameters(),
            self.lane_encoder.parameters(),
            self.actor.parameters(),
        ))
        self.actor_opt = optim.Adam(self.actor_params, lr=config.sac_actor_lr)

        self.q1_opt = optim.Adam(self.q1.parameters(), lr=config.sac_q_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=config.sac_q_lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=config.sac_alpha_lr)
        self.target_entropy = -float(action_dim)

        self.replay_buffer = ReplayBuffer(capacity=config.sac_buffer_size)

    def encoding(self, extracted_state) -> EncodedFeatureDict:
        device = self.device

        ego_v = extracted_state.ego_vector.to(device).unsqueeze(0)
        gnss_v = extracted_state.gnss_vector.to(device).unsqueeze(0)
        tl_v = extracted_state.tl_vector.to(device).unsqueeze(0)
        lane_v = extracted_state.lane_vector.to(device).unsqueeze(0)
        nearest_v = extracted_state.nearest_vector.to(device).unsqueeze(0)

        ego_feat = self.ego_encoder(ego_v)
        gnss_feat = self.gnss_encoder(gnss_v)
        tl_feat = self.tl_encoder(tl_v)
        lane_feat = self.lane_encoder(lane_v)
        nearest_feat = self.vehicle_encoder(nearest_v)

        if extracted_state.vehicles_tensor is not None:
            vehicles_tensor = extracted_state.vehicles_tensor.to(device)
            vehicles_feat = self.vehicle_encoder(vehicles_tensor)
        else:
            vehicles_feat = None

        return {
            "ego": ego_feat,
            "gnss": gnss_feat,
            "traffic_light": tl_feat,
            "lane": lane_feat,
            "nearest_vehicle": nearest_feat,
            "vehicles": vehicles_feat,
        }

    def get_global_feat(self, feats: EncodedFeatureDict) -> torch.Tensor:
        device = self.device

        ego = feats["ego"]
        gnss = feats["gnss"]
        tl = feats["traffic_light"]
        lane = feats["lane"]
        nearest = feats["nearest_vehicle"]

        parts = [ego, gnss, tl, lane, nearest]

        veh = feats["vehicles"]
        if veh is not None and veh.numel() > 0:
            veh_mean = veh.mean(dim=0, keepdim=True)
        else:
            veh_mean = torch.zeros_like(ego, device=device)

        parts.append(veh_mean)

        global_feat = torch.cat(parts, dim=-1)
        return global_feat

    @torch.no_grad()
    def select_action(self,
                      global_feat: torch.Tensor,
                      yaw_err: torch.Tensor,
                      v_err: torch.Tensor,
                      eval_mode: bool = False):

        self.actor.eval()
        self.actor.use_exploration = not eval_mode

        act, info = self.actor(
            global_feat=global_feat,
            yaw_err=yaw_err,
            v_err=v_err,
            dt=1.0 / config.default_fps,
        )

        action_tensor = act.squeeze(0)
        return action_tensor, info

    def update_sac(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return {}

        device = self.device
        (state, action, reward, next_state, done,
         yaw_err, v_err, next_yaw_err, next_v_err) = self.replay_buffer.sample(batch_size, device)

        with torch.no_grad():
            next_act, next_info = self.actor(
                global_feat=next_state,
                yaw_err=next_yaw_err,
                v_err=next_v_err,
                dt=1.0 / config.default_fps,
            )
            dist_next = next_info["dist"]
            squashed_next = next_info["squashed"]
            logp_next = self.actor.log_prob_squashed(
                dist_next, squashed_next
            ).unsqueeze(-1)

            q1_next = self.q1_target(next_state, next_act).unsqueeze(-1)
            q2_next = self.q2_target(next_state, next_act).unsqueeze(-1)
            q_next_min = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp()
            target_q = reward + (1.0 - done) * config.gamma * (q_next_min - alpha * logp_next)

        q1 = self.q1(state, action).unsqueeze(-1)
        q2 = self.q2(state, action).unsqueeze(-1)

        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        new_act, new_info = self.actor(
            global_feat=state,
            yaw_err=yaw_err,
            v_err=v_err,
            dt=1.0 / config.default_fps,
        )
        dist = new_info["dist"]
        squashed = new_info["squashed"]
        logp = self.actor.log_prob_squashed(dist, squashed).unsqueeze(-1)

        q1_pi = self.q1(state, new_act).unsqueeze(-1)
        q2_pi = self.q2(state, new_act).unsqueeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, config.max_grad_norm)
        self.actor_opt.step()

        with torch.no_grad():
            logp_detach = logp.detach()

        alpha_loss = -(self.log_alpha * (logp_detach + self.target_entropy)).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update(self.q1, self.q1_target, tau=config.sac_tau)
        self.soft_update(self.q2, self.q2_target, tau=config.sac_tau)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }

    @staticmethod
    def soft_update(source: nn.Module, target: nn.Module, tau: float):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

    def state_dict(self):
        return {
            "ego_encoder": self.ego_encoder.state_dict(),
            "gnss_encoder": self.gnss_encoder.state_dict(),
            "vehicle_encoder": self.vehicle_encoder.state_dict(),
            "tl_encoder": self.tl_encoder.state_dict(),
            "lane_encoder": self.lane_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, payload: dict):
        self.ego_encoder.load_state_dict(payload["ego_encoder"])
        self.gnss_encoder.load_state_dict(payload["gnss_encoder"])
        self.vehicle_encoder.load_state_dict(payload["vehicle_encoder"])
        self.tl_encoder.load_state_dict(payload["tl_encoder"])
        self.lane_encoder.load_state_dict(payload["lane_encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.q1_opt.load_state_dict(payload["q1_opt"])
        self.q2_opt.load_state_dict(payload["q2_opt"])
        self.alpha_opt.load_state_dict(payload["alpha_opt"])

    def load(self, filename: str, save_dir: Optional[str] = None):
        if save_dir is None:
            save_dir = config.sac_save_path

        path = os.path.join(save_dir, filename)
        payload = torch.load(path, map_location=self.device)

        self.load_state_dict(payload)

        print(f"[SAC][Checkpoint] loaded from: {path}")
