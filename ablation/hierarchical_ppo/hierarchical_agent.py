import os
import random
from itertools import chain
from typing import TypedDict, Optional

import torch

from ablation.hierarchical_ppo.hierarchical_actor_network import HierarchicalActorNetwork
from config import config
from environment.state import ExtractedState
from model.actor_critic import CriticNetwork
from model.encoder_net import EncodingNetwork


class EncodedFeatureDict(TypedDict):
    ego: torch.Tensor
    gnss: torch.Tensor
    # imu: torch.Tensor
    traffic_light: torch.Tensor
    lane: torch.Tensor
    nearest_vehicle: torch.Tensor
    vehicles: Optional[torch.Tensor]


class HierarchicalAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ego_encoder = EncodingNetwork(
            config.encoding_layers, config.ego_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)  # ego

        self.gnss_encoder = EncodingNetwork(
            config.encoding_layers, config.gnss_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)  # gnss

        # self.imu_encoder = EncodingNetwork(
        #     config.encoding_layers, config.imu_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        # ).to(self.device)  # imu

        self.vehicle_encoder = EncodingNetwork(
            config.encoding_layers, config.vehicle_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)  # vehicle

        self.tl_encoder = EncodingNetwork(
            config.encoding_layers, config.tl_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)  # tl

        self.lane_encoder = EncodingNetwork(
            config.encoding_layers, config.lane_feature_dim, config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)  # lane

        self.actor = HierarchicalActorNetwork().to(self.device)
        self.old_actor = HierarchicalActorNetwork.create_from_existing(another_net=self.actor).to(self.device)
        self.critic = CriticNetwork(input_dim=config.low_input_dim+(6 * config.node_feature_dim)).to(self.device)

        self._actor_param_list = list(chain(
            self.ego_encoder.parameters(),
            self.gnss_encoder.parameters(),
            # self.imu_encoder.parameters(),
            self.vehicle_encoder.parameters(),
            self.tl_encoder.parameters(),
            self.lane_encoder.parameters(),
            self.actor.parameters(),
        ))

        self.optimizer_actor = torch.optim.Adam(
            self._actor_param_list,
            lr=config.actor_lr, betas=(0.9, 0.999), eps=1e-8
        )

        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr, betas=(0.9, 0.999), eps=1e-8
        )

        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_actor,
            T_max=config.episodes,
            eta_min=config.actor_lr * config.eta_min_factor
        )

        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_critic,
            T_max=config.episodes,
            eta_min=config.critic_lr * config.eta_min_factor
        )

    def encoding(self, extracted_state: ExtractedState) -> EncodedFeatureDict:
        device = self.device

        ego_v = extracted_state.ego_vector.to(device).unsqueeze(0)
        gnss_v = extracted_state.gnss_vector.to(device).unsqueeze(0)
        # imu_v = extracted_state.imu_vector.to(device).unsqueeze(0)
        tl_v = extracted_state.tl_vector.to(device).unsqueeze(0)
        lane_v = extracted_state.lane_vector.to(device).unsqueeze(0)
        nearest_v = extracted_state.nearest_vector.to(device).unsqueeze(0)

        ego_feat = self.ego_encoder(ego_v)
        gnss_feat = self.gnss_encoder(gnss_v)
        # imu_feat = self.imu_encoder(imu_v)
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
            # "imu": imu_feat,
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

    def select_action(self,
                      global_feat: torch.Tensor,
                      obs: torch.Tensor,
                      yaw_err,
                      v_err,
                      dt: float = 1.0 / config.default_fps
                      ):
        self.actor.eval()
        with torch.no_grad():
            action, info = self.actor(
                global_feat=global_feat, obs=obs, yaw_err=yaw_err, v_err=v_err, dt=dt
            )

        log_p_old_high, ent_old_high, log_p_old_low, ent_old_low = self.old_actor.evaluate(
            global_feat=global_feat,
            obs=obs,
            dir_idx=info["dir_idx"],
            spd_idx=info["spd_idx"],
            low_squashed=info["low_squashed"],
        )

        out = {
            "dir_idx": info["dir_idx"],
            "spd_idx": info["spd_idx"],
            "low_squashed": info["low_squashed"],
            "log_p_old_high": log_p_old_high.detach(),
            "log_p_old_low": log_p_old_low.detach()
        }

        if torch.is_tensor(action):
            a = action.squeeze(0).cpu().numpy()
            action = tuple(map(float, a))

        return action, out

    def evaluate_state(self, state: torch.Tensor):
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state)
        return value

    def update_actor(self,
                     global_feat: torch.Tensor,
                     obs: torch.Tensor,
                     dir_idx: torch.Tensor,
                     spd_idx: torch.Tensor,
                     low_squashed: torch.Tensor,
                     advantages: torch.Tensor,
                     epsilon: float,
                     device: torch.device,
                     log_p_old_high: torch.Tensor,
                     log_p_old_low: torch.Tensor,
                     entropy_coef: float = 0.01,
                     ):
        self.actor.train()

        log_p_new_high, entropy_high, log_p_new_low, entropy_low = self.actor.evaluate(
            global_feat=global_feat,
            obs=obs,
            dir_idx=dir_idx,
            spd_idx=spd_idx,
            low_squashed=low_squashed,
        )

        log_p_old = (log_p_old_high + log_p_old_low).to(device)
        log_p_new = (log_p_new_high + log_p_new_low).to(device)

        log_ratio = (log_p_new - log_p_old).float()
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        valid = torch.isfinite(ratio) & torch.isfinite(advantages)
        if not valid.any():
            approx_kl = (log_p_old - log_p_new).mean().detach()
            return {"approx_kl": approx_kl.item()}

        ratio = ratio[valid]
        adv = advantages[valid].to(device)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则
        entropy = (entropy_high + entropy_low).mean()
        loss = policy_loss - entropy_coef * entropy

        self.optimizer_actor.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     list(self.actor.parameters()), config.max_grad_norm
        # )
        torch.nn.utils.clip_grad_norm_(self._actor_param_list, config.max_grad_norm)
        self.optimizer_actor.step()

        return {
            "loss_actor": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clip_frac": ((ratio > 1.0 + epsilon) | (ratio < 1.0 - epsilon)).float().mean().item(),
        }

    def update_critic(self,
                      states: torch.Tensor,
                      rewards: torch.Tensor,
                      next_states: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float,
                      device: torch.device,
                      value_clip: float = 0.2,
                      v_old: Optional[torch.Tensor] = None
                      ):

        self.critic.train()

        values = self.critic(states.to(device))

        with torch.no_grad():
            next_v = self.critic(next_states.to(device))
            targets = rewards.to(device) + gamma * (1.0 - dones.to(device)) * next_v

        if value_clip is not None:
            v_old = v_old.to(device).detach() if v_old is not None else values.detach()
            v_clipped = v_old + (values - v_old).clamp(-value_clip, value_clip)
            # 价值裁剪
            loss_un_clipped = (values - targets).pow(2)
            loss_clipped = (v_clipped - targets).pow(2)
            value_loss = torch.max(loss_un_clipped, loss_clipped).mean()
        else:
            value_loss = torch.nn.functional.mse_loss(values, targets)

        self.optimizer_critic.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
        self.optimizer_critic.step()

        return {
            "loss_critic": value_loss.item(),
            "value_mean": values.mean().item(),
            "v_tgt_mean": targets.mean().item(),
        }

    def update_all_net(self,
                       batch_trajectories,
                       batch_advantages
                       ):
        device = self.device

        global_feat = torch.stack([t["global_feat"] for t in batch_trajectories]).to(device).to(torch.float32)
        obs = torch.stack([t["obs"] for t in batch_trajectories]).to(device).to(torch.float32)
        dir_idx = torch.stack([t["dir_idx"] for t in batch_trajectories]).to(device).long().squeeze(-1)
        spd_idx = torch.stack([t["spd_idx"] for t in batch_trajectories]).to(device).long().squeeze(-1)

        low_squashed = torch.stack([t["low_squashed"] for t in batch_trajectories]).to(device)

        log_p_old_high = torch.stack([t["log_p_old_high"] for t in batch_trajectories]).to(device).to(torch.float32)
        log_p_old_low = torch.stack([t["log_p_old_low"] for t in batch_trajectories]).to(device).to(torch.float32)

        advantages = torch.as_tensor(batch_advantages, dtype=torch.float32, device=device)

        states_for_critic = torch.stack([t["critic_state"] for t in batch_trajectories]).to(device).to(torch.float32)
        next_states_for_critic = torch.stack([t["next_critic_state"] for t in batch_trajectories]).to(device).to(
            torch.float32)
        rewards = torch.as_tensor([t["reward"] for t in batch_trajectories], dtype=torch.float32, device=device)
        dones = torch.as_tensor([t["done"] for t in batch_trajectories], dtype=torch.float32, device=device)
        v_old = torch.stack([t["v_old"] for t in batch_trajectories]).to(device).to(torch.float32)

        actor_stat = self.update_actor(
            global_feat=global_feat,
            obs=obs,
            dir_idx=dir_idx,
            spd_idx=spd_idx,
            low_squashed=low_squashed,
            advantages=advantages,
            epsilon=config.epsilon,
            device=device,
            log_p_old_high=log_p_old_high,
            log_p_old_low=log_p_old_low,
            entropy_coef=getattr(config, "entropy_coef", 0.01),
        )

        critic_stat = self.update_critic(
            states=states_for_critic,
            rewards=rewards,
            next_states=next_states_for_critic,
            dones=dones,
            gamma=config.gamma,
            device=device,
            value_clip=getattr(config, "value_clip", 0.2),
            v_old=v_old,
        )

        return actor_stat, critic_stat

    def update_old_policy(self):
        self.old_actor.copy_parameter(self.actor)

    def state_dict(self):
        return {
            "ego_encoder": self.ego_encoder.state_dict(),
            "gnss_encoder": self.gnss_encoder.state_dict(),
            # "imu_encoder": self.imu_encoder.state_dict(),
            "vehicle_encoder": self.vehicle_encoder.state_dict(),
            "tl_encoder": self.tl_encoder.state_dict(),
            "lane_encoder": self.lane_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "old_actor": self.old_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }

    def load_state_dict(self, payload: dict):
        self.ego_encoder.load_state_dict(payload["ego_encoder"])
        self.gnss_encoder.load_state_dict(payload["gnss_encoder"])
        # self.imu_encoder.load_state_dict(payload["imu_encoder"])
        self.vehicle_encoder.load_state_dict(payload["vehicle_encoder"])
        self.tl_encoder.load_state_dict(payload["tl_encoder"])
        self.lane_encoder.load_state_dict(payload["lane_encoder"])
        self.actor.load_state_dict(payload["actor"])
        self.old_actor.load_state_dict(payload["old_actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer_actor.load_state_dict(payload["optimizer_actor"])
        self.optimizer_critic.load_state_dict(payload["optimizer_critic"])

    def save_checkpoint(self, episode: int, metrics: dict, filename: str):
        save_dir = config.hierarchical_ppo_save_path
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        payload = {
            "episode": episode,
            "metrics": metrics,
            "agent": self.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state().tolist(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        torch.save(payload, path)
        print(f"[Checkpoint] saved to: {path}")

    def load_checkpoint(self, filename: str) -> int:
        save_dir = config.hierarchical_ppo_save_path
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["agent"])
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"]["python"])
            torch.set_rng_state(torch.tensor(ckpt["rng_state"]["torch"], dtype=torch.uint8))
        print(f"[Checkpoint] loaded from: {path}")
        return ckpt.get("episode", 0)
