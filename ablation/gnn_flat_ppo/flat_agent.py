import os
import random
from itertools import chain
from typing import TypedDict, Optional

import torch
from torch_geometric.data import HeteroData

from ablation.gnn_flat_ppo.flat_actor_network import FlatActorNetwork
from config import config
from environment.state import ExtractedState, RawState
from model.encoder_net import EncodingNetwork
from model.gcn import HeteroGCN
from model.actor_critic import CriticNetwork


class EncodedFeatureDict(TypedDict):
    ego: torch.Tensor
    gnss: torch.Tensor
    # imu: torch.Tensor
    traffic_light: torch.Tensor
    lane: torch.Tensor
    nearest_vehicle: torch.Tensor
    vehicles: Optional[torch.Tensor]


class AgentFlat:

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

        # self.imu_encoder = EncodingNetwork(
        #     config.encoding_layers, config.imu_feature_dim,
        #     config.encoding_hidden_dim, config.node_feature_dim
        # ).to(self.device)

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

        self.gcn = HeteroGCN(
            config.node_feature_dim, config.gcn_hidden_dim, config.gcn_out_dim, config.gcn_num_layers
        ).to(self.device)

        self.actor = FlatActorNetwork(
            input_dim=config.gcn_out_dim,
            hidden_dim=config.high_hidden_dim,
            num_layers=config.high_layers,
            use_exploration=config.use_exploration,
            soft_exclusive=getattr(config, "soft_exclusive", True),
            action_clamp_steer=tuple(config.action_clamp_steer),
            action_clamp_throttle=tuple(config.action_clamp_throttle),
            action_clamp_brake=tuple(config.action_clamp_brake),
        ).to(self.device)

        self.old_actor = FlatActorNetwork(
            input_dim=config.gcn_out_dim,
            hidden_dim=config.high_hidden_dim,
            num_layers=config.high_layers,
            use_exploration=config.use_exploration,
            soft_exclusive=getattr(config, "soft_exclusive", True),
            action_clamp_steer=tuple(config.action_clamp_steer),
            action_clamp_throttle=tuple(config.action_clamp_throttle),
            action_clamp_brake=tuple(config.action_clamp_brake),
        ).to(self.device)
        self.old_actor.copy_parameter(self.actor)

        self.critic = CriticNetwork(
            input_dim=config.gcn_out_dim,
            num_layers=config.critic_num_layers,
            hidden_dim=config.critic_hidden_dim,
            output_dim=config.critic_output_dim,
            activation=config.critic_activation,
            use_layer_norm=config.critic_use_layer_norm,
            use_residual=config.critic_use_residual,
            dropout=config.critic_dropout,
            final_init_std=config.critic_final_init_std,
        ).to(self.device)

        self._actor_param_list = list(chain(
            self.ego_encoder.parameters(),
            self.gnss_encoder.parameters(),
            # self.imu_encoder.parameters(),
            self.vehicle_encoder.parameters(),
            self.tl_encoder.parameters(),
            self.lane_encoder.parameters(),
            self.gcn.parameters(),
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

    def build_hetero_graph(self, feats: EncodedFeatureDict, raw_state: RawState) -> HeteroData:
        device = self.device
        data = HeteroData()

        data["ego"].x = feats["ego"].to(device)
        data["gnss"].x = feats["gnss"].to(device)
        data["traffic_light"].x = feats["traffic_light"].to(device)
        data["lane"].x = feats["lane"].to(device)

        has_vehicles = feats["vehicles"] is not None
        if has_vehicles:
            vehicles_x = feats["vehicles"].to(device)
            N = vehicles_x.size(0)
            data["vehicle"].x = vehicles_x
        else:
            N = 0

        def add_self_loop(ntype: str, num_nodes: int):
            if num_nodes > 0:
                idx = torch.arange(num_nodes, dtype=torch.long, device=device)
                data[(ntype, "self", ntype)].edge_index = torch.stack([idx, idx], dim=0)

        add_self_loop("ego", 1)
        add_self_loop("gnss", 1)
        add_self_loop("traffic_light", 1)
        add_self_loop("lane", 1)
        if has_vehicles:
            add_self_loop("vehicle", N)

        data[("ego", "to", "lane")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        data[("lane", "to", "ego")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

        data[("ego", "to", "traffic_light")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        data[("traffic_light", "to", "ego")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

        data[("ego", "to", "gnss")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        data[("gnss", "to", "ego")].edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

        if has_vehicles and N > 0:
            src = torch.zeros(N, dtype=torch.long, device=device)
            dst = torch.arange(N, dtype=torch.long, device=device)
            data[("ego", "to", "vehicle")].edge_index = torch.stack([src, dst], dim=0)
            data[("vehicle", "to", "ego")].edge_index = torch.stack([dst, src], dim=0)

            sv = getattr(raw_state, "surrounding_vehicles", None)
            veh_list = sv.vehicles if (sv and getattr(sv, "vehicles", None)) else []
            M = min(N, len(veh_list))

            if M > 0:
                rel_xy = torch.tensor(
                    [[veh_list[i].relative_x, veh_list[i].relative_y] for i in range(M)],
                    dtype=torch.float32, device=device
                )
                diff = rel_xy.unsqueeze(1) - rel_xy.unsqueeze(0)
                dists = torch.norm(diff, dim=-1)

                k = 3
                k_eff = min(k + 1, M)
                knn_idx = torch.topk(-dists, k=k_eff, dim=-1).indices

                src_list, dst_list = [], []
                for i in range(M):
                    neigh = knn_idx[i].tolist()
                    neigh = [j for j in neigh if j != i][:k]
                    if len(neigh) == 0:
                        continue
                    src_list.append(torch.full((len(neigh),), i, dtype=torch.long, device=device))
                    dst_list.append(torch.tensor(neigh, dtype=torch.long, device=device))

                if len(src_list) > 0:
                    src_all = torch.cat(src_list, dim=0)
                    dst_all = torch.cat(dst_list, dim=0)
                    data[("vehicle", "to", "vehicle")].edge_index = torch.stack([src_all, dst_all], dim=0)
                else:
                    data[("vehicle", "to", "vehicle")].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                data[("vehicle", "to", "vehicle")].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        return data

    def get_global_feat(self, feats: EncodedFeatureDict, raw_state: RawState) -> torch.Tensor:
        graph = self.build_hetero_graph(feats, raw_state)
        global_feat = self.gcn(graph)
        return global_feat

    @torch.no_grad()
    def select_action(self,
                      global_feat: torch.Tensor,
                      yaw_err: Optional[torch.Tensor] = None,
                      v_err: Optional[torch.Tensor] = None,
                      dt: float = 1 / config.default_fps):
        act, info = self.actor.forward(
            global_feat,
            yaw_err=yaw_err,
            v_err=v_err,
            dt=dt,
        )
        log_p_old, _entropy_old = self.old_actor.evaluate(
            global_feat=global_feat,
            squashed=info["squashed"],
        )
        info["log_p_old"] = log_p_old.detach()
        action_np = act.squeeze(0).cpu().numpy()
        return tuple(map(float, action_np)), info

    def evaluate_state(self, state: torch.Tensor):
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state.to(self.device))
        return value

    def update_actor(self,
                     global_feat: torch.Tensor,
                     squashed: torch.Tensor,
                     advantages: torch.Tensor,
                     epsilon: float,
                     device: torch.device,
                     log_p_old: torch.Tensor,
                     entropy_coef: float = 0.01,
                     ):
        self.actor.train()

        log_p_new, entropy = self.actor.evaluate(
            global_feat=global_feat,
            squashed=squashed,
        )

        log_p_old = log_p_old.to(device)
        log_p_new = log_p_new.to(device)
        advantages = advantages.to(device)

        log_ratio = (log_p_new - log_p_old).float()
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        valid = torch.isfinite(ratio) & torch.isfinite(advantages)
        if not valid.any():
            approx_kl = (log_p_old - log_p_new).mean().detach()
            return {"approx_kl": approx_kl.item()}

        ratio = ratio[valid]
        adv = advantages[valid]

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = entropy.mean()
        loss = policy_loss - entropy_coef * entropy

        self.optimizer_actor.zero_grad(set_to_none=True)
        loss.backward()
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
                       batch_advantages):

        device = self.device

        global_feat = torch.stack([t["global_feat"] for t in batch_trajectories]).to(device).float()
        squashed = torch.stack([t["squashed"] for t in batch_trajectories]).to(device).float()
        log_p_old = torch.stack([t["log_p_old"] for t in batch_trajectories]).to(device).float()

        advantages = torch.as_tensor(batch_advantages, dtype=torch.float32, device=device)

        states_for_critic = torch.stack([t["critic_state"] for t in batch_trajectories]).to(device).float()
        next_states_for_critic = torch.stack([t["next_critic_state"] for t in batch_trajectories]).to(device).float()
        rewards = torch.as_tensor([t["reward"] for t in batch_trajectories],
                                  dtype=torch.float32, device=device)
        dones = torch.as_tensor([t["done"] for t in batch_trajectories],
                                dtype=torch.float32, device=device)
        v_old = torch.stack([t["v_old"] for t in batch_trajectories]).to(device).float()

        actor_stat = self.update_actor(
            global_feat=global_feat,
            squashed=squashed,
            advantages=advantages,
            epsilon=config.epsilon,
            device=device,
            log_p_old=log_p_old,
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
        self.old_actor.copy_parameter(self.actor, device=self.device)

    def state_dict(self):
        return {
            "ego_encoder": self.ego_encoder.state_dict(),
            "gnss_encoder": self.gnss_encoder.state_dict(),
            "vehicle_encoder": self.vehicle_encoder.state_dict(),
            "tl_encoder": self.tl_encoder.state_dict(),
            "lane_encoder": self.lane_encoder.state_dict(),
            "gcn": self.gcn.state_dict(),
            "actor": self.actor.state_dict(),
            "old_actor": self.old_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }

    def load_state_dict(self, payload: dict):
        self.ego_encoder.load_state_dict(payload["ego_encoder"])
        self.gnss_encoder.load_state_dict(payload["gnss_encoder"])
        self.vehicle_encoder.load_state_dict(payload["vehicle_encoder"])
        self.tl_encoder.load_state_dict(payload["tl_encoder"])
        self.lane_encoder.load_state_dict(payload["lane_encoder"])
        self.gcn.load_state_dict(payload["gcn"])
        self.actor.load_state_dict(payload["actor"])
        self.old_actor.load_state_dict(payload["old_actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer_actor.load_state_dict(payload["optimizer_actor"])
        self.optimizer_critic.load_state_dict(payload["optimizer_critic"])

    def save_checkpoint(self, episode: int, metrics: dict, filename: str):
        save_dir = config.gnn_flat_ppo_save_path
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
            },
        }
        torch.save(payload, path)
        print(f"[Checkpoint] saved to: {path}")

    def load_checkpoint(self, filename: str) -> int:
        save_dir = config.gnn_flat_ppo_save_path
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["agent"])
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"]["python"])
            torch.set_rng_state(torch.tensor(ckpt["rng_state"]["torch"], dtype=torch.uint8))
        print(f"[Checkpoint] loaded from: {path}")
        return ckpt.get("episode", 0)
