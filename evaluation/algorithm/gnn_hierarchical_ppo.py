from __future__ import annotations

import time
from typing import Tuple

import torch

from environment.carla_env import CarlaEnv
from environment.state import ExtractedState, RawState
from evaluation.utils import estimate_errors, make_low_obs_from_raw, compute_errors_from_raw
from model.agent import Agent


class GnnHierarchicalPPOEvalPolicy:
    def __init__(self, ckpt_filename: str, device: torch.device | None = None):
        self.agent = Agent()
        self.device = device or self.agent.device

        self.agent.load_checkpoint(ckpt_filename)

        self.agent.actor.eval()
        self.agent.ego_encoder.eval()
        self.agent.gnss_encoder.eval()
        self.agent.vehicle_encoder.eval()
        self.agent.tl_encoder.eval()
        self.agent.lane_encoder.eval()
        self.agent.gcn.eval()
        self.agent.critic.eval()

    def act(self, env: CarlaEnv) -> Tuple[float, float, float]:
        device = self.device
        raw_state = env.raw_obs

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) 高层：GNN 全局特征
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats, raw_state)  # (1, gcn_out_dim)

        # 3) 低层：原始向量拼接
        low_obs = make_low_obs_from_raw(es, device)  # (1, low_input_dim)

        # 4) 误差估计
        yaw_err, v_err = estimate_errors(env, device)

        # 5) 前向：层次策略
        with torch.no_grad():
            action, _info = self.agent.select_action(
                global_feat=global_feat,
                obs=low_obs,
                yaw_err=yaw_err,
                v_err=v_err,
            )

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        return action  # (steer, throttle, brake)

    def predict(self, raw_state: RawState) -> Tuple[float, float, float]:
        device = self.device

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) 高层：GNN 全局特征
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats, raw_state)  # (1, gcn_out_dim)

        # 3) 低层：原始向量拼接
        low_obs = make_low_obs_from_raw(es, device)  # (1, low_input_dim)

        # 4) 误差（raw 口径）
        yaw_err, v_err = compute_errors_from_raw(raw_state, device)

        # 5) 前向：层次策略
        with torch.no_grad():
            action, _info = self.agent.select_action(
                global_feat=global_feat,
                obs=low_obs,
                yaw_err=yaw_err,
                v_err=v_err,
            )

        if isinstance(action, torch.Tensor):
            action = action.view(-1).detach().cpu().tolist()
        else:
            action = list(action)

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        return steer, throttle, brake


def run_eval(
        episodes: int = 3,
        no_rendering_mode: bool = True,
        npc_vehicle_num: int = 10,
        ckpt_filename: str = "gnn_hierarchical_ppo_best.pt",
):
    # 1) 环境
    env = CarlaEnv(no_rendering_mode=no_rendering_mode, npc_vehicle_num=npc_vehicle_num)

    # 2) 策略
    policy = GnnHierarchicalPPOEvalPolicy(ckpt_filename=ckpt_filename)

    for ep in range(episodes):
        env.reset()
        ep_ret = 0.0
        steps = 0

        while True:
            action = policy.act(env)  # (steer, throttle, brake)

            raw_state, reward, done, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

            if done:
                print(
                    f"[GNN+Hierarchical PPO] Episode {ep} done | "
                    f"steps={steps} | return={ep_ret:.2f} | "
                    f"reason={info.get('done_reason', 'unknown')}"
                )
                break

    print("[GNN+Hierarchical PPO] Eval 完成")


if __name__ == "__main__":
    run_eval(
        episodes=5,
        no_rendering_mode=True,
        npc_vehicle_num=10,
        ckpt_filename="gnn_hierarchical_ppo_best.pt",
    )
