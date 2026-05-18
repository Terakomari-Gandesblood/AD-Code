from __future__ import annotations

import time
from typing import Tuple

import torch

from ablation.gnn_flat_ppo.flat_agent import AgentFlat
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState, RawState
from evaluation.utils import estimate_errors, compute_errors_from_raw


class GnnFlatPPOEvalPolicy:
    def __init__(self, ckpt_filename: str, device: torch.device | None = None):
        self.agent = AgentFlat()
        self.device = device or self.agent.device

        # 加载训练好的参数
        self.agent.load_checkpoint(ckpt_filename)

        # 评估模式
        for name in ["actor", "old_actor", "critic", "ego_encoder", "gnss_encoder",
                     "vehicle_encoder", "tl_encoder", "lane_encoder", "gcn"]:
            m = getattr(self.agent, name, None)
            if m is not None:
                m.eval()

    def act(self, env: CarlaEnv) -> Tuple[float, float, float]:
        device = self.device
        raw_state = env.raw_obs

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) 编码 + GNN 得到 global_feat
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats, raw_state)  # (1, gcn_out_dim)

        # 3) 估算 yaw_err, v_err
        yaw_err, v_err = estimate_errors(env, device)

        # 4) 前向
        with torch.no_grad():
            action, _info = self.agent.select_action(global_feat, yaw_err, v_err)

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        return action  # (steer, throttle, brake)

    def predict(self, raw_state: RawState) -> Tuple[float, float, float]:
        device = self.device

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) 编码 + GNN 得到 global_feat
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats, raw_state)  # (1, gcn_out_dim)

        # 3) 误差（与训练/离线评估口径一致）
        yaw_err, v_err = compute_errors_from_raw(raw_state, device)  # (1,), (1,)

        # 4) 前向
        with torch.no_grad():
            action, _info = self.agent.select_action(global_feat, yaw_err, v_err)

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
        ckpt_filename: str = "gnn_flat_ppo_best.pt",
):
    # 1) 环境
    env = CarlaEnv(no_rendering_mode=no_rendering_mode, npc_vehicle_num=npc_vehicle_num)

    # 2) 策略
    policy = GnnFlatPPOEvalPolicy(ckpt_filename=ckpt_filename)

    for ep in range(episodes):
        raw_state = env.reset()
        ep_ret = 0.0
        steps = 0

        while True:
            action = policy.act(env)  # (steer, throttle, brake)

            raw_state, reward, done, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

            if done:
                print(
                    f"[GNN+Flat PPO] Episode {ep} done | steps={steps} | "
                    f"return={ep_ret:.2f} | reason={info.get('done_reason', 'unknown')}"
                )
                break

    print("[GNN+Flat PPO] Eval 完成")


if __name__ == "__main__":
    run_eval(
        episodes=5,
        no_rendering_mode=True,
        npc_vehicle_num=10,
        ckpt_filename="gnn_flat_ppo_best.pt",
    )
