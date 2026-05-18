from __future__ import annotations

import os
import time
from typing import Tuple

import torch

from config import config
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState, RawState
from baseline.sac.sac_agent import SacAgent
from evaluation.utils import estimate_errors, compute_errors_from_raw


class SACEvalPolicy:
    def __init__(self, ckpt_path: str, device: torch.device | None = None):
        # 初始化 agent
        self.agent = SacAgent()
        self.device = device or self.agent.device

        # 加载权重
        self.agent.load(ckpt_path)

        for name in ["actor", "q1", "q2", "q1_target", "q2_target",
                     "ego_encoder", "gnss_encoder", "vehicle_encoder", "tl_encoder", "lane_encoder"]:
            m = getattr(self.agent, name, None)
            if m is not None:
                m.eval()

    def act(self, env: CarlaEnv):
        device = self.device
        raw_state = env.raw_obs

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) GNN 编码 + 全局特征
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats)  # (1, state_dim)

        # 3) 误差估计
        yaw_err, v_err = estimate_errors(env, device)

        # 4) SAC 策略选择动作
        with torch.no_grad():
            action_tensor, _info = self.agent.select_action(
                global_feat=global_feat,
                yaw_err=yaw_err,
                v_err=v_err,
                eval_mode=True,
            )

        if isinstance(action_tensor, torch.Tensor):
            action_tensor = action_tensor.view(-1)
            action = action_tensor.detach().cpu().numpy().tolist()
        else:
            action = list(action_tensor)

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        return action

    def predict(self, raw_state: RawState) -> Tuple[float, float, float]:
        device = self.device

        t0 = time.perf_counter()

        # 1) RawState -> ExtractedState
        es = ExtractedState.from_raw(raw_state, device=device)

        # 2) 编码 + 全局特征
        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats)  # (1, state_dim)

        # 3) 误差
        yaw_err, v_err = compute_errors_from_raw(raw_state, device)

        # 4) SAC 策略选择动作
        with torch.no_grad():
            action_tensor, _info = self.agent.select_action(
                global_feat=global_feat,
                yaw_err=yaw_err,
                v_err=v_err,
                eval_mode=True,
            )

        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.view(-1).detach().cpu().tolist()
        else:
            action = list(action_tensor)

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        return steer, throttle, brake


def run_eval_sac(
        episodes: int = 5,
        no_rendering_mode: bool = True,
        npc_vehicle_num: int = 10,
        ckpt_filename: str = "sac_template_best.pt",
):
    # 1) 环境
    env = CarlaEnv(no_rendering_mode=no_rendering_mode,
                   npc_vehicle_num=npc_vehicle_num)

    # 2) 加载策略
    if os.path.isabs(ckpt_filename):
        ckpt_path = ckpt_filename
    else:
        ckpt_path = os.path.join(config.sac_save_path, ckpt_filename)

    policy = SACEvalPolicy(ckpt_path)

    # 3) 多 episode 评估
    for ep in range(episodes):
        raw = env.reset()
        ep_ret = 0.0
        steps = 0

        while True:
            action = policy.act(env)

            raw, reward, done, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

            if done:
                print(
                    f"[SAC+Template Eval] Episode {ep} done | "
                    f"steps={steps} | return={ep_ret:.2f} | "
                    f"reason={info.get('done_reason', 'unknown')}"
                )
                break

    print("[SAC+Template Eval] 所有 episode 评估完成")


if __name__ == "__main__":
    run_eval_sac(
        episodes=5,
        no_rendering_mode=True,
        npc_vehicle_num=10,
        ckpt_filename="sac_template_best.pt",
    )
