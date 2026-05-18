# ablation/pure_pid/run_pid_baseline.py
import math
import time
from typing import Tuple

import torch

from baseline.rule_based.rule import PIDRulePolicy
from config import config
from environment.carla_env import CarlaEnv


def estimate_errors(env: CarlaEnv, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rs = env.raw_state

    # 方向误差
    yaw_diff_deg = 0.0
    if getattr(rs, "lane_info", None) is not None and hasattr(rs.lane_info, "yaw_diff"):
        yaw_diff_deg = float(rs.lane_info.yaw_diff)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)

    # 速度误差
    v_des = float(env.desired_speed()) if hasattr(env, "desired_speed") else 0.0
    v_cur = float(getattr(rs.ego_info, "speed", 0.0))
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=device)

    return yaw_err, v_err


def run_pid_baseline(num_episodes: int = 10,
                     max_steps_per_ep: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CarlaEnv(no_rendering_mode=True)
    policy = PIDRulePolicy()

    all_rewards = []

    for ep in range(num_episodes):
        raw = env.reset()
        policy.reset()

        ep_reward = 0.0

        for t in range(max_steps_per_ep):
            t0 = time.perf_counter()
            yaw_err, v_err = estimate_errors(env, device)

            action = policy.act(yaw_err, v_err, dt=1.0 / config.default_fps)

            t1 = time.perf_counter()
            print((t1 - t0) * 1000.0)

            raw_next, reward, done, info = env.step(action)
            ep_reward += float(reward)

            if done:
                break
            raw = raw_next

        all_rewards.append(ep_reward)

    print("\n=== PID 规则基线结束 ===")
    print(f"平均回报: {sum(all_rewards) / len(all_rewards):.2f} over {num_episodes} episodes")


if __name__ == "__main__":
    run_pid_baseline()
