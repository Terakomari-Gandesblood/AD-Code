import os
import math
from typing import List

import torch

from baseline.sac.sac_agent import SacAgent
from config import config
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState


def estimate_errors(env: CarlaEnv, device: torch.device):
    rs = env.raw_state

    yaw_diff_deg = 0.0
    if getattr(rs, "lane_info", None) is not None and hasattr(rs.lane_info, "yaw_diff"):
        yaw_diff_deg = float(rs.lane_info.yaw_diff)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)

    v_des = float(env.desired_speed()) if hasattr(env, "desired_speed") else 0.0
    v_cur = float(getattr(rs.ego_info, "speed", 0.0))
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=device)

    return yaw_err, v_err


def train():
    env = CarlaEnv(no_rendering_mode=True)
    agent = SacAgent()
    device = agent.device

    num_episodes = getattr(config, "sac_episodes", 150)
    max_ep_steps = getattr(config, "sac_max_ep_steps", 1000)
    batch_size = getattr(config, "sac_batch_size", 128)
    updates_per_step = getattr(config, "sac_updates_per_step", 1)
    warmup_steps = getattr(config, "sac_warmup_steps", 1000)
    log_window = getattr(config, "sac_reward_window", 10)
    save_every = getattr(config, "sac_save_every", 10)

    total_steps = 0
    episode_rewards: List[float] = []
    best_avg_reward = float("-inf")

    for ep in range(num_episodes):
        print(f"\n=== [SAC+Template] Episode {ep} 开始 ===")
        raw = env.reset()
        ep_reward = 0.0

        for t in range(max_ep_steps):
            # === 1) 状态编码 ===
            es = ExtractedState.from_raw(raw, device=device)
            feats = agent.encoding(es)
            global_feat = agent.get_global_feat(feats)
            state_vec = global_feat.squeeze(0)

            yaw_err, v_err = estimate_errors(env, device)

            action_tensor, _info = agent.select_action(
                global_feat=global_feat,
                yaw_err=yaw_err,
                v_err=v_err,
                eval_mode=False
            )
            action = tuple(map(float, action_tensor.detach().cpu().numpy()))

            next_raw, reward, done, info = env.step(action)
            ep_reward += float(reward)

            es_next = ExtractedState.from_raw(next_raw, device=device)
            feats_next = agent.encoding(es_next)
            global_feat_next = agent.get_global_feat(feats_next)
            next_state_vec = global_feat_next.squeeze(0)

            next_yaw_err, next_v_err = estimate_errors(env, device)

            agent.replay_buffer.push(
                state=state_vec,
                action=action_tensor,
                reward=float(reward),
                next_state=next_state_vec,
                done=float(done),
                yaw_err=yaw_err.view(1),
                v_err=v_err.view(1),
                next_yaw_err=next_yaw_err.view(1),
                next_v_err=next_v_err.view(1),
            )

            total_steps += 1

            if len(agent.replay_buffer) >= max(warmup_steps, batch_size):
                for _ in range(updates_per_step):
                    stat = agent.update_sac(batch_size=batch_size)

            raw = env.reset() if done else next_raw
            if done:
                break

        episode_rewards.append(ep_reward)
        window_len = min(len(episode_rewards), log_window)
        avg_reward = sum(episode_rewards[-window_len:]) / window_len

        print(f"[SAC+Template][Episode {ep}] 回报: {ep_reward:.2f} | "
              f"滑动均回报({window_len}) = {avg_reward:.2f} | "
              f"总步数 = {total_steps}")

        save_dir = config.sac_save_path
        os.makedirs(save_dir, exist_ok=True)

        if (ep + 1) % save_every == 0:
            path = os.path.join(save_dir, f"sac_template_ep{ep}.pt")
            torch.save(agent.state_dict(), path)
            print(f"[SAC+Template][Checkpoint] saved to: {path}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            path = os.path.join(save_dir, "sac_template_best.pt")
            torch.save(agent.state_dict(), path)
            print(f"[SAC+Template][Best] 更新最佳模型，保存到: {path}")


if __name__ == '__main__':
    train()
