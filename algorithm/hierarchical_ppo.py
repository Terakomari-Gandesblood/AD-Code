import math
import random
from typing import Tuple, List

import torch

from ablation.hierarchical_ppo.hierarchical_agent import HierarchicalAgent

from config import config
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState


def make_low_obs_from_raw(es: ExtractedState, device: torch.device) -> torch.Tensor:

    parts = [
        es.ego_vector,
        es.gnss_vector,
        # es.imu_vector,
        es.tl_vector,
        es.lane_vector,
        es.nearest_vector
    ]
    obs = torch.cat(parts, dim=-1).to(device)  # [1, low_input_dim]
    obs = obs.view(1, -1)
    return obs


def estimate_errors(env: CarlaEnv, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rs = env.raw_state
    # 方向误差
    yaw_diff_deg = 0.0
    if getattr(rs, "lane_info", None) is not None and hasattr(rs.lane_info, "yaw_diff"):
        yaw_diff_deg = float(rs.lane_info.yaw_diff)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)  # 转为弧度

    # 速度误差
    v_des = float(env.desired_speed()) if hasattr(env, "desired_speed") else 0.0  # 期望速度
    v_cur = float(getattr(rs.ego_info, "speed", 0.0))  # 当前速度
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=yaw_err.device)

    return yaw_err, v_err


@torch.no_grad()
def collect_trajectories(agent: HierarchicalAgent, env: CarlaEnv, max_steps: int) -> List[dict]:
    device = agent.device
    trajectories: List[dict] = []
    raw = env.reset()
    step_count = 0

    while step_count < max_steps:
        es = ExtractedState.from_raw(raw, device=device)
        feats = agent.encoding(es)
        global_feat = agent.get_global_feat(feats)
        low_obs = make_low_obs_from_raw(es, device)

        yaw_err, v_err = estimate_errors(env, device)

        action, act_info = agent.select_action(global_feat, low_obs, yaw_err, v_err)

        critic_state = torch.cat([global_feat, low_obs], dim=-1)
        v_old = agent.evaluate_state(critic_state).detach().squeeze(0)

        raw_next, reward, done, info = env.step(action)

        es_next = ExtractedState.from_raw(raw_next, device=device)
        feats_next = agent.encoding(es_next)
        global_feat_next = agent.get_global_feat(feats_next)
        low_obs_next = make_low_obs_from_raw(es_next, device)
        next_critic_state = torch.cat([global_feat_next, low_obs_next], dim=-1)
        v_next = agent.evaluate_state(next_critic_state).detach().squeeze(0)

        trajectories.append({
            "global_feat": global_feat.detach().squeeze(0),
            "obs": low_obs.detach().squeeze(0),
            "dir_idx": act_info["dir_idx"].detach().view(1),
            "spd_idx": act_info["spd_idx"].detach().view(1),
            "low_squashed": act_info["low_squashed"].detach().squeeze(0),
            "log_p_old_high": act_info["log_p_old_high"].detach().squeeze(0),
            "log_p_old_low": act_info["log_p_old_low"].detach().squeeze(0),
            "critic_state": critic_state.detach().squeeze(0),
            "next_critic_state": next_critic_state.detach().squeeze(0),
            "reward": float(reward),
            "done": float(done),
            "v_old": v_old,
            "v_next_old": v_next
        })

        step_count += 1
        raw = env.reset() if done else raw_next

    return trajectories


def compute_advantages(rewards: List[float],
                       values: torch.Tensor,
                       dones: List[float],
                       gamma: float,
                       lam: float,
                       device: torch.device):
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32, device=device)  # 优势函数
    last_gae = 0.0  # 用于反向累积的临时变量

    # 反向遍历时间步
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])  # 是否终止
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]  # TD误差 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        last_gae = delta + gamma * lam * nonterminal * last_gae  # GAE递归公式 A_t = δ_t + γ * λ * A_{t+1}
        adv[t] = last_gae  # 存储当前时间步的优势估计
    returns = adv + values[:-1]  # 计算实际回报
    return adv, returns


@torch.no_grad()
def compute_advantages_from_trajectories(agent, trajectories, gamma: float, lam: float, use_bootstrap_last=True):
    device = agent.device

    rewards = [t["reward"] for t in trajectories]
    dones = [t["done"] for t in trajectories]

    v = torch.stack([t["v_old"] for t in trajectories]).to(torch.float32).to(device)

    if use_bootstrap_last:
        last_v_next = trajectories[-1]["v_next_old"].to(torch.float32).to(device)
        if dones[-1] >= 0.5:
            last_v_next = torch.zeros((), dtype=torch.float32, device=device)
    else:
        last_v_next = torch.zeros((), dtype=torch.float32, device=device)

    values = torch.cat([v, last_v_next.view(1)], dim=0)

    adv, returns = compute_advantages(rewards, values, dones, gamma, lam, device=device)
    return adv, returns


def train():
    env = CarlaEnv(no_rendering_mode=True)
    agent = HierarchicalAgent()

    if getattr(config, "hierarchical_ppo_resume_filename", None):
        try:
            last_ep = agent.load_checkpoint(config.resume_filename)
            print(f"Resumed from episode {last_ep}")
        except FileNotFoundError:
            print("No resume file found, start fresh.")

    mini_batch = config.mini_batch  # 小批次的大小
    max_step = mini_batch * 32  # 采样的批次大小

    episode_rewards: List[float] = []  # 记录所有回合的奖励

    best_avg_reward = float("-inf")
    ep_window = config.reward_window

    for episode in range(config.episodes):
        print(f"\n=== Episode {episode} 开始 ===")

        # 采集样本
        print("采集轨迹开始...")
        trajectories = collect_trajectories(agent, env, max_step)
        print(f"[Episode {episode}] 采集样本完成: {len(trajectories)} steps")

        episode_reward = sum([x['reward'] for x in trajectories])  # 当前回合总奖励
        episode_rewards.append(episode_reward)
        avg_reward = sum(episode_rewards[-ep_window:]) / min(len(episode_rewards), ep_window)
        print(
            f"[Episode {episode}] 平均每步奖励: {episode_reward / max_step:.4f} | 窗口均回报({ep_window}ep): {avg_reward:.2f}")

        advantages, _returns = compute_advantages_from_trajectories(agent, trajectories, config.gamma, config.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 优势标准化
        adv_clip = config.adv_clip
        advantages = advantages.clamp(-adv_clip, adv_clip)  # 优势截断

        # 将轨迹和优势函数打包为列表并打乱顺序
        data = list(zip(trajectories, advantages))
        random.shuffle(data)
        trajectories, advantages = zip(*data)

        for epoch in range(config.epochs):
            # 小批次更新
            for i in range(0, len(trajectories), mini_batch):
                mini_batch_trajectories = trajectories[i:i + mini_batch]
                mini_batch_advantages = advantages[i:i + mini_batch]

                # 更新所有网络
                actor_stat, critic_stat = agent.update_all_net(mini_batch_trajectories, mini_batch_advantages)

                if i % 100 == 0:
                    print(f"[Episode {episode} | Epoch {epoch}] "
                          f"actor_loss={actor_stat['loss_actor']:.4f} "
                          f"policy_loss={actor_stat['policy_loss']:.4f} "
                          f"critic_loss={critic_stat['loss_critic']:.4f} "
                          f"v_mean={critic_stat['value_mean']:.3f} "
                          f"v_tgt_mean={critic_stat['v_tgt_mean']:.3f}")

        # 更新旧策略
        agent.update_old_policy()

        # 学习率收敛
        agent.scheduler_actor.step()
        agent.scheduler_critic.step()

        # 保存模型
        if (episode + 1) % getattr(config, "save_every", 5) == 0:
            agent.save_checkpoint(episode, {"avg_reward": avg_reward, "ep_reward": episode_reward},
                                  filename=f"hierarchical_ppo_ep{episode}.pt")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_checkpoint(episode, {"avg_reward": avg_reward, "ep_reward": episode_reward},
                                  filename="hierarchical_ppo_best.pt")


if __name__ == '__main__':
    train()
