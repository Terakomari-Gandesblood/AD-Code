import math
import random
from typing import Tuple, List

import torch

from ablation.pure_policy.pure_agent import PureAgent
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
    obs = torch.cat(parts, dim=-1).to(device)  # [low_input_dim]
    return obs.view(1, -1)


def estimate_errors(env: CarlaEnv, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rs = env.raw_state

    yaw_diff_deg = 0.0
    if getattr(rs, "lane_info", None) is not None and hasattr(rs.lane_info, "yaw_diff"):
        yaw_diff_deg = float(rs.lane_info.yaw_diff)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)

    v_des = float(env.desired_speed()) if hasattr(env, "desired_speed") else 0.0
    v_cur = float(getattr(rs.ego_info, "speed", 0.0))
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=device)

    return yaw_err, v_err


@torch.no_grad()
def collect_trajectories(agent: PureAgent,
                         env: CarlaEnv,
                         max_steps: int) -> List[dict]:
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

        action, act_info = agent.select_action(global_feat, yaw_err, v_err)

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
            "squashed": act_info["squashed"].detach().squeeze(0),
            "log_p_old": act_info["log_p_old"].detach().squeeze(0),

            "critic_state": critic_state.detach().squeeze(0),
            "next_critic_state": next_critic_state.detach().squeeze(0),

            "reward": float(reward),
            "done": float(done),
            "v_old": v_old,
            "v_next_old": v_next,
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
    adv = torch.zeros(T, dtype=torch.float32, device=device)
    last_gae = 0.0

    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae

    returns = adv + values[:-1]
    return adv, returns


@torch.no_grad()
def compute_advantages_from_trajectories(agent: PureAgent,
                                         trajectories: List[dict],
                                         gamma: float,
                                         lam: float,
                                         use_bootstrap_last: bool = True):
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
    agent = PureAgent()

    if getattr(config, "pure_ppo_resume_filename", None):
        try:
            last_ep = agent.load_checkpoint(config.resume_filename)
            print(f"Resumed from episode {last_ep}")
        except FileNotFoundError:
            print("No resume file found, start fresh.")

    mini_batch = config.mini_batch
    max_step = mini_batch * 32

    episode_rewards: List[float] = []
    best_avg_reward = float("-inf")
    ep_window = config.reward_window

    for episode in range(config.episodes):
        print(f"\n=== [Pure PPO] Episode {episode} 开始 ===")

        print("采集轨迹开始...")
        trajectories = collect_trajectories(agent, env, max_step)
        print(f"[Episode {episode}] 采集样本完成: {len(trajectories)} steps")

        episode_reward = sum([x['reward'] for x in trajectories])
        episode_rewards.append(episode_reward)
        avg_reward = sum(episode_rewards[-ep_window:]) / min(len(episode_rewards), ep_window)
        print(f"[Episode {episode}] 总回报: {episode_reward:.2f} | "
              f"平均每步奖励: {episode_reward / max_step:.4f} | "
              f"窗口均回报({ep_window}ep): {avg_reward:.2f}")

        advantages, _returns = compute_advantages_from_trajectories(
            agent, trajectories, config.gamma, config.lam
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_clip = config.adv_clip
        advantages = advantages.clamp(-adv_clip, adv_clip)

        data = list(zip(trajectories, advantages))
        random.shuffle(data)
        trajectories, advantages = zip(*data)

        for epoch in range(config.epochs):
            for i in range(0, len(trajectories), mini_batch):
                mini_batch_trajectories = trajectories[i:i + mini_batch]
                mini_batch_advantages = advantages[i:i + mini_batch]

                actor_stat, critic_stat = agent.update_all_net(
                    mini_batch_trajectories,
                    mini_batch_advantages
                )

                if i % 100 == 0:
                    print(f"[Ep {episode} | Epoch {epoch}] "
                          f"actor_loss={actor_stat['loss_actor']:.4f} "
                          f"policy_loss={actor_stat['policy_loss']:.4f} "
                          f"critic_loss={critic_stat['loss_critic']:.4f} "
                          f"v_mean={critic_stat['value_mean']:.3f} "
                          f"v_tgt_mean={critic_stat['v_tgt_mean']:.3f}")

        agent.update_old_policy()
        agent.scheduler_actor.step()
        agent.scheduler_critic.step()

        if (episode + 1) % getattr(config, "save_every", 5) == 0:
            agent.save_checkpoint(
                episode,
                {"avg_reward": avg_reward, "ep_reward": episode_reward},
                filename=f"pure_ppo_ep{episode}.pt"
            )

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_checkpoint(
                episode,
                {"avg_reward": avg_reward, "ep_reward": episode_reward},
                filename="pure_ppo_best.pt"
            )


if __name__ == '__main__':
    train()
