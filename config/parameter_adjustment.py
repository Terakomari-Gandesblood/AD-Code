import math
import random
from typing import Tuple, List
import torch

import config
from environment.carla_env import CarlaEnv
from environment.state import ExtractedState
from model.agent import Agent
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def make_low_obs_from_raw(es: ExtractedState, device: torch.device) -> torch.Tensor:
    parts = [
        es.ego_vector,
        es.gnss_vector,
        # es.imu_vector,
        es.tl_vector,
        es.lane_vector,
        es.nearest_vector
    ]
    obs = torch.cat(parts, dim=-1).to(device)
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
def collect_trajectories(agent: Agent, env: CarlaEnv, max_steps: int) -> List[dict]:
    device = agent.device
    trajectories: List[dict] = []
    raw = env.reset()
    step_count = 0

    while step_count < max_steps:
        es = ExtractedState.from_raw(raw, device=device)
        feats = agent.encoding(es)
        global_feat = agent.get_global_feat(feats, raw)
        low_obs = make_low_obs_from_raw(es, device)

        yaw_err, v_err = estimate_errors(env, device)

        action, act_info = agent.select_action(global_feat, low_obs, yaw_err, v_err)

        critic_state = torch.cat([global_feat, low_obs], dim=-1)
        v_old = agent.evaluate_state(critic_state).detach().squeeze(0)

        raw_next, reward, done, info = env.step(action)

        es_next = ExtractedState.from_raw(raw_next, device=device)
        feats_next = agent.encoding(es_next)
        global_feat_next = agent.get_global_feat(feats_next, raw_next)
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


def run_training(num_episodes: int) -> float:
    env = CarlaEnv(no_rendering_mode=True)
    agent = Agent()

    if getattr(config, "resume_filename", None):
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

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} 开始 ===")
        trajectories = collect_trajectories(agent, env, max_step)
        print(f"[Episode {episode}] 采集样本完成: {len(trajectories)} steps")

        episode_reward = sum([x['reward'] for x in trajectories])
        episode_rewards.append(episode_reward)
        avg_reward = sum(episode_rewards[-ep_window:]) / min(len(episode_rewards), ep_window)
        print(
            f"[Episode {episode}] 平均每步奖励: {episode_reward / max_step:.4f} | "
            f"窗口均回报({ep_window}ep): {avg_reward:.2f}"
        )
        advantages, _returns = compute_advantages_from_trajectories(
            agent, trajectories, config.gamma, config.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-config.adv_clip, config.adv_clip)

        data = list(zip(trajectories, advantages))
        random.shuffle(data)
        trajectories, advantages = zip(*data)

        for epoch in range(config.epochs):
            for i in range(0, len(trajectories), mini_batch):
                mini_batch_trajectories = trajectories[i:i + mini_batch]
                mini_batch_advantages = advantages[i:i + mini_batch]

                actor_stat, critic_stat = agent.update_all_net(
                    mini_batch_trajectories, mini_batch_advantages
                )

                if i % 100 == 0:
                    print(
                        f"[Episode {episode} | Epoch {epoch}] "
                        f"actor_loss={actor_stat['loss_actor']:.4f} "
                        f"policy_loss={actor_stat['policy_loss']:.4f} "
                        f"critic_loss={critic_stat['loss_critic']:.4f} "
                        f"v_mean={critic_stat['value_mean']:.3f} "
                        f"v_tgt_mean={critic_stat['v_tgt_mean']:.3f}"
                    )

        # 更新旧策略与学习率
        agent.update_old_policy()
        agent.scheduler_actor.step()
        agent.scheduler_critic.step()

        # 更新 best_avg_reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

    return best_avg_reward


def objective(trial: optuna.Trial) -> float:
    config.actor_lr = trial.suggest_float("actor_lr", 1e-5, 3e-4, log=True)
    config.critic_lr = trial.suggest_float("critic_lr", 3e-5, 3e-3, log=True)

    config.gamma = trial.suggest_float("gamma", 0.90, 0.99)
    config.lam = trial.suggest_float("lam", 0.90, 0.97)
    config.epsilon = trial.suggest_float("epsilon", 0.1, 0.3)

    config.max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    config.adv_clip = trial.suggest_float("adv_clip", 5.0, 20.0)

    config.mini_batch = trial.suggest_categorical("mini_batch", [32, 64, 128])
    config.epochs = trial.suggest_categorical("epochs", [5, 10, 15])

    # 层次策略相关
    config.alpha = trial.suggest_float("alpha", 0.02, 0.5, log=True)
    config.cone_ratio = trial.suggest_float("cone_ratio", 0.1, 0.5)
    residual_scale_scalar = trial.suggest_float("residual_scale_scalar", 0.1, 0.5)
    config.residual_scale = (residual_scale_scalar, residual_scale_scalar, residual_scale_scalar)

    config.cond_dim = trial.suggest_categorical("cond_dim", [8, 16, 32])

    # 为加快 HPO 不用 config.episodes 的全量，
    num_episodes_for_hpo = min(config.episodes, 20)

    best_avg_reward = run_training(num_episodes_for_hpo)

    return best_avg_reward


if __name__ == '__main__':
    # 使用 TPE + 中位数剪枝
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print("  value:", study.best_trial.value)
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
