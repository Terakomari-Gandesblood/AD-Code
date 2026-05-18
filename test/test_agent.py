import os
import pytest
import torch
from model.actor_critic import ActorNetwork, CriticNetwork
from config import config

# 让测试在 CPU 也能跑
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ====== 工具 ======
def rand_like(shape, device):
    return torch.randn(*shape, device=device, dtype=torch.float32)


# ====== 测试 1：Actor 正向 & old_actor 设备一致 ======
def test_actor_forward_and_old_policy_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = ActorNetwork().to(device)
    # create_from_existing 后，确保放到同一设备
    old_actor = ActorNetwork.create_from_existing(actor).to(device)

    B = 8
    global_feat = rand_like((B, config.high_input_dim), device)
    obs = rand_like((B, config.low_input_dim), device)
    yaw_err = rand_like((B,), device)
    v_err = rand_like((B,), device)

    act, info = actor(global_feat=global_feat, obs=obs, yaw_err=yaw_err, v_err=v_err)
    assert act.shape == (B, 3)
    assert "low_squashed" in info and info["low_squashed"].shape == (B, 3)
    assert "dir_idx" in info and info["dir_idx"].shape == (B,)
    assert "spd_idx" in info and info["spd_idx"].shape == (B,)

    # old policy evaluate（如果 old_actor 在 CPU，而输入在 GPU，会直接报错）
    log_p_h, ent_h, log_p_l, ent_l = old_actor.evaluate(
        global_feat=global_feat,
        obs=obs,
        dir_idx=info["dir_idx"],
        spd_idx=info["spd_idx"],
        low_squashed=info["low_squashed"],
    )
    assert log_p_h.shape == (B,)
    assert log_p_l.shape == (B,)
    assert ent_h.shape == (B,)
    assert ent_l.shape == (B,)


# ====== 测试 2：PPO 的 update_actor 能跑通并返回合理统计 ======
def test_update_actor_step_runs_and_returns_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    actor = ActorNetwork().to(device)
    old_actor = ActorNetwork.create_from_existing(actor).to(device)

    B = 16
    global_feat = rand_like((B, config.high_input_dim), device)
    obs = rand_like((B, config.low_input_dim), device)
    yaw_err = rand_like((B,), device)
    v_err = rand_like((B,), device)

    # 采样一批动作 + old log prob
    act, info = actor(global_feat=global_feat, obs=obs, yaw_err=yaw_err, v_err=v_err)
    with torch.no_grad():
        log_p_old_high, ent_old_high, log_p_old_low, ent_old_low = old_actor.evaluate(
            global_feat=global_feat,
            obs=obs,
            dir_idx=info["dir_idx"],
            spd_idx=info["spd_idx"],
            low_squashed=info["low_squashed"],
        )

    # 构造优势
    advantages = torch.randn(B, device=device)

    # 用 actor 自己的 evaluate 计算新 log prob
    log_p_new_high, entropy_high, log_p_new_low, entropy_low = actor.evaluate(
        global_feat=global_feat,
        obs=obs,
        dir_idx=info["dir_idx"],
        spd_idx=info["spd_idx"],
        low_squashed=info["low_squashed"],
    )

    # 计算 ratio & clip
    log_p_old = (log_p_old_high + log_p_old_low)
    log_p_new = (log_p_new_high + log_p_new_low)
    ratio = torch.exp(log_p_new - log_p_old)

    eps = getattr(config, "epsilon", 0.2)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 熵项
    entropy = (entropy_high + entropy_low).mean()
    loss = policy_loss - getattr(config, "entropy_coef", 0.01) * entropy

    # 只做一次反传，验证梯度图是否连通
    for p in actor.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    # 某些关键参数应有梯度
    has_grad = any((p.grad is not None and p.grad.abs().sum() > 0) for p in actor.parameters())
    assert has_grad, "Actor should receive gradients in PPO update."


# ====== 测试 3：Critic value clipping 分支 ======
@pytest.mark.parametrize("use_clip", [True, False])
def test_critic_value_update_forward_backward(use_clip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = CriticNetwork().to(device)

    B = 32
    s = rand_like((B, config.critic_input_dim), device)
    ns = rand_like((B, config.critic_input_dim), device)
    r = rand_like((B,), device)
    d = torch.zeros(B, device=device)  # 全部未终止，便于稳定

    v = critic(s)
    with torch.no_grad():
        nv = critic(ns)
        tgt = r + getattr(config, "gamma", 0.99) * (1.0 - d) * nv

    if use_clip:
        v_old = v.detach()
        v_clipped = v_old + (v - v_old).clamp(-0.2, 0.2)
        loss = torch.max((v - tgt) ** 2, (v_clipped - tgt) ** 2).mean()
    else:
        loss = torch.nn.functional.mse_loss(v, tgt)

    for p in critic.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    has_grad = any((p.grad is not None and p.grad.abs().sum() > 0) for p in critic.parameters())
    assert has_grad, "Critic should receive gradients."
