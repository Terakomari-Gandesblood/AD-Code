import torch
import pytest

from config import config
from model.agent import Agent


@pytest.mark.parametrize("device", ["cpu"])
def test_init_and_old_policy_copy(device):
    torch.manual_seed(0)
    agent = Agent()
    # 快照是否同构
    s1 = agent.actor.state_dict()
    s2 = agent.old_actor.state_dict()
    # 完全一致
    for k in s1:
        assert torch.equal(s1[k].cpu(), s2[k].cpu()), f"param mismatch at {k}"

    # 行为标志同步
    assert agent.actor.use_residual == agent.old_actor.use_residual
    assert agent.actor.use_condition == agent.old_actor.use_condition
    assert agent.actor.use_exploration == agent.old_actor.use_exploration


def _rand(batch, dim):
    return torch.randn(batch, dim, dtype=torch.float32)


def test_select_action_and_logprob_flow():
    torch.manual_seed(1)
    agent = Agent()

    B = 4
    # 直接造 feature，绕过编码 & GCN
    global_feat = _rand(B, config.high_input_dim)
    obs = _rand(B, config.low_input_dim)
    yaw_err = torch.randn(B)
    v_err = torch.randn(B)

    # 选动作（会调用 old_actor.evaluate 计算 old logprob）
    action, out = agent.select_action(global_feat, obs, yaw_err, v_err)

    # 形状检查
    assert action.shape == (B, 3)
    for k in ["dir_idx", "spd_idx", "low_squashed", "log_p_old_high", "log_p_old_low"]:
        assert k in out

    assert out["dir_idx"].shape == (B,)
    assert out["spd_idx"].shape == (B,)
    assert out["low_squashed"].shape == (B, 3)
    assert out["log_p_old_high"].shape == (B,)
    assert out["log_p_old_low"].shape == (B,)


def test_one_update_step_smoke():
    torch.manual_seed(2)
    agent = Agent()
    device = agent.device

    B = 4
    # 造一批伪数据，使 update_all_net 能完整跑通
    trajs = []
    for _ in range(B):
        gf = torch.randn(1, config.high_input_dim)
        ob = torch.randn(1, config.low_input_dim)
        yaw = torch.randn(1)
        verr = torch.randn(1)

        # 采样一次动作，拿到索引/low_squashed 和 old logprob
        action, info = agent.select_action(gf, ob, yaw, verr)

        critic_state = torch.randn(1, config.critic_input_dim)
        next_critic_state = torch.randn(1, config.critic_input_dim)

        trajs.append({
            "global_feat": gf.squeeze(0),  # (H)
            "obs": ob.squeeze(0),  # (L)
            "dir_idx": info["dir_idx"],  # (1,)
            "spd_idx": info["spd_idx"],  # (1,)
            "low_squashed": info["low_squashed"],  # (1,3)
            "log_p_old_high": info["log_p_old_high"],  # (1,)
            "log_p_old_low": info["log_p_old_low"],  # (1,)

            "critic_state": critic_state.squeeze(0),  # (C)
            "next_critic_state": next_critic_state.squeeze(0),
            "reward": torch.randn(()).item(),
            "done": torch.rand(()).item() > 0.7,  # 随机 done
            "v_old": torch.randn(1),  # 示意缓存的旧 V(s)
        })

    # 造 GAE/优势
    advantages = torch.randn(B).tolist()

    actor_stat, critic_stat = agent.update_all_net(trajs, advantages)

    # 基本字段存在
    for k in ["loss_actor", "policy_loss", "entropy", "ratio_mean", "ratio_clip_frac"]:
        assert k in actor_stat
    for k in ["loss_critic", "v_pred_mean", "v_tgt_mean"]:
        assert k in critic_stat
