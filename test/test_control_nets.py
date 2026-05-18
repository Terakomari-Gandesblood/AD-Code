import pytest
import torch

from model.actor_critic import CriticNetwork, ActorNetwork, LowLevelNet, HighLevelNet

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

BATCH = 4
HIGH_IN, HIGH_H = 64, 128
LOW_IN, LOW_H = 128, 128
DT = 0.05


@pytest.mark.parametrize("device", DEVICES)
def test_high_level_forward_evaluate(device):
    torch.manual_seed(0)
    model = HighLevelNet(layers=3, input_dim=HIGH_IN, hidden_dim=HIGH_H).to(device)
    global_feat = torch.randn(BATCH, HIGH_IN, device=device)

    dir_idx, spd_idx = model(global_feat)
    assert dir_idx.shape == (BATCH,) and spd_idx.shape == (BATCH,)
    assert dir_idx.dtype == torch.long and spd_idx.dtype == torch.long

    log_p, ent = model.evaluate(global_feat, dir_idx, spd_idx)
    assert log_p.shape == (BATCH,) and ent.shape == (BATCH,)
    assert torch.isfinite(log_p).all() and (ent >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_low_level_conditioned(device):
    torch.manual_seed(0)
    low = LowLevelNet(
        layers=3, input_dim=LOW_IN, hidden_dim=LOW_H,
        use_condition=True, use_residual=False, use_exploration=True
    ).to(device)

    obs = torch.randn(BATCH, LOW_IN, device=device)
    dir_idx = torch.randint(0, 3, (BATCH,), device=device)
    spd_idx = torch.randint(0, 3, (BATCH,), device=device)

    act, dist, info = low(obs, dir_idx, spd_idx, dt=DT)

    # 形状与范围
    assert act.shape == (BATCH, 3)
    assert torch.all(act[:, 0].abs() <= 1.0001)  # steer in [-1,1]
    assert torch.all((act[:, 1:] >= -1e-6) & (act[:, 1:] <= 1.0001))  # throttle/brake in [0,1]

    # evaluate 要对 squashed_action（=tanh(u)）求 log_prob
    assert "squashed_action" in info and info["squashed_action"] is not None
    log_p, ent = low.evaluate(obs, dir_idx, spd_idx, low_squashed=info["squashed_action"])
    assert log_p.shape == (BATCH,) and ent.shape == (BATCH,)
    assert torch.isfinite(log_p).all() and (ent >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_low_level_residual(device):
    torch.manual_seed(0)
    low = LowLevelNet(
        layers=3, input_dim=LOW_IN, hidden_dim=LOW_H,
        use_condition=True, use_residual=True, use_exploration=True,
        alpha=0.5, cone_ratio=0.5, residual_scale=(0.3, 0.2, 0.2)
    ).to(device)

    obs = torch.randn(BATCH, LOW_IN, device=device)
    dir_idx = torch.randint(0, 3, (BATCH,), device=device)
    spd_idx = torch.randint(0, 3, (BATCH,), device=device)
    yaw_err = torch.randn(BATCH, device=device) * 0.2
    v_err = torch.randn(BATCH, device=device) * 0.5

    act, dist, residual_squash = low(obs, dir_idx, spd_idx, yaw_err=yaw_err, v_err=v_err, dt=DT)

    # 形状与范围
    assert act.shape == (BATCH, 3)
    assert torch.all(act[:, 0].abs() <= 1.0001)
    assert torch.all((act[:, 1:] >= -1e-6) & (act[:, 1:] <= 1.0001))

    log_p, ent = low.evaluate(
        obs, dir_idx, spd_idx, low_squashed=residual_squash
    )
    assert log_p.shape == (BATCH,) and ent.shape == (BATCH,)
    assert torch.isfinite(log_p).all() and (ent >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_actor_net_forward_and_evaluate(device):
    torch.manual_seed(0)
    actor = ActorNetwork(
        high_layers=3, high_input_dim=HIGH_IN, high_hidden_dim=HIGH_H,
        low_layers=3, low_input_dim=LOW_IN, low_hidden_dim=LOW_H,
        use_condition=True, use_residual=True, use_exploration=True
    ).to(device)

    global_feat = torch.randn(BATCH, HIGH_IN, device=device)
    obs = torch.randn(BATCH, LOW_IN, device=device)
    yaw_err = torch.randn(BATCH, device=device) * 0.2
    v_err = torch.randn(BATCH, device=device) * 0.5

    act, info = actor(global_feat, obs, yaw_err=yaw_err, v_err=v_err, dt=DT)
    assert act.shape == (BATCH, 3)

    log_p_h, ent_h, log_p_l, ent_l = actor.evaluate(
        global_feat, obs,
        dir_idx=info["dir_idx"],
        spd_idx=info["spd_idx"],
        low_squashed=info["low_squashed"]
    )
    for t in (log_p_h, ent_h, log_p_l, ent_l):
        assert t.shape == (BATCH,)
        assert torch.isfinite(t).all()
    assert (ent_h >= 0).all() and (ent_l >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layers", [1, 3])
def test_critic_shapes(device, layers):
    torch.manual_seed(0)
    x_dim, h_dim = 96, 128
    critic = CriticNetwork(num_layers=layers, input_dim=x_dim, hidden_dim=h_dim).to(device)
    x = torch.randn(BATCH, x_dim, device=device)
    v = critic(x)
    assert v.shape == (BATCH,)
    assert torch.isfinite(v).all()
