from __future__ import annotations

import os
import time
from typing import Tuple, Optional

import torch

from baseline.behavioral_cloning.bc_agent import BCResidualAgent
from config import config
from environment.carla_env import CarlaEnv
from environment.state import RawState, ExtractedState
from evaluation.utils import compute_errors_from_raw


def _resolve_bc_ckpt_path(ckpt_filename: str) -> str:
    base = config.behavioral_cloning_save_path
    if os.path.isabs(ckpt_filename) or (os.sep in ckpt_filename) or ("/" in ckpt_filename):
        return ckpt_filename
    return os.path.join(base, ckpt_filename)


def _load_actor_state_dict(ckpt_path: str, device: torch.device) -> dict:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "actor_state_dict" in obj:
        return obj["actor_state_dict"]
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "actor"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


class BCEvalPolicy:

    def __init__(self, ckpt_filename: str, device: Optional[torch.device] = None):
        self.agent = BCResidualAgent(device=device)  # 内部会自己选 cuda/cpu
        self.device = device or self.agent.device

        ckpt_path = _resolve_bc_ckpt_path(ckpt_filename)
        state_dict = _load_actor_state_dict(ckpt_path, self.device)
        self.agent.actor.load_state_dict(state_dict)

        for name in ["ego_encoder", "gnss_encoder", "vehicle_encoder", "tl_encoder",
                     "lane_encoder", "actor"]:
            m = getattr(self.agent, name, None)
            if m is not None:
                m.eval()

    def act(self, env: CarlaEnv) -> Tuple[float, float, float]:
        device = self.device
        raw_state: RawState = env.raw_obs

        t0 = time.perf_counter()

        es = ExtractedState.from_raw(raw_state, device=device)

        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats)

        yaw_err, v_err = compute_errors_from_raw(raw_state, device)

        with torch.no_grad():
            pred_actions, _info = self.agent.actor(
                global_feat=global_feat,
                yaw_err=yaw_err,
                v_err=v_err,
            )

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)
        a = pred_actions.view(-1).detach().cpu().tolist()
        if len(a) != 3:
            raise ValueError(f"BC actor output shape mismatch: got {pred_actions.shape}, flattened len={len(a)}")
        steer, throttle, brake = float(a[0]), float(a[1]), float(a[2])
        return steer, throttle, brake

    def predict(self, raw_state: RawState) -> Tuple[float, float, float]:

        device = self.device

        t0 = time.perf_counter()

        es = ExtractedState.from_raw(raw_state, device=device)

        feats = self.agent.encoding(es)
        global_feat = self.agent.get_global_feat(feats)  # (1, 6*D)

        yaw_err, v_err = compute_errors_from_raw(raw_state, device)  # (1,), (1,)

        with torch.no_grad():
            pred_actions, _pred_residual = self.agent.actor(
                global_feat=global_feat,
                yaw_err=yaw_err,
                v_err=v_err,
            )

        t1 = time.perf_counter()
        # print((t1 - t0) * 1000.0)

        a = pred_actions.view(-1).detach().cpu().tolist()  # [steer, throttle, brake]
        steer, throttle, brake = float(a[0]), float(a[1]), float(a[2])
        return steer, throttle, brake


def run_bc_eval(
        episodes: int = 3,
        no_rendering_mode: bool = True,
        npc_vehicle_num: int = 10,
        ckpt_filename: str = "bc_residual_best.pt",
):
    env = CarlaEnv(no_rendering_mode=no_rendering_mode, npc_vehicle_num=npc_vehicle_num)

    policy = BCEvalPolicy(ckpt_filename=ckpt_filename)

    for ep in range(episodes):
        env.reset()
        ep_ret = 0.0
        steps = 0

        while True:
            action = policy.act(env)

            raw_state, reward, done, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

            if done:
                print(
                    f"[BC Residual] Episode {ep} done | "
                    f"steps={steps} | return={ep_ret:.2f} | "
                    f"reason={info.get('done_reason', 'unknown')}"
                )
                break

    print("[BC Residual] Eval 完成")


if __name__ == "__main__":
    run_bc_eval(
        episodes=5,
        no_rendering_mode=True,
        npc_vehicle_num=10,
        ckpt_filename="bc_residual_best.pt",
    )
