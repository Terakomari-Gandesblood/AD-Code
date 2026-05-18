"""
职责：
1) 读取 scene_cache.py 生成的缓存
2) 对选定 scene 做离线开环评估
3) 输出 CSV（global_summary + detail + episodes）


说明：
nuscenes的指标和实验前前后后修改了很多，时间关系这个脚本写的不是很好，见谅一下
比如最开始实验做了1个scene然后对每个scene生成了一个excel，后续做了50个生成了1200多个excel
推荐直接用excel的数据源获取功能合并一下
这里面有些指标是最开始计划的，但是实际落地发现不是很好，后续修改了指标
由于那些指标要么是基于clean的指标（与攻击无关），要么是用detail计算的，不需要再跑很多轮次
时间关系这里脚本就没改（免得改的乱七八糟出问题），两个脚本分别是compute_clean_risk_metrics.py和make_offline_summary_tables.py

另外说明：在强化学习的安全这部分，真实数据集并不能完全说明策略的安全性，只能说明策略和真人的行为是否一致（毕竟安全路径可能有很多，不能说和人走的不一致就是危险策略）
毕竟强化学习是探索策略，在真实数据集上并不合适，更推荐用模拟器去模拟，真实数据集更适合训练策略
"""

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Callable, Tuple, Protocol, Optional, List, Any

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from config import config
import config as config_module  # 用于打印 config 模块路径，排除“多实例 config”问题

from data_loader.nuscenes_loader import build_raw_state_from_sample
from environment.state import RawState
from evaluation.offline.scene_cache import (
    load_ranked_scenes,
    default_cache_dir,
)
from perturbation.attack import attack
from perturbation.attack_scenarios import (
    INTENSITY_LEVELS,
    make_default_scenarios,
    AttackScenario,
    INTENSITY_PRESETS,  # 用于 debug 打印当前场景的强度 preset 覆盖项
)

try:
    from entity.types import VehicleInfo, SurroundingVehicles
except Exception:
    VehicleInfo = None
    SurroundingVehicles = None


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _make_run_dir(out_root: str) -> str:
    jst = timezone(timedelta(hours=9))
    run_id = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_scene_lookup(nusc: NuScenes, scene_name: str) -> dict:
    matches = [s for s in nusc.scene if s["name"] == scene_name]
    if not matches:
        raise ValueError(f"scene_name={scene_name} not found in this NuScenes version.")
    return matches[0]


def _get_scene_location(nusc: NuScenes, scene: dict) -> str:
    log = nusc.get("log", scene["log_token"])
    return str(log.get("location", ""))


def _try_get_vehicle_monitor_msgs(nusc_can: NuScenesCanBus, scene_name: str) -> Optional[List[dict]]:
    try:
        msgs = nusc_can.get_messages(scene_name, "vehicle_monitor")
        msgs = sorted(msgs, key=lambda m: int(m.get("utime", 0)))
        return msgs if msgs else None
    except Exception:
        return None


def _iter_scene_tokens(nusc: NuScenes, scene: dict, max_pairs: Optional[int]) -> List[str]:
    tokens: List[str] = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        s = nusc.get("sample", tok)
        tok = s["next"]
        if max_pairs is not None and len(tokens) >= max_pairs + 1:
            break
    return tokens


def _postprocess_action(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    steer, thr, brk = a
    steer = float(np.clip(float(steer), -1.0, 1.0))
    thr = float(np.clip(float(thr), 0.0, 1.0))
    brk = float(np.clip(float(brk), 0.0, 1.0))
    if thr > 0.2 and brk > 0.2:
        if thr >= brk:
            brk = 0.0
        else:
            thr = 0.0
    return steer, thr, brk


def _mean_or_zero(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _debug_state_signature(rs: RawState) -> Dict[str, Any]:
    li = getattr(rs, "lane_info", None)
    tl = getattr(rs, "traffic_light", None)
    sv = getattr(rs, "surrounding_vehicles", None)
    vehs = (getattr(sv, "vehicles", None) or []) if sv is not None else []
    v0 = vehs[0] if len(vehs) > 0 else None

    ego = getattr(rs, "ego_info", None)
    gnss = getattr(rs, "gnss", None)

    return {
        # ego
        "ego_speed": float(_safe_float(getattr(ego, "speed", 0.0))),
        "ego_yaw": float(_safe_float(getattr(ego, "rotation_yaw", 0.0))),
        "ego_vx": float(_safe_float(getattr(ego, "velocity_x", 0.0))),
        "ego_vy": float(_safe_float(getattr(ego, "velocity_y", 0.0))),

        # lane
        "lane_yaw_diff": (float(_safe_float(getattr(li, "yaw_diff", 0.0))) if li is not None else None),
        "lane_has_next": (bool(getattr(li, "has_next", False)) if li is not None else None),

        # tl
        "tl_state": (str(getattr(tl, "state", None)) if tl is not None else None),
        "tl_dist": (float(_safe_float(getattr(tl, "distance", 0.0))) if tl is not None else None),

        # vehicles
        "veh_n": int(len(vehs)),
        "veh0_relx": (float(_safe_float(getattr(v0, "relative_x", 0.0))) if v0 is not None else None),
        "veh0_rely": (float(_safe_float(getattr(v0, "relative_y", 0.0))) if v0 is not None else None),
        "veh0_dist": (float(_safe_float(getattr(v0, "distance", 0.0))) if v0 is not None else None),

        # gnss
        "gnss_lat": (float(_safe_float(getattr(gnss, "latitude", 0.0))) if gnss is not None else None),
        "gnss_lon": (float(_safe_float(getattr(gnss, "longitude", 0.0))) if gnss is not None else None),

        # potential caches
        "has_extracted_state": bool(hasattr(rs, "extracted_state")),
        "has_graph": bool(hasattr(rs, "graph") or hasattr(rs, "hetero_data") or hasattr(rs, "hetero")),
    }


# 判断两个签名在指定 key 集上是否发生变化
def _sig_changed(
        sig_a: Dict[str, Any],
        sig_b: Dict[str, Any],
        *,
        keys: List[str],
        float_tol: float = 1e-6,
) -> bool:
    for k in keys:
        va = sig_a.get(k, None)
        vb = sig_b.get(k, None)

        if va is None and vb is None:
            continue

        # 数值：容差比较
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            fa, fb = float(va), float(vb)
            if (not np.isfinite(fa)) or (not np.isfinite(fb)):
                if fa != fb:
                    return True
            else:
                if abs(fa - fb) > float_tol:
                    return True
        else:
            # 非数值：直接比较
            if va != vb:
                return True
    return False


# 判断 apply(level) 是否生效（debug用）
def _debug_print_config(scenario_key: str, level: Optional[int]) -> None:
    flags = {
        "enable_perturbation": getattr(config, "enable_perturbation", None),
        "attack_intensity_level": getattr(config, "attack_intensity_level", None),
        "attack_ego": getattr(config, "attack_ego", None),
        "attack_gnss": getattr(config, "attack_gnss", None),
        "attack_tl": getattr(config, "attack_tl", None),
        "attack_lane": getattr(config, "attack_lane", None),
        "attack_vehicles": getattr(config, "attack_vehicles", None),
        "attack_collision": getattr(config, "attack_collision", None),
        "attack_lane_invasion": getattr(config, "attack_lane_invasion", None),
    }

    preset_keys = []
    if level is not None:
        preset = INTENSITY_PRESETS.get(scenario_key, {}).get(int(level), {})
        preset_keys = list(preset.keys())

    print(
        f"[pert-debug][apply] scenario={scenario_key} level={level} "
        f"flags={flags} preset_keys={preset_keys}"
    )


def _compute_min_dist_ttc_headway(rs: RawState) -> Tuple[float, float, float]:
    min_dist = float("inf")
    min_ttc = float("inf")
    min_headway = float("inf")

    evx = _safe_float(rs.ego_info.velocity_x)
    evy = _safe_float(rs.ego_info.velocity_y)
    ego_speed = max(1e-3, float(math.hypot(evx, evy)))

    yaw = _safe_float(rs.ego_info.rotation_yaw)
    hx, hy = math.cos(yaw), math.sin(yaw)

    sv = getattr(rs, "surrounding_vehicles", None)
    if sv is None or not getattr(sv, "vehicles", None):
        return float("inf"), float("inf"), float("inf")

    for veh in sv.vehicles:
        rx = _safe_float(getattr(veh, "relative_x", 0.0))
        ry = _safe_float(getattr(veh, "relative_y", 0.0))
        dist = float(math.hypot(rx, ry))
        if dist < min_dist:
            min_dist = dist

        ovx = _safe_float(getattr(veh, "velocity_x", 0.0))
        ovy = _safe_float(getattr(veh, "velocity_y", 0.0))

        ux, uy = (rx / max(dist, 1e-6), ry / max(dist, 1e-6))
        closing = (ovx - evx) * ux + (ovy - evy) * uy
        if closing < -1e-3:
            ttc = dist / (-closing)
            if ttc < min_ttc:
                min_ttc = ttc

        forward_proj = rx * hx + ry * hy
        if forward_proj > 0.0:
            headway = forward_proj / ego_speed
            if headway < min_headway:
                min_headway = headway

    return min_dist, min_ttc, min_headway


def _adaptive_ttc_threshold(
        min_ttc_list: List[float],
        base_thr: float = 1.5,
        q: float = 0.2,
        min_count: int = 30,
        ignore_upper: float = 1e6,
) -> Tuple[float, dict]:
    vals = np.asarray(min_ttc_list, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals < ignore_upper]
    info = {"base_thr": float(base_thr), "q": float(q), "valid_n": int(vals.size)}

    if vals.size < min_count:
        info["mode"] = "fixed_base (insufficient_valid_samples)"
        info["thr"] = float(base_thr)
        return float(base_thr), info

    qv = float(np.quantile(vals, q))
    thr = float(max(base_thr, qv))
    info["mode"] = "max(base, quantile)"
    info["quantile_value"] = float(qv)
    info["thr"] = float(thr)
    return thr, info


def _precompute_scene_min_ttc_series(
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        nusc_map: NuScenesMap,
        scene: dict,
        vehicle_monitor_msgs: List[dict],
        max_pairs: Optional[int],
) -> List[float]:
    tokens = _iter_scene_tokens(nusc, scene, max_pairs=max_pairs)
    if len(tokens) < 2:
        return []

    n_pairs = len(tokens) - 1
    s0 = nusc.get("sample", tokens[0])
    rs = build_raw_state_from_sample(
        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
        sample=s0, vehicle_monitor_msgs=vehicle_monitor_msgs
    )

    out: List[float] = []
    for i in range(n_pairs):
        _, min_ttc, _ = _compute_min_dist_ttc_headway(rs)
        out.append(float(min_ttc))

        s1 = nusc.get("sample", tokens[i + 1])
        rs = build_raw_state_from_sample(
            nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
            sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
        )
    return out


class OfflineEvalPolicy(Protocol):
    def predict(self, raw_state: RawState) -> Tuple[float, float, float]: ...


AlgoBuilder = Callable[[], OfflineEvalPolicy]


def build_algo_builders() -> Dict[str, AlgoBuilder]:
    from evaluation.algorithm.behavioral_cloning import BCEvalPolicy
    from evaluation.algorithm.gnn_flat_ppo import GnnFlatPPOEvalPolicy
    from evaluation.algorithm.gnn_hierarchical_ppo import GnnHierarchicalPPOEvalPolicy
    from evaluation.algorithm.hierarchical_ppo import HierarchicalPPOEvalPolicy
    from evaluation.algorithm.pure_ppo import PurePPOEvalPolicy
    from evaluation.algorithm.rule_pid import RulePIDEvalPolicy
    from evaluation.algorithm.sac import SACEvalPolicy

    return {
        "rule_pid": lambda: RulePIDEvalPolicy(),
        "bc": lambda: BCEvalPolicy("bc_residual_best.pt"),
        "gnn_hier_ppo": lambda: GnnHierarchicalPPOEvalPolicy("gnn_hierarchical_ppo_best.pt"),
        "gnn_flat_ppo": lambda: GnnFlatPPOEvalPolicy("gnn_flat_ppo_best.pt"),
        "hier_ppo": lambda: HierarchicalPPOEvalPolicy("hierarchical_ppo_best.pt"),
        "pure_ppo": lambda: PurePPOEvalPolicy("pure_ppo_best.pt"),
        "sac": lambda: SACEvalPolicy("sac_template_best.pt"),
    }


@dataclass
class OfflineScenarioSummary:
    pairs: int
    perturbed_pairs: int
    mean_sad_steer: float
    mean_mae_throttle: float
    mean_mae_brake: float
    mean_action_l1: float
    mean_action_l2: float
    mean_min_dist_m: float
    mean_min_ttc_s: float
    mean_min_headway_s: float
    risk_ttc_pairs: int
    mean_action_l2_on_ttc_risk: float

    def to_row(self, scene_name: str, scenario_key: str, desc: str) -> list:
        return [
            scene_name, scenario_key, desc,
            self.pairs,
            self.perturbed_pairs,
            f"{self.mean_sad_steer:.6f}",
            f"{self.mean_mae_throttle:.6f}",
            f"{self.mean_mae_brake:.6f}",
            f"{self.mean_action_l1:.6f}",
            f"{self.mean_action_l2:.6f}",
            f"{self.mean_min_dist_m:.6f}",
            f"{self.mean_min_ttc_s:.6f}",
            f"{self.mean_min_headway_s:.6f}",
            self.risk_ttc_pairs,
            f"{self.mean_action_l2_on_ttc_risk:.6f}",
        ]


def eval_scene_one_algo_one_scenario_offline(
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        nusc_map: NuScenesMap,
        scene: dict,
        scene_name: str,
        vehicle_monitor_msgs: List[dict],
        policy: OfflineEvalPolicy,

        scenario: AttackScenario,
        level: Optional[int],

        ttc_risk_threshold_s: float,
        max_pairs: Optional[int],
        episode_writer: Optional[csv.writer],
        method_name: str,
        scenario_key: str,
        scenario_desc: str,
        seed: Optional[int],
        imitation_speed_gate_mps: Optional[float],

        # debug controls
        debug_perturbation: bool = False,
        debug_pairs_print: int = 3,
) -> OfflineScenarioSummary:
    # 应用场景配置
    scenario.apply(level=level)

    if seed is not None:
        random.seed(int(seed))

    if debug_perturbation:
        _debug_print_config(scenario_key=scenario.name, level=level)

    tokens = _iter_scene_tokens(nusc, scene, max_pairs=max_pairs)
    if len(tokens) < 2:
        raise ValueError(f"{scene_name} has <2 samples.")

    n_pairs = len(tokens) - 1
    s0 = nusc.get("sample", tokens[0])
    rs_t_clean = build_raw_state_from_sample(
        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
        sample=s0, vehicle_monitor_msgs=vehicle_monitor_msgs
    )

    sad_list, thr_list, brk_list, l1_list, l2_list = [], [], [], [], []
    min_dist_list, min_ttc_list, min_head_list = [], [], []
    l2_on_risk = []
    risk_cnt = 0
    valid_pairs_for_imitation = 0

    # 统计多少 pair 观测签名发生变化（debug）
    perturbed_pairs = 0

    for i in range(n_pairs):
        # expert action
        expert_action = (
            float(_safe_float(rs_t_clean.ego_info.steer)),
            float(_safe_float(rs_t_clean.ego_info.throttle)),
            float(_safe_float(rs_t_clean.ego_info.brake)),
        )

        # safety proxy
        min_dist, min_ttc, min_head = _compute_min_dist_ttc_headway(rs_t_clean)
        min_dist_list.append(min_dist if np.isfinite(min_dist) else 1e9)
        min_ttc_list.append(min_ttc if np.isfinite(min_ttc) else 1e9)
        min_head_list.append(min_head if np.isfinite(min_head) else 1e9)

        # 生成扰动观测
        rs_obs = attack(rs_t_clean)

        sig_clean = None
        sig_obs = None

        # 统计成功扰动的 pair 数
        if scenario_key != "clean":
            sig_clean = _debug_state_signature(rs_t_clean)
            sig_obs = _debug_state_signature(rs_obs)
            keys_to_check = [
                "ego_speed", "ego_yaw",
                "lane_yaw_diff", "lane_has_next",
                "veh_n", "veh0_relx", "veh0_rely", "veh0_dist",
                "tl_state", "tl_dist",
                "gnss_lat", "gnss_lon",
            ]
            if _sig_changed(sig_clean, sig_obs, keys=keys_to_check, float_tol=1e-6):
                perturbed_pairs += 1

        # 对比关键字段签名
        if debug_perturbation and i < max(0, int(debug_pairs_print)):
            if sig_clean is None:
                sig_clean = _debug_state_signature(rs_t_clean)
            if sig_obs is None:
                sig_obs = _debug_state_signature(rs_obs)
            print(f"[pert-debug][sig] scene={scene_name} algo={method_name} scenario={scenario_key} "
                  f"level={level} pair={i}")
            print("  clean:", sig_clean)
            print("  obs  :", sig_obs)

        pred_action = _postprocess_action(policy.predict(rs_obs))

        # 低速帧可选择不计入 imitation error
        if imitation_speed_gate_mps is not None:
            if float(_safe_float(rs_t_clean.ego_info.speed)) < float(imitation_speed_gate_mps):
                if episode_writer is not None:
                    is_ttc_risk = int(np.isfinite(min_ttc) and (min_ttc < ttc_risk_threshold_s))
                    episode_writer.writerow([
                        scene_name, method_name, scenario_key, scenario_desc,
                        int(seed) if seed is not None else "",
                        i,
                        tokens[i], tokens[i + 1],
                        expert_action[0], expert_action[1], expert_action[2],
                        pred_action[0], pred_action[1], pred_action[2],
                        "",  # gated
                        float(min_dist if np.isfinite(min_dist) else 1e9),
                        float(min_ttc if np.isfinite(min_ttc) else 1e9),
                        float(min_head if np.isfinite(min_head) else 1e9),
                        is_ttc_risk,
                    ])

                # step forward
                s1 = nusc.get("sample", tokens[i + 1])
                rs_t_clean = build_raw_state_from_sample(
                    nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
                    sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
                )
                continue

        st_p, th_p, br_p = pred_action
        st_e, th_e, br_e = expert_action

        sad = abs(st_p - st_e)
        mae_th = abs(th_p - th_e)
        mae_br = abs(br_p - br_e)
        l1 = sad + mae_th + mae_br
        l2 = float(math.sqrt((st_p - st_e) ** 2 + (th_p - th_e) ** 2 + (br_p - br_e) ** 2))

        valid_pairs_for_imitation += 1

        if episode_writer is not None:
            is_ttc_risk = int(np.isfinite(min_ttc) and (min_ttc < ttc_risk_threshold_s))
            episode_writer.writerow([
                scene_name, method_name, scenario_key, scenario_desc,
                int(seed) if seed is not None else "",
                i,
                tokens[i], tokens[i + 1],
                expert_action[0], expert_action[1], expert_action[2],
                pred_action[0], pred_action[1], pred_action[2],
                float(l2),
                float(min_dist if np.isfinite(min_dist) else 1e9),
                float(min_ttc if np.isfinite(min_ttc) else 1e9),
                float(min_head if np.isfinite(min_head) else 1e9),
                is_ttc_risk,
            ])

        sad_list.append(sad)
        thr_list.append(mae_th)
        brk_list.append(mae_br)
        l1_list.append(l1)
        l2_list.append(l2)

        if np.isfinite(min_ttc) and min_ttc < ttc_risk_threshold_s:
            risk_cnt += 1
            l2_on_risk.append(l2)

        # step forward
        s1 = nusc.get("sample", tokens[i + 1])
        rs_t_clean = build_raw_state_from_sample(
            nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
            sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
        )

    if debug_perturbation and scenario_key != "clean":
        print(f"[pert-debug][summary] scene={scene_name} algo={method_name} scenario={scenario_key} "
              f"level={level} perturbed_pairs={perturbed_pairs}/{n_pairs}")
        if getattr(config, "enable_perturbation", False) and perturbed_pairs == 0:
            print("[pert-debug][WARN] enable_perturbation=True but no signature change observed. "
                  "Possible causes: (1) 概率型扰动未触发 (prob 太小)；(2) 改了字段但模型不使用；"
                  "(3) RawState 内有缓存特征/图，predict() 未重建。")

    return OfflineScenarioSummary(
        pairs=int(valid_pairs_for_imitation),
        perturbed_pairs=int(perturbed_pairs),
        mean_sad_steer=_mean_or_zero(sad_list),
        mean_mae_throttle=_mean_or_zero(thr_list),
        mean_mae_brake=_mean_or_zero(brk_list),
        mean_action_l1=_mean_or_zero(l1_list),
        mean_action_l2=_mean_or_zero(l2_list),
        mean_min_dist_m=_mean_or_zero(min_dist_list),
        mean_min_ttc_s=_mean_or_zero(min_ttc_list),
        mean_min_headway_s=_mean_or_zero(min_head_list),
        risk_ttc_pairs=int(risk_cnt),
        mean_action_l2_on_ttc_risk=_mean_or_zero(l2_on_risk),
    )


def init_nuscenes() -> Tuple[NuScenes, NuScenesCanBus]:
    version = getattr(config, "NUSCENES_VERSION", "v1.0-mini")
    nusc = NuScenes(version=version, dataroot=config.nuscenes_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=config.can_root)
    return nusc, nusc_can


# 把多个 scene 的 OfflineScenarioSummary 聚合成一行
@dataclass
class _DetailAgg:
    n_scenes: int = 0
    pairs_sum: int = 0  # valid imitation pairs 总数
    perturbed_pairs_sum: int = 0
    sad_sum: float = 0.0
    thr_sum: float = 0.0
    brk_sum: float = 0.0
    l1_sum: float = 0.0
    l2_sum: float = 0.0
    mindist_sum: float = 0.0
    minttc_sum: float = 0.0
    minhead_sum: float = 0.0

    # risk subset
    risk_pairs_sum: int = 0
    risk_l2_sum: float = 0.0  # risk_pairs-weighted


def _detail_accumulate(agg: _DetailAgg, s: OfflineScenarioSummary):
    agg.n_scenes += 1
    agg.perturbed_pairs_sum += int(getattr(s, "perturbed_pairs", 0))
    w = int(s.pairs)
    if w <= 0:
        return
    agg.pairs_sum += w

    agg.sad_sum += float(s.mean_sad_steer) * w
    agg.thr_sum += float(s.mean_mae_throttle) * w
    agg.brk_sum += float(s.mean_mae_brake) * w
    agg.l1_sum += float(s.mean_action_l1) * w
    agg.l2_sum += float(s.mean_action_l2) * w
    agg.mindist_sum += float(s.mean_min_dist_m) * w
    agg.minttc_sum += float(s.mean_min_ttc_s) * w
    agg.minhead_sum += float(s.mean_min_headway_s) * w

    rp = int(s.risk_ttc_pairs)
    agg.risk_pairs_sum += rp
    if rp > 0:
        agg.risk_l2_sum += float(s.mean_action_l2_on_ttc_risk) * rp


# 把聚合后的 agg 转成均值
def _detail_finalize_row(agg: _DetailAgg) -> Dict[str, float]:
    eps = 1e-9
    w = max(eps, float(agg.pairs_sum))
    out = {
        "n_scenes": float(agg.n_scenes),
        "pairs_sum": float(agg.pairs_sum),
        "perturbed_pairs": float(agg.perturbed_pairs_sum),
        "mean_sad_steer": agg.sad_sum / w,
        "mean_mae_throttle": agg.thr_sum / w,
        "mean_mae_brake": agg.brk_sum / w,
        "mean_action_l1": agg.l1_sum / w,
        "mean_action_l2": agg.l2_sum / w,

        "mean_min_dist_m(clean)": agg.mindist_sum / w,
        "mean_min_ttc_s(clean)": agg.minttc_sum / w,
        "mean_min_headway_s(clean)": agg.minhead_sum / w,

        "risk_ttc_pairs": float(agg.risk_pairs_sum),
        "mean_action_l2_on_ttc_risk": (agg.risk_l2_sum / max(eps, float(agg.risk_pairs_sum)))
        if agg.risk_pairs_sum > 0 else 0.0,
    }
    return out


def run_all_offline_and_save_csv(
        *,
        out_root: str = "offline_eval_out",
        seed: int = 0,

        max_pairs_eval: Optional[int] = None,
        max_scenes_eval: int = 10,

        cache_json_path: Optional[str] = None,

        base_ttc_thr: float = 1.5,
        risk_quantile: float = 0.2,

        imitation_speed_gate_mps: Optional[float] = 1.0,

        intensity_names: Optional[List[str]] = None,

        algo_whitelist: Optional[List[str]] = None,
        scenario_whitelist: Optional[List[str]] = None,

        profile: bool = True,

        # debug
        debug_perturbation: bool = True,
        debug_print_pairs: int = 3,
        debug_only_first_scene_algo: bool = True,
):
    run_root = _make_run_dir(out_root)

    if cache_json_path is None:
        cache_json_path = os.path.join(default_cache_dir(), "selected_scenes.json")

    if not os.path.exists(cache_json_path):
        raise FileNotFoundError(
            f"scene cache not found: {cache_json_path}\n"
            f"Please run: python evaluation/offline/scene_cache.py first."
        )

    t0 = time.perf_counter()
    nusc, nusc_can = init_nuscenes()
    t1 = time.perf_counter()

    # load scenes
    scene_names = load_ranked_scenes(cache_json_path, take=max_scenes_eval)
    if len(scene_names) == 0:
        raise RuntimeError("No scenes loaded from cache. Check cache json content.")
    t2 = time.perf_counter()

    _write_json(os.path.join(run_root, "meta.json"), {
        "nuscenes_version": getattr(config, "NUSCENES_VERSION", "v1.0-mini"),
        "seed": seed,
        "max_pairs_eval": max_pairs_eval,
        "max_scenes_eval": max_scenes_eval,
        "cache_json_path": cache_json_path,
        "risk": {"base_ttc_thr": base_ttc_thr, "risk_quantile": risk_quantile},
        "imitation_speed_gate_mps": imitation_speed_gate_mps,
        "intensity_names": intensity_names,
        "algo_whitelist": algo_whitelist,
        "scenario_whitelist": scenario_whitelist,
        "note": "offline uses AttackScenario.apply(level)+attack(raw_state) (same as simulator)",
        "config_module_file": getattr(config_module, "__file__", ""),
    })

    # snapshot cache
    with open(cache_json_path, "r", encoding="utf-8") as f:
        cache_obj = json.load(f)
    _write_json(os.path.join(run_root, "scene_cache_snapshot.json"), cache_obj)

    if profile:
        print(f"[profile] init nusc+can: {(t1 - t0):.3f}s | load scenes: {(t2 - t1):.3f}s")
        print(f"[scenes] eval top-{len(scene_names)}: {scene_names[:10]}{' ...' if len(scene_names) > 10 else ''}")
        print(f"[debug] config module file: {getattr(config_module, '__file__', '')}")

    algo_builders = build_algo_builders()

    scenarios = make_default_scenarios()

    if algo_whitelist is not None:
        algo_set = set(algo_whitelist)
        algo_builders = {k: v for k, v in algo_builders.items() if k in algo_set}

    if scenario_whitelist is not None:
        scen_set = set(scenario_whitelist)
        scenarios = {k: v for k, v in scenarios.items() if (k in scen_set) or (k == "clean")}

    scen_keys = [k for k in scenarios.keys() if k != "clean"]  # attacks only

    levels = list(INTENSITY_LEVELS)
    if intensity_names is not None:
        names = set(intensity_names)
        levels = [x for x in levels if x.name in names]

    map_cache: Dict[str, NuScenesMap] = {}

    merged_global_path = os.path.join(run_root, "offline_global_merged_L1_L3.csv")
    merged_detail_path = os.path.join(run_root, "offline_detail_merged_L1_L3.csv")

    detail_agg: Dict[Tuple[str, str, str], _DetailAgg] = {}

    merged_global_f = open(merged_global_path, "w", newline="", encoding="utf-8")
    merged_global_writer = csv.writer(merged_global_f)
    merged_global_header_written = False

    try:
        for lvl in levels:
            run_dir = os.path.join(run_root, f"intensity_{lvl.name}")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "detail"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "episodes"), exist_ok=True)

            global_csv = os.path.join(run_dir, "offline_global_summary.csv")

            global_header = (
                    ["scene", "method",
                     "clean_action_l2", "clean_sad_steer", "clean_l2_ttc_risk"]
                    + [f"{k}_action_l2_ratio" for k in scen_keys]
                    + [f"{k}_sad_ratio" for k in scen_keys]
                    + [f"{k}_l2_ttc_risk_ratio" for k in scen_keys]
            )

            with open(global_csv, "w", newline="", encoding="utf-8") as f_global:
                writer_g = csv.writer(f_global)
                writer_g.writerow(global_header)

                if (not merged_global_header_written) and (lvl.name in {"L1", "L2", "L3"}):
                    merged_global_writer.writerow(["attack"] + global_header)
                    merged_global_header_written = True

                for scene_idx, scene_name in enumerate(scene_names):
                    scene = _safe_scene_lookup(nusc, scene_name)
                    location = _get_scene_location(nusc, scene)

                    if location not in map_cache:
                        map_cache[location] = NuScenesMap(dataroot=config.map_root, map_name=location)
                    nusc_map = map_cache[location]

                    vm_msgs = _try_get_vehicle_monitor_msgs(nusc_can, scene_name)
                    if vm_msgs is None:
                        print(f"[skip] no CAN for scene={scene_name}")
                        continue

                    ttc_series = _precompute_scene_min_ttc_series(
                        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
                        scene=scene, vehicle_monitor_msgs=vm_msgs,
                        max_pairs=max_pairs_eval,
                    )
                    ttc_thr_scene, dbg = _adaptive_ttc_threshold(ttc_series, base_thr=base_ttc_thr, q=risk_quantile)

                    if profile:
                        print(f"[risk-thr] {scene_name} thr={ttc_thr_scene:.3f}s dbg={dbg}")

                    for algo_idx, (algo_name, builder) in enumerate(algo_builders.items()):
                        if profile:
                            print(f"\n[offline] intensity={lvl.name} scene={scene_name} algo={algo_name}")

                        policy = builder()

                        episodes_csv = os.path.join(run_dir, "episodes", f"{scene_name}__{algo_name}__episodes.csv")

                        write_scene_detail = (lvl.name == "L0")
                        if write_scene_detail:
                            detail_csv = os.path.join(run_dir, "detail", f"{scene_name}__{algo_name}__detail.csv")
                            f_detail = open(detail_csv, "w", newline="", encoding="utf-8")
                            writer_d = csv.writer(f_detail)
                            writer_d.writerow([
                                "scene", "scenario", "desc", "pairs(valid_imitation)", "perturbed_pairs",
                                "mean_sad_steer", "mean_mae_throttle", "mean_mae_brake",
                                "mean_action_l1", "mean_action_l2",
                                "mean_min_dist_m(clean)", "mean_min_ttc_s(clean)", "mean_min_headway_s(clean)",
                                "risk_ttc_pairs", "mean_action_l2_on_ttc_risk",
                            ])
                        else:
                            f_detail = None
                            writer_d = None

                        with open(episodes_csv, "w", newline="", encoding="utf-8") as f_ep:
                            writer_ep = csv.writer(f_ep)
                            writer_ep.writerow([
                                "scene", "method", "scenario", "desc", "seed", "pair_idx",
                                "token_t", "token_tp1",
                                "expert_steer", "expert_throttle", "expert_brake",
                                "pred_steer", "pred_throttle", "pred_brake",
                                "action_l2_or_empty_if_gated",
                                "min_dist_m_clean", "min_ttc_s_clean", "min_headway_s_clean",
                                "is_ttc_risk",
                            ])

                            # debug
                            debug_this = bool(debug_perturbation)
                            if debug_only_first_scene_algo:
                                debug_this = debug_this and (scene_idx == 0) and (algo_idx == 0)

                            attack_summaries: Dict[str, OfflineScenarioSummary] = {}

                            # clean
                            clean_sum = eval_scene_one_algo_one_scenario_offline(
                                nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
                                scene=scene, scene_name=scene_name,
                                vehicle_monitor_msgs=vm_msgs,
                                policy=policy,
                                scenario=scenarios["clean"],
                                level=None,
                                ttc_risk_threshold_s=ttc_thr_scene,
                                max_pairs=max_pairs_eval,
                                episode_writer=writer_ep,
                                method_name=algo_name,
                                scenario_key="clean",
                                scenario_desc=scenarios["clean"].desc,
                                seed=seed,
                                imitation_speed_gate_mps=imitation_speed_gate_mps,
                                debug_perturbation=debug_this,
                                debug_pairs_print=debug_print_pairs,
                            )
                            if writer_d is not None:
                                writer_d.writerow(clean_sum.to_row(scene_name, "clean", scenarios["clean"].desc))

                            # attacks
                            for k in scen_keys:
                                sc = scenarios[k]
                                att_sum = eval_scene_one_algo_one_scenario_offline(
                                    nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
                                    scene=scene, scene_name=scene_name,
                                    vehicle_monitor_msgs=vm_msgs,
                                    policy=policy,
                                    scenario=sc,
                                    level=lvl.level,
                                    ttc_risk_threshold_s=ttc_thr_scene,
                                    max_pairs=max_pairs_eval,
                                    episode_writer=writer_ep,
                                    method_name=algo_name,
                                    scenario_key=k,
                                    scenario_desc=sc.desc,
                                    seed=seed,
                                    imitation_speed_gate_mps=imitation_speed_gate_mps,
                                    debug_perturbation=debug_this,
                                    debug_pairs_print=debug_print_pairs,
                                )
                                attack_summaries[k] = att_sum

                                if writer_d is not None:
                                    writer_d.writerow(att_sum.to_row(scene_name, k, sc.desc))

                                if lvl.name in {"L1", "L2", "L3"}:
                                    key = (lvl.name, k, algo_name)
                                    agg = detail_agg.get(key)
                                    if agg is None:
                                        agg = _DetailAgg()
                                        detail_agg[key] = agg
                                    _detail_accumulate(agg, att_sum)

                        if f_detail is not None:
                            f_detail.close()

                        eps = 1e-9
                        clean_l2 = max(eps, clean_sum.mean_action_l2)
                        clean_sad = max(eps, clean_sum.mean_sad_steer)
                        clean_l2_risk = max(eps, clean_sum.mean_action_l2_on_ttc_risk)

                        row = [
                            scene_name, algo_name,
                            f"{clean_sum.mean_action_l2:.6f}",
                            f"{clean_sum.mean_sad_steer:.6f}",
                            f"{clean_sum.mean_action_l2_on_ttc_risk:.6f}",
                        ]

                        for k in scen_keys:
                            att = max(eps, attack_summaries[k].mean_action_l2)
                            row.append(f"{(clean_l2 / att):.6f}")

                        for k in scen_keys:
                            att = max(eps, attack_summaries[k].mean_sad_steer)
                            row.append(f"{(clean_sad / att):.6f}")

                        for k in scen_keys:
                            att = max(eps, attack_summaries[k].mean_action_l2_on_ttc_risk)
                            row.append(f"{(clean_l2_risk / att):.6f}")

                        writer_g.writerow(row)

                        if lvl.name in {"L1", "L2", "L3"}:
                            merged_global_writer.writerow([lvl.name] + row)

            print(f"[offline] done intensity={lvl.name}. dir={run_dir}\n  - global: {global_csv}")

    finally:
        merged_global_f.close()

    intensity_order = ["L1", "L2", "L3"]
    method_order = list(algo_builders.keys())
    scenario_order = list(scen_keys)

    with open(merged_detail_path, "w", newline="", encoding="utf-8") as f_md:
        w = csv.writer(f_md)
        w.writerow([
            "attack", "method", "ep", "perturbed_pairs", "scenario", "desc",
            "n_scenes",
            "mean_sad_steer", "mean_mae_throttle", "mean_mae_brake",
            "mean_action_l1", "mean_action_l2",
            "mean_min_dist_m(clean)", "mean_min_ttc_s(clean)", "mean_min_headway_s(clean)",
            "risk_ttc_pairs", "mean_action_l2_on_ttc_risk",
        ])

        for scen in scenario_order:
            desc = scenarios[scen].desc if scen in scenarios else ""
            for lvl_name in intensity_order:
                for algo_name in method_order:
                    key = (lvl_name, scen, algo_name)
                    agg = detail_agg.get(key)
                    if agg is None or agg.pairs_sum <= 0:
                        continue
                    m = _detail_finalize_row(agg)
                    w.writerow([
                        lvl_name,
                        algo_name,
                        int(m["pairs_sum"]),
                        int(m["perturbed_pairs"]),
                        scen,
                        desc,
                        int(m["n_scenes"]),
                        f"{m['mean_sad_steer']:.6f}",
                        f"{m['mean_mae_throttle']:.6f}",
                        f"{m['mean_mae_brake']:.6f}",
                        f"{m['mean_action_l1']:.6f}",
                        f"{m['mean_action_l2']:.6f}",
                        f"{m['mean_min_dist_m(clean)']:.6f}",
                        f"{m['mean_min_ttc_s(clean)']:.6f}",
                        f"{m['mean_min_headway_s(clean)']:.6f}",
                        int(m["risk_ttc_pairs"]),
                        f"{m['mean_action_l2_on_ttc_risk']:.6f}",
                    ])

    print(f"[merged] global  (L1-L3) -> {merged_global_path}")
    print(f"[merged] detail  (L1-L3) -> {merged_detail_path}")
    print(f"[note] L0 per-scene detail kept under intensity_L0/detail/ (unchanged).")


def main():
    run_all_offline_and_save_csv(
        out_root="offline_eval_out",
        seed=0,
        max_pairs_eval=120,
        max_scenes_eval=50,
        cache_json_path=os.path.join(default_cache_dir(), "selected_scenes.json"),
        base_ttc_thr=1.5,
        risk_quantile=0.2,
        imitation_speed_gate_mps=1.0,
        intensity_names=["L0", "L1", "L2", "L3"],
        profile=True,
        debug_perturbation=True,
        debug_print_pairs=3,
        debug_only_first_scene_algo=True,
    )


if __name__ == '__main__':
    main()
