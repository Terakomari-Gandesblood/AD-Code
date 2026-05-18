import os
import json
import csv
import math
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

from config import config
from data_loader.nuscenes_loader import build_raw_state_from_sample
from environment.state import RawState


def _jst_now_str() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y%m%d_%H%M%S")


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


def _ego_speed_mps(rs: RawState) -> float:
    sp = getattr(rs.ego_info, "speed", None)
    if sp is not None:
        v = _safe_float(sp, default=0.0)
        if v > 0:
            return float(v)

    evx = _safe_float(getattr(rs.ego_info, "velocity_x", 0.0))
    evy = _safe_float(getattr(rs.ego_info, "velocity_y", 0.0))
    return float(math.hypot(evx, evy))


def _compute_min_dist_ttc_headway(rs: RawState) -> Tuple[float, float, float]:
    """
    - min_dist: 所有周车最小欧氏距离
    - min_ttc: 沿相对位置方向 closing speed < 0 才计算
    - min_headway: 仅前方车辆 forward_proj > 0，headway=forward_proj/ego_speed
    """
    min_dist = float("inf")
    min_ttc = float("inf")
    min_headway = float("inf")

    evx = _safe_float(getattr(rs.ego_info, "velocity_x", 0.0))
    evy = _safe_float(getattr(rs.ego_info, "velocity_y", 0.0))
    ego_speed = max(1e-3, float(math.hypot(evx, evy)))

    yaw = _safe_float(getattr(rs.ego_info, "rotation_yaw", 0.0))
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
        closing = (ovx - evx) * ux + (ovy - evy) * uy  # <0 approaching
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
        base_thr: float,
        q: float,
        min_count: int = 30,
        ignore_upper: float = 1e6,
) -> Tuple[float, dict]:
    vals = np.asarray(min_ttc_list, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals < float(ignore_upper)]
    info = {"base_thr": float(base_thr), "q": float(q), "valid_n": int(vals.size)}

    if vals.size < int(min_count):
        info["mode"] = "fixed_base (insufficient_valid_samples)"
        info["thr"] = float(base_thr)
        return float(base_thr), info

    qv = float(np.quantile(vals, float(q)))
    thr = float(max(float(base_thr), qv))
    info["mode"] = "max(base, quantile)"
    info["quantile_value"] = float(qv)
    info["thr"] = float(thr)
    return thr, info


def load_ranked_scenes(cache_json_path: str, take: Optional[int] = None) -> List[str]:
    cache_json_path = os.path.abspath(cache_json_path)
    if not os.path.exists(cache_json_path):
        raise FileNotFoundError(f"Scene cache not found: {cache_json_path}")

    with open(cache_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scenes = (
            payload.get("chosen_scenes")
            or payload.get("ranked_scenes")
            or payload.get("chosen_scenes_default")
            or payload.get("scenes")
            or []
    )

    print("[scene-cache] path =", cache_json_path)
    print("[scene-cache] keys =", list(payload.keys()))
    print("[scene-cache] loaded scenes =", len(scenes))

    if not scenes:
        raise RuntimeError(f"No scenes loaded from cache. Available keys: {list(payload.keys())}")

    scenes = list(scenes)
    if take is not None:
        scenes = scenes[: int(take)]
    return scenes


@dataclass
class SceneCleanRiskSummary:
    scene: str
    pairs_total: int
    pairs_valid: int
    pairs_low_speed_skipped: int
    pairs_used_for_proxy: int
    ttc_thr_s: float
    thr_mode: str
    thr_quantile_value: float

    mean_min_dist_m_clean: float
    mean_min_ttc_s_clean: float
    mean_min_headway_s_clean: float

    dist_valid_n: int
    ttc_valid_n: int
    headway_valid_n: int

    risk_ttc_pairs: int
    risk_ttc_pairs_ratio: float

    # diagnostics
    no_vehicle_ratio: float


def _mean_finite(xs: List[float]) -> Tuple[float, int]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0
    return float(arr.mean()), int(arr.size)


def summarize_one_scene(
        *,
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        nusc_map: NuScenesMap,
        scene_name: str,
        scene: dict,
        vehicle_monitor_msgs: List[dict],
        max_pairs: Optional[int],
        base_ttc_thr: float,
        risk_quantile: float,
        speed_gate_mps: Optional[float],
) -> SceneCleanRiskSummary:
    tokens = _iter_scene_tokens(nusc, scene, max_pairs=max_pairs)
    if len(tokens) < 2:
        raise ValueError(f"{scene_name} has <2 samples.")

    n_pairs_total = len(tokens) - 1

    s0 = nusc.get("sample", tokens[0])
    rs = build_raw_state_from_sample(
        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
        sample=s0, vehicle_monitor_msgs=vehicle_monitor_msgs
    )

    min_ttc_all: List[float] = []

    # valid pairs 上统计
    valid_min_dist: List[float] = []
    valid_min_ttc: List[float] = []
    valid_min_head: List[float] = []

    pairs_valid = 0
    no_vehicle_valid = 0
    low_speed_skipped = 0  # NEW

    for i in range(n_pairs_total):
        min_dist, min_ttc, min_head = _compute_min_dist_ttc_headway(rs)

        # 阈值
        min_ttc_all.append(float(min_ttc))

        # speed gate
        if speed_gate_mps is not None and _ego_speed_mps(rs) < float(speed_gate_mps):
            low_speed_skipped += 1
            s1 = nusc.get("sample", tokens[i + 1])
            rs = build_raw_state_from_sample(
                nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
                sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
            )
            continue

        pairs_valid += 1

        if (not np.isfinite(min_dist)) and (not np.isfinite(min_ttc)) and (not np.isfinite(min_head)):
            no_vehicle_valid += 1

        if np.isfinite(min_dist):
            valid_min_dist.append(float(min_dist))
        if np.isfinite(min_ttc):
            valid_min_ttc.append(float(min_ttc))
        if np.isfinite(min_head):
            valid_min_head.append(float(min_head))

        s1 = nusc.get("sample", tokens[i + 1])
        rs = build_raw_state_from_sample(
            nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
            sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
        )

    mean_dist, n_dist = _mean_finite(valid_min_dist)
    mean_ttc, n_ttc = _mean_finite(valid_min_ttc)
    mean_head, n_head = _mean_finite(valid_min_head)

    # scene-level 自适应阈值
    ttc_thr, dbg = _adaptive_ttc_threshold(
        min_ttc_all, base_thr=base_ttc_thr, q=risk_quantile
    )
    thr_mode = str(dbg.get("mode", ""))
    thr_qv = float(dbg.get("quantile_value", float("nan")))

    # risk subset
    risk_cnt = 0
    for v in valid_min_ttc:
        if np.isfinite(v) and v < ttc_thr:
            risk_cnt += 1

    pairs_valid_safe = max(1, pairs_valid)
    risk_ratio = float(risk_cnt / pairs_valid_safe)
    no_vehicle_ratio = float(no_vehicle_valid / pairs_valid_safe)

    return SceneCleanRiskSummary(
        scene=scene_name,
        pairs_total=int(n_pairs_total),
        pairs_valid=int(pairs_valid),

        pairs_low_speed_skipped=int(low_speed_skipped),
        pairs_used_for_proxy=int(pairs_valid),

        ttc_thr_s=float(ttc_thr),
        thr_mode=thr_mode,
        thr_quantile_value=thr_qv,

        mean_min_dist_m_clean=mean_dist,
        mean_min_ttc_s_clean=mean_ttc,
        mean_min_headway_s_clean=mean_head,

        dist_valid_n=n_dist,
        ttc_valid_n=n_ttc,
        headway_valid_n=n_head,

        risk_ttc_pairs=int(risk_cnt),
        risk_ttc_pairs_ratio=float(risk_ratio),
        no_vehicle_ratio=float(no_vehicle_ratio),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=os.path.join("offline_eval_out", f"clean_risk_{_jst_now_str()}"))
    ap.add_argument("--cache_json", type=str,
                    default=os.path.join("scene_cache", getattr(config, "NUSCENES_VERSION", "v1.0-trainval"),
                                         "selected_scenes.json"))
    ap.add_argument("--take_scenes", type=int, default=50)
    ap.add_argument("--max_pairs", type=int, default=120)

    ap.add_argument("--base_ttc_thr", type=float, default=1.5)
    ap.add_argument("--risk_quantile", type=float, default=0.2)
    ap.add_argument("--speed_gate_mps", type=float, default=1.0,
                    help="与离线 imitation 口径一致；设为 0 可近似关闭 gate")

    args = ap.parse_args()

    speed_gate = None if args.speed_gate_mps is None else float(args.speed_gate_mps)
    if speed_gate is not None and speed_gate <= 0:
        speed_gate = None

    os.makedirs(args.out_dir, exist_ok=True)
    per_scene_csv = os.path.join(args.out_dir, "per_scene_clean_risk.csv")
    global_csv = os.path.join(args.out_dir, "global_clean_risk.csv")
    meta_json = os.path.join(args.out_dir, "meta.json")

    t0 = time.perf_counter()
    version = getattr(config, "NUSCENES_VERSION", "v1.0-mini")
    nusc = NuScenes(version=version, dataroot=config.nuscenes_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=config.can_root)
    t1 = time.perf_counter()

    scene_names = load_ranked_scenes(args.cache_json, take=args.take_scenes)
    map_cache: Dict[str, NuScenesMap] = {}

    rows: List[SceneCleanRiskSummary] = []

    total_pairs_total = 0
    total_pairs_valid = 0
    total_risk_pairs = 0
    total_low_speed_skipped = 0

    sum_dist = 0.0
    sum_ttc = 0.0
    sum_head = 0.0
    W_dist = 0
    W_ttc = 0
    W_head = 0

    skipped_no_can = 0
    skipped_failed = 0

    for idx, scene_name in enumerate(scene_names):
        try:
            scene = _safe_scene_lookup(nusc, scene_name)
            location = _get_scene_location(nusc, scene)
            if location not in map_cache:
                map_cache[location] = NuScenesMap(dataroot=config.map_root, map_name=location)
            nusc_map = map_cache[location]

            vm_msgs = _try_get_vehicle_monitor_msgs(nusc_can, scene_name)
            if vm_msgs is None:
                skipped_no_can += 1
                print(f"[skip] no CAN vehicle_monitor for scene={scene_name}")
                continue

            s = summarize_one_scene(
                nusc=nusc,
                nusc_can=nusc_can,
                nusc_map=nusc_map,
                scene_name=scene_name,
                scene=scene,
                vehicle_monitor_msgs=vm_msgs,
                max_pairs=args.max_pairs,
                base_ttc_thr=float(args.base_ttc_thr),
                risk_quantile=float(args.risk_quantile),
                speed_gate_mps=speed_gate,
            )
            rows.append(s)

            total_pairs_total += int(s.pairs_total)
            total_pairs_valid += int(s.pairs_valid)
            total_risk_pairs += int(s.risk_ttc_pairs)
            total_low_speed_skipped += int(s.pairs_low_speed_skipped)

            if s.dist_valid_n > 0:
                W_dist += int(s.dist_valid_n)
                sum_dist += float(s.mean_min_dist_m_clean) * float(s.dist_valid_n)

            if s.ttc_valid_n > 0:
                W_ttc += int(s.ttc_valid_n)
                sum_ttc += float(s.mean_min_ttc_s_clean) * float(s.ttc_valid_n)

            if s.headway_valid_n > 0:
                W_head += int(s.headway_valid_n)
                sum_head += float(s.mean_min_headway_s_clean) * float(s.headway_valid_n)

            print(
                f"[{idx + 1}/{len(scene_names)}] {scene_name} "
                f"pairs_total={s.pairs_total} "
                f"used_for_proxy={s.pairs_used_for_proxy} "
                f"low_speed_skipped={s.pairs_low_speed_skipped} "
                f"thr={s.ttc_thr_s:.3f}s "
                f"risk={s.risk_ttc_pairs} (risk/used={s.risk_ttc_pairs_ratio:.3f}) "
                f"no_vehicle_ratio={s.no_vehicle_ratio:.3f}"
            )

        except Exception as e:
            skipped_failed += 1
            print(f"[skip] scene={scene_name} failed: {type(e).__name__}: {e}")
            continue

    # write per-scene
    with open(per_scene_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scene",
            "pairs_total", "pairs_valid",
            "ttc_thr_s", "thr_mode", "thr_quantile_value",
            "mean_min_dist_m(clean)", "mean_min_ttc_s(clean)", "mean_min_headway_s(clean)",
            "risk_ttc_pairs", "risk_ttc_pairs_ratio",
            "no_vehicle_ratio(valid_pairs)",
        ])
        for s in rows:
            w.writerow([
                s.scene,
                s.pairs_total, s.pairs_valid,
                f"{s.ttc_thr_s:.6f}", s.thr_mode,
                f"{s.thr_quantile_value:.6f}" if np.isfinite(s.thr_quantile_value) else "",
                f"{s.mean_min_dist_m_clean:.6f}",
                f"{s.mean_min_ttc_s_clean:.6f}",
                f"{s.mean_min_headway_s_clean:.6f}",
                s.risk_ttc_pairs,
                f"{s.risk_ttc_pairs_ratio:.6f}",
                f"{s.no_vehicle_ratio:.6f}",
            ])

    # write global
    mean_dist = (sum_dist / max(1, W_dist)) if W_dist > 0 else 0.0
    mean_ttc = (sum_ttc / max(1, W_ttc)) if W_ttc > 0 else 0.0
    mean_head = (sum_head / max(1, W_head)) if W_head > 0 else 0.0
    risk_ratio = float(total_risk_pairs / max(1, total_pairs_valid))

    with open(global_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenes_used",
            "pairs_total_sum", "pairs_valid_sum",
            "mean_min_dist_m(clean)", "mean_min_ttc_s(clean)", "mean_min_headway_s(clean)",
            "risk_ttc_pairs", "risk_ttc_pairs_ratio",
        ])
        w.writerow([
            len(rows),
            total_pairs_total, total_pairs_valid,
            f"{mean_dist:.6f}", f"{mean_ttc:.6f}", f"{mean_head:.6f}",
            total_risk_pairs, f"{risk_ratio:.6f}",
        ])

    t2 = time.perf_counter()
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({
            "nuscenes_version": version,
            "cache_json": os.path.abspath(args.cache_json),
            "take_scenes": args.take_scenes,
            "max_pairs": args.max_pairs,
            "base_ttc_thr": args.base_ttc_thr,
            "risk_quantile": args.risk_quantile,
            "speed_gate_mps": speed_gate,
            "out_dir": os.path.abspath(args.out_dir),
            "skipped": {
                "no_can_vehicle_monitor": skipped_no_can,
                "failed_exception": skipped_failed,
            },
            "profile_sec": {
                "init_nusc_can": t1 - t0,
                "total": t2 - t0,
            },
        }, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("out_dir:", os.path.abspath(args.out_dir))
    print("per_scene:", os.path.abspath(per_scene_csv))
    print("global  :", os.path.abspath(global_csv))
    print(f"GLOBAL pairs_valid={total_pairs_valid} risk_pairs={total_risk_pairs} ratio={risk_ratio:.6f}")

    pairs_used_for_proxy_sum = max(1, total_pairs_valid)
    low_speed_skipped_sum = total_low_speed_skipped
    low_speed_skipped_ratio = low_speed_skipped_sum / max(1, total_pairs_total)

    risk_ratio_used = total_risk_pairs / max(1, pairs_used_for_proxy_sum)
    risk_ratio_total = total_risk_pairs / max(1, total_pairs_total)

    print("\n=== SPEED GATE STATS (PRINT ONLY) ===")
    print(f"speed_gate_mps = {speed_gate}")
    print(f"pairs_total_sum       = {total_pairs_total}")
    print(f"pairs_used_for_proxy  = {pairs_used_for_proxy_sum}")
    print(f"low_speed_skipped_sum = {low_speed_skipped_sum}  (ratio vs total = {low_speed_skipped_ratio:.4f})")
    print(f"risk_pairs_sum        = {total_risk_pairs}")
    print(f"risk_ratio (risk/used_for_proxy) = {risk_ratio_used:.6f}")
    print(f"risk_ratio (risk/pairs_total)    = {risk_ratio_total:.6f}")


if __name__ == "__main__":
    main()
