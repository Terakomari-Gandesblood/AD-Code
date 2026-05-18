import os
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

from config import config
from data_loader.nuscenes_loader import build_raw_state_from_sample


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


def _jst_now_str() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y%m%d_%H%M%S")


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


@dataclass
class SceneScore:
    scene: str
    pairs: int
    dist_m: float
    speed_mean: float
    speed_std: float
    moving_ratio: float
    control_activity: float
    location: str


def score_scene(
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        scene: dict,
        scene_name: str,
        vehicle_monitor_msgs: List[dict],
        max_pairs: Optional[int],
        v_move_thr: float,
        map_cache: Optional[Dict[str, NuScenesMap]] = None,
) -> Optional[SceneScore]:
    """
    计算 scene 的质量分数。
    - dist_m：ego 轨迹累计位移
    - moving_ratio：speed > v_move_thr 的比例
    - control_activity：steer/throttle/brake 三者 std 之和
    """
    tokens = _iter_scene_tokens(nusc, scene, max_pairs=max_pairs)
    if len(tokens) < 3:
        return None

    location = _get_scene_location(nusc, scene)

    if map_cache is not None and location in map_cache:
        nusc_map = map_cache[location]
    else:
        nusc_map = NuScenesMap(dataroot=config.map_root, map_name=location)
        if map_cache is not None:
            map_cache[location] = nusc_map

    n_pairs = len(tokens) - 1
    s0 = nusc.get("sample", tokens[0])
    rs = build_raw_state_from_sample(
        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
        sample=s0, vehicle_monitor_msgs=vehicle_monitor_msgs
    )

    xs, ys = [], []
    speeds, steers, thrs, brks = [], [], [], []

    for i in range(n_pairs):
        xs.append(float(_safe_float(rs.ego_info.location_x)))
        ys.append(float(_safe_float(rs.ego_info.location_y)))
        speeds.append(float(_safe_float(rs.ego_info.speed)))
        steers.append(float(_safe_float(rs.ego_info.steer)))
        thrs.append(float(_safe_float(rs.ego_info.throttle)))
        brks.append(float(_safe_float(rs.ego_info.brake)))

        s1 = nusc.get("sample", tokens[i + 1])
        rs = build_raw_state_from_sample(
            nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
            sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
        )

    dist = 0.0
    for i in range(1, len(xs)):
        dist += float(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))

    speeds_np = np.asarray(speeds, dtype=np.float32)
    moving_ratio = float(np.mean(speeds_np > float(v_move_thr)))

    control_activity = float(
        np.std(np.asarray(steers, dtype=np.float32))
        + np.std(np.asarray(thrs, dtype=np.float32))
        + np.std(np.asarray(brks, dtype=np.float32))
    )

    return SceneScore(
        scene=scene_name,
        pairs=int(n_pairs),
        dist_m=float(dist),
        speed_mean=float(np.mean(speeds_np)),
        speed_std=float(np.std(speeds_np)),
        moving_ratio=float(moving_ratio),
        control_activity=float(control_activity),
        location=location,
    )


def _filter_and_rank(
        scored: List[SceneScore],
        min_dist_m: float,
        min_moving_ratio: float,
        min_control_activity: float,
) -> List[SceneScore]:
    filtered = [
        x for x in scored
        if x.dist_m >= min_dist_m
           and x.moving_ratio >= min_moving_ratio
           and x.control_activity >= min_control_activity
    ]
    # 排序
    filtered.sort(key=lambda d: (d.dist_m, d.control_activity, d.moving_ratio), reverse=True)
    return filtered


def _choose_from_ranked(
        ranked: List[SceneScore],
        max_scenes: int,
        per_location_cap: Optional[int],
) -> List[SceneScore]:
    if per_location_cap is None:
        return ranked[:max_scenes]

    chosen: List[SceneScore] = []
    loc_cnt: Dict[str, int] = {}
    for x in ranked:
        if len(chosen) >= max_scenes:
            break
        c = loc_cnt.get(x.location, 0)
        if c >= per_location_cap:
            continue
        chosen.append(x)
        loc_cnt[x.location] = c + 1
    return chosen


def prescan_and_cache_scenes(
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        out_dir: str,
        *,
        max_pairs_prescan: Optional[int] = 120,
        max_scenes_default: int = 50,

        min_dist_m: float = 80.0,
        min_moving_ratio: float = 0.30,
        min_control_activity: float = 0.05,
        v_move_thr: float = 1.0,
        per_location_cap: Optional[int] = None,
        verbose_every: int = 50,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "selected_scenes.json")
    scores_csv_path = os.path.join(out_dir, "scene_scores.csv")

    scored: List[SceneScore] = []
    skipped_no_can = 0
    skipped_too_short = 0
    skipped_score_fail = 0

    map_cache: Dict[str, NuScenesMap] = {}

    total = len(nusc.scene)
    for idx, s in enumerate(nusc.scene):
        scene_name = s["name"]

        if verbose_every > 0 and (idx % verbose_every == 0 or idx == total - 1):
            print(f"[prescan] {idx + 1}/{total} scene={scene_name}")

        vm_msgs = _try_get_vehicle_monitor_msgs(nusc_can, scene_name)
        if vm_msgs is None:
            skipped_no_can += 1
            continue

        tokens = _iter_scene_tokens(nusc, s, max_pairs=max_pairs_prescan)
        if len(tokens) < 3:
            skipped_too_short += 1
            continue

        try:
            info = score_scene(
                nusc=nusc,
                nusc_can=nusc_can,
                scene=s,
                scene_name=scene_name,
                vehicle_monitor_msgs=vm_msgs,
                max_pairs=max_pairs_prescan,
                v_move_thr=v_move_thr,
                map_cache=map_cache,
            )
        except Exception:
            skipped_score_fail += 1
            continue

        if info is not None:
            scored.append(info)

    ranked = _filter_and_rank(
        scored,
        min_dist_m=min_dist_m,
        min_moving_ratio=min_moving_ratio,
        min_control_activity=min_control_activity,
    )
    chosen_scores = _choose_from_ranked(ranked, max_scenes_default, per_location_cap)

    payload: Dict[str, Any] = {
        "nuscenes_version": getattr(nusc, "version", ""),
        "cache_type": "scene_prescan",
        "created_at_jst": _jst_now_str(),
        "max_pairs_prescan": max_pairs_prescan,
        "thresholds": {
            "min_dist_m": min_dist_m,
            "min_moving_ratio": min_moving_ratio,
            "min_control_activity": min_control_activity,
            "v_move_thr": v_move_thr,
            "per_location_cap": per_location_cap,
            "max_scenes_default": max_scenes_default,
        },
        "stats": {
            "total_scenes": len(nusc.scene),
            "scored": len(scored),
            "filtered": len(ranked),
            "chosen_default": len(chosen_scores),
            "skipped_no_can": skipped_no_can,
            "skipped_too_short": skipped_too_short,
            "skipped_score_fail": skipped_score_fail,
        },
        "ranked_scenes": [x.scene for x in ranked],
        "ranked_scores": [asdict(x) for x in ranked],
        "chosen_scenes_default": [x.scene for x in chosen_scores],
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(scores_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "pairs", "dist_m", "speed_mean", "speed_std",
                    "moving_ratio", "control_activity", "location"])
        for x in ranked:
            w.writerow([
                x.scene, x.pairs,
                f"{x.dist_m:.3f}",
                f"{x.speed_mean:.3f}",
                f"{x.speed_std:.3f}",
                f"{x.moving_ratio:.3f}",
                f"{x.control_activity:.6f}",
                x.location
            ])

    print(
        f"[prescan] total={len(nusc.scene)} scored={len(scored)} filtered={len(ranked)} "
        f"chosen_default={len(chosen_scores)} | skip_no_can={skipped_no_can} "
        f"skip_short={skipped_too_short} skip_fail={skipped_score_fail}"
    )
    print(f"[prescan] cache saved: {cache_path}")
    print(f"[prescan] scores saved: {scores_csv_path}")
    print("[prescan] top-10 ranked scenes:")
    for i, x in enumerate(ranked[:10]):
        print(f"  {i + 1:02d} {x.scene} dist={x.dist_m:.1f} moving={x.moving_ratio:.2f} "
              f"ctrl={x.control_activity:.3f} v={x.speed_mean:.2f}±{x.speed_std:.2f} loc={x.location}")

    return cache_path, scores_csv_path


def load_ranked_scenes(cache_json_path: str, take: Optional[int] = None) -> List[str]:
    """
    从 scene prescan cache 中读取 scene 列表，并取前 take 个。
    """
    import os, json

    cache_json_path = os.path.abspath(cache_json_path)
    if not os.path.exists(cache_json_path):
        raise FileNotFoundError(f"Scene cache not found: {cache_json_path}")

    with open(cache_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scenes = (
            payload.get("chosen_scenes")
            or payload.get("ranked_scenes")
            or payload.get("scenes")
            or []
    )

    print("[scene-cache] path =", cache_json_path)
    print("[scene-cache] keys =", list(payload.keys()))
    print("[scene-cache] loaded scenes =", len(scenes))

    if not scenes:
        raise RuntimeError(
            f"No scenes loaded from cache. "
            f"Available keys: {list(payload.keys())}"
        )

    scenes = list(scenes)
    if take is not None:
        scenes = scenes[: int(take)]
    return scenes


def default_cache_dir() -> str:
    version = getattr(config, "NUSCENES_VERSION", "v1.0-trainval")
    return os.path.join("scene_cache", version)


def main():
    version = getattr(config, "NUSCENES_VERSION", "v1.0-trainval")
    out_dir = default_cache_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f"[init] NuScenes version={version}")
    nusc = NuScenes(version=version, dataroot=config.nuscenes_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=config.can_root)

    cache_path = os.path.join(out_dir, "selected_scenes.json")
    if os.path.exists(cache_path):
        scenes = load_ranked_scenes(cache_path, take=10)
        print(f"[scene-cache] exists -> {cache_path}")
        print(f"[scene-cache] sample top-10 ranked = {scenes}")
        return

    prescan_and_cache_scenes(
        nusc=nusc,
        nusc_can=nusc_can,
        out_dir=out_dir,
        max_pairs_prescan=120,
        max_scenes_default=50,
        min_dist_m=80.0,
        min_moving_ratio=0.30,
        min_control_activity=0.05,
        v_move_thr=1.0,
        per_location_cap=None,
        verbose_every=50,
    )


# 对 nuScenes scene 做质量预扫（CAN bus 可用性 + 运动距离 + moving_ratio + control_activity）
if __name__ == "__main__":
    main()
