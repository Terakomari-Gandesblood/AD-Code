import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

from config import config
from data_loader.nuscenes_loader import build_raw_state_from_sample
from environment.state import RawState
from evaluation.algorithm.behavioral_cloning import BCEvalPolicy


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _angle_wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _safe_scene_lookup(nusc: NuScenes, scene_name: str) -> dict:
    matches = [s for s in nusc.scene if s["name"] == scene_name]
    if not matches:
        avail = [s["name"] for s in nusc.scene]
        raise ValueError(f"scene_name={scene_name} not found in this version. Available={avail}")
    return matches[0]


def _get_sample_lidar_timestamp_us(nusc: NuScenes, sample: dict) -> int:
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    return int(sd["timestamp"])  # microseconds


def _postprocess_action(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    steer, thr, brk = a
    steer = float(np.clip(steer, -1.0, 1.0))
    thr = float(np.clip(thr, 0.0, 1.0))
    brk = float(np.clip(brk, 0.0, 1.0))

    # 同时较大时保留更大的那个
    if thr > 0.2 and brk > 0.2:
        if thr >= brk:
            brk = 0.0
        else:
            thr = 0.0

    return steer, thr, brk


@dataclass
class OfflineStepMetrics:
    # action regression
    sad_steer: float
    mae_throttle: float
    mae_brake: float
    action_l1: float
    action_l2: float

    # one-step proxy
    de_pos_1: float
    err_yaw_deg_1: float
    err_speed_1: float

    # risk proxies
    min_dist_m: float
    min_ttc_s: float
    min_headway_s: float

    # optional lane proxy (if exists)
    lane_yaw_diff_deg: Optional[float] = None


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


def _compute_min_dist_ttc_headway(rs: RawState) -> Tuple[float, float, float]:
    """
    使用 rs.surrounding_vehicles + ego pose/vel 计算：
    - min_dist: 所有周车最小欧氏距离
    - min_ttc: 前方且在靠近的车辆的最小 TTC
    - min_headway: 前方车辆的最小 time headway
    """
    sv = rs.surrounding_vehicles
    if sv is None or not getattr(sv, "vehicles", None):
        return float("inf"), float("inf"), float("inf")

    ego_yaw = float(rs.ego_info.rotation_yaw)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)

    ego_vx = float(rs.ego_info.velocity_x)
    ego_vy = float(rs.ego_info.velocity_y)
    ego_speed = max(1e-6, float(rs.ego_info.speed))

    min_dist = float("inf")
    min_ttc = float("inf")
    min_headway = float("inf")

    for v in sv.vehicles:
        dx = float(v.relative_x)
        dy = float(v.relative_y)
        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist

        # 投影到车体前向坐标：x_fwd>0 表示在前方
        x_fwd = dx * c + dy * s

        # 计算相对闭合速度
        dvx = ego_vx - float(v.velocity_x)
        dvy = ego_vy - float(v.velocity_y)
        v_closing = dvx * c + dvy * s  # >0 表示在逼近前方目标

        if x_fwd > 0.0:
            headway = x_fwd / ego_speed
            if headway < min_headway:
                min_headway = headway

            if v_closing > 1e-6:
                ttc = x_fwd / v_closing
                if ttc < min_ttc:
                    min_ttc = ttc

    return min_dist, min_ttc, min_headway


def _predict_next_state_simple(
        rs_t: RawState,
        action_pred: Tuple[float, float, float],
        dt_s: float,
        wheelbase_m: float = 2.7,
        max_steer_rad: float = 0.5,
        a_max: float = 3.0,
        b_max: float = 6.0,
) -> Tuple[float, float, float, float]:
    """
    简化 bicycle + 纵向加速度模型：
    - steer_norm in [-1,1] -> delta = steer_norm * max_steer_rad
    - a = throttle*a_max - brake*b_max
    返回 (x_pred, y_pred, yaw_pred, v_pred)
    """
    x = float(rs_t.ego_info.location_x)
    y = float(rs_t.ego_info.location_y)
    yaw = float(rs_t.ego_info.rotation_yaw)
    v = float(rs_t.ego_info.speed)

    steer, thr, brk = action_pred
    steer = float(np.clip(steer, -1.0, 1.0))
    thr = float(np.clip(thr, 0.0, 1.0))
    brk = float(np.clip(brk, 0.0, 1.0))

    delta = steer * max_steer_rad
    a = thr * a_max - brk * b_max

    # 更新速度
    v2 = max(0.0, v + a * dt_s)

    # yaw 变化
    beta = math.tan(delta)
    yaw_rate = (v / wheelbase_m) * beta
    yaw2 = _wrap_to_pi(yaw + yaw_rate * dt_s)

    # 位置更新
    x2 = x + v * math.cos(yaw) * dt_s
    y2 = y + v * math.sin(yaw) * dt_s

    return x2, y2, yaw2, v2


def compute_offline_step_metrics(
        rs_t,
        rs_tp1,
        action_pred: Tuple[float, float, float],
        action_expert: Tuple[float, float, float],
        dt_s: float,
) -> OfflineStepMetrics:
    st_p, th_p, br_p = action_pred
    st_e, th_e, br_e = action_expert

    sad_steer = abs(st_p - st_e)
    mae_throttle = abs(th_p - th_e)
    mae_brake = abs(br_p - br_e)
    action_l1 = abs(st_p - st_e) + abs(th_p - th_e) + abs(br_p - br_e)
    action_l2 = float(math.sqrt((st_p - st_e) ** 2 + (th_p - th_e) ** 2 + (br_p - br_e) ** 2))

    # trajectory proxy：用 (t -> t+1) 的 ego 位移/航向/速度作为参考轨迹的下一点
    # de_pos_1: 以 rs_t 的位置当 pred(t+1)，与真实 rs_{t+1} 的位置做差
    x0, y0 = _safe_float(rs_t.ego_info.location_x), _safe_float(rs_t.ego_info.location_y)
    x1, y1 = _safe_float(rs_tp1.ego_info.location_x), _safe_float(rs_tp1.ego_info.location_y)
    de_pos_1 = float(math.hypot(x1 - x0, y1 - y0))

    yaw0 = _safe_float(rs_t.ego_info.rotation_yaw)
    yaw1 = _safe_float(rs_tp1.ego_info.rotation_yaw)
    err_yaw_deg_1 = float(abs(math.degrees(_angle_wrap_pi(yaw1 - yaw0))))

    v0 = _safe_float(rs_t.ego_info.speed)
    v1 = _safe_float(rs_tp1.ego_info.speed)
    err_speed_1 = float(abs(v1 - v0))

    # safety proxy：最小距离 / TTC / headway
    min_dist = float("inf")
    min_ttc = float("inf")
    min_headway = float("inf")

    sv = getattr(rs_t, "surrounding_vehicles", None)
    if sv is not None and getattr(sv, "vehicles", None):
        # ego 速度向量
        evx = _safe_float(rs_t.ego_info.velocity_x)
        evy = _safe_float(rs_t.ego_info.velocity_y)
        ego_speed = max(1e-3, float(math.hypot(evx, evy)))

        for veh in sv.vehicles:
            rx = _safe_float(getattr(veh, "relative_x", 0.0))
            ry = _safe_float(getattr(veh, "relative_y", 0.0))
            dist = float(math.hypot(rx, ry))
            if dist < min_dist:
                min_dist = dist

            # TTC：沿相对位置方向的接近速度
            # rel_vel = (v_other - v_ego) dot unit_rel
            ovx = _safe_float(getattr(veh, "velocity_x", 0.0))
            ovy = _safe_float(getattr(veh, "velocity_y", 0.0))
            ux, uy = (rx / max(dist, 1e-6), ry / max(dist, 1e-6))
            closing = (ovx - evx) * ux + (ovy - evy) * uy  # >0 表示在远离，<0 表示在接近
            if closing < -1e-3:
                ttc = dist / (-closing)
                if ttc < min_ttc:
                    min_ttc = ttc

            # Headway：只看前向车辆
            hx, hy = math.cos(yaw0), math.sin(yaw0)
            forward_proj = rx * hx + ry * hy
            if forward_proj > 0.0:
                headway = forward_proj / ego_speed
                if headway < min_headway:
                    min_headway = headway

    if not np.isfinite(min_dist):
        min_dist = float("inf")
    if not np.isfinite(min_ttc):
        min_ttc = float("inf")
    if not np.isfinite(min_headway):
        min_headway = float("inf")

    lane_yaw_diff = None
    li = getattr(rs_t, "lane_info", None)
    if li is not None and hasattr(li, "yaw_diff"):
        lane_yaw_diff = float(getattr(li, "yaw_diff"))

    return OfflineStepMetrics(
        sad_steer=sad_steer,
        mae_throttle=mae_throttle,
        mae_brake=mae_brake,
        action_l1=action_l1,
        action_l2=action_l2,
        de_pos_1=de_pos_1,
        err_yaw_deg_1=err_yaw_deg_1,
        err_speed_1=err_speed_1,
        min_dist_m=min_dist,
        min_ttc_s=min_ttc,
        min_headway_s=min_headway,
        lane_yaw_diff_deg=lane_yaw_diff,
    )


def _summ_stats(x: List[float]) -> Dict[str, float]:
    arr = np.array(x, dtype=np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def eval_scene_offline(
        nusc,
        nusc_can,
        nusc_map,
        scene: dict,
        vehicle_monitor_msgs: List[dict],
        policy,
        dt_s: float = 0.5,  # nuScenes keyframe 默认 2Hz
        max_pairs: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    # 取出 sample token 序列
    tokens: List[str] = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        s = nusc.get("sample", tok)
        tok = s["next"]

    if len(tokens) < 2:
        raise ValueError("Scene has <2 samples; cannot form (t,t+1) pairs.")

    n_pairs_total = len(tokens) - 1
    n_pairs = n_pairs_total if max_pairs is None else min(n_pairs_total, max_pairs)

    # 滚动构建 RawState：rs_t, rs_tp1
    s0 = nusc.get("sample", tokens[0])
    rs_t = build_raw_state_from_sample(
        nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
        sample=s0, vehicle_monitor_msgs=vehicle_monitor_msgs
    )

    # 收集所有 step metrics
    sad_list, thr_list, brk_list, l1_list, l2_list = [], [], [], [], []
    de_list, yawerr_list, verr_list = [], [], []
    mindist_list, minttc_list, minhead_list = [], [], []
    lane_yaw_list = []

    # 风险比例阈值
    TTC_DANGER = 1.5
    HEADWAY_DANGER = 1.0
    MINDIST_DANGER = 3.0
    n_ttc_danger = 0
    n_headway_danger = 0
    n_dist_danger = 0

    for i in range(n_pairs):
        s1 = nusc.get("sample", tokens[i + 1])
        rs_tp1 = build_raw_state_from_sample(
            nusc=nusc, nusc_can=nusc_can, nusc_map=nusc_map,
            sample=s1, vehicle_monitor_msgs=vehicle_monitor_msgs
        )

        # expert action
        expert_action = (
            float(rs_t.ego_info.steer),
            float(rs_t.ego_info.throttle),
            float(rs_t.ego_info.brake),
        )

        # predicted action：来自模型
        pred_action = policy.predict(rs_t)
        pred_action = _postprocess_action(pred_action)

        m = compute_offline_step_metrics(
            rs_t=rs_t,
            rs_tp1=rs_tp1,
            action_pred=pred_action,
            action_expert=expert_action,
            dt_s=dt_s,
        )

        sad_list.append(m.sad_steer)
        thr_list.append(m.mae_throttle)
        brk_list.append(m.mae_brake)
        l1_list.append(m.action_l1)
        l2_list.append(m.action_l2)

        de_list.append(m.de_pos_1)
        yawerr_list.append(m.err_yaw_deg_1)
        verr_list.append(m.err_speed_1)

        mindist_list.append(m.min_dist_m if np.isfinite(m.min_dist_m) else 1e9)
        minttc_list.append(m.min_ttc_s if np.isfinite(m.min_ttc_s) else 1e9)
        minhead_list.append(m.min_headway_s if np.isfinite(m.min_headway_s) else 1e9)

        if m.lane_yaw_diff_deg is not None:
            lane_yaw_list.append(abs(float(m.lane_yaw_diff_deg)))

        # 风险计数
        if np.isfinite(m.min_ttc_s) and m.min_ttc_s < TTC_DANGER:
            n_ttc_danger += 1
        if np.isfinite(m.min_headway_s) and m.min_headway_s < HEADWAY_DANGER:
            n_headway_danger += 1
        if np.isfinite(m.min_dist_m) and m.min_dist_m < MINDIST_DANGER:
            n_dist_danger += 1

        # 滑动窗口
        rs_t = rs_tp1

    # 4) 聚合统计
    out = {
        "SAD_steer": _summ_stats(sad_list),
        "MAE_throttle": _summ_stats(thr_list),
        "MAE_brake": _summ_stats(brk_list),
        "Action_L1": _summ_stats(l1_list),
        "Action_L2": _summ_stats(l2_list),

        "DE_pos_1m_proxy": _summ_stats(de_list),
        "Yaw_err_deg_1_proxy": _summ_stats(yawerr_list),
        "Speed_err_mps_1_proxy": _summ_stats(verr_list),

        "MinDist_m": _summ_stats(mindist_list),
        "MinTTC_s": _summ_stats(minttc_list),
        "MinHeadway_s": _summ_stats(minhead_list),
    }
    if len(lane_yaw_list) > 0:
        out["LaneYawDiff_deg"] = _summ_stats(lane_yaw_list)

    out["RiskRates"] = {
        "pairs": float(n_pairs),
        "ttc_lt_1p5_ratio": float(n_ttc_danger / max(1, n_pairs)),
        "headway_lt_1p0_ratio": float(n_headway_danger / max(1, n_pairs)),
        "mindist_lt_3m_ratio": float(n_dist_danger / max(1, n_pairs)),
    }
    return out


# 随机动作测试
def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=config.nuscenes_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=config.can_root)

    avail_scenes = [s["name"] for s in nusc.scene]
    print("当前版本可用 scenes:", avail_scenes)

    scene_name = "scene-0916"
    scene = _safe_scene_lookup(nusc, scene_name)

    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    nusc_map = NuScenesMap(dataroot=config.map_root, map_name=location)
    print(f"[scene] {scene_name} | location={location}")

    vehicle_monitor_msgs = nusc_can.get_messages(scene_name, "vehicle_monitor")
    vehicle_monitor_msgs = sorted(vehicle_monitor_msgs, key=lambda m: int(m.get("utime", 0)))
    print(f"vehicle_monitor msgs={len(vehicle_monitor_msgs)}")

    # 初始化 policy
    policy = BCEvalPolicy("bc_residual_best.pt")

    dt_s = 0.5

    summary = eval_scene_offline(
        nusc=nusc,
        nusc_can=nusc_can,
        nusc_map=nusc_map,
        scene=scene,
        vehicle_monitor_msgs=vehicle_monitor_msgs,
        policy=policy,
        dt_s=dt_s,
        max_pairs=None,
    )

    print("\n=== Scene Offline Summary ===")
    for k, v in summary.items():
        if k == "RiskRates":
            print(f"{k}: {v}")
        else:
            print(f"{k}: mean={v['mean']:.6f}, std={v['std']:.6f}, p50={v['p50']:.6f}, p90={v['p90']:.6f}")


if __name__ == "__main__":
    main()
