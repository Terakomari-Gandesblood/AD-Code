import math
from typing import Optional, Tuple, List

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None

# 车道中心线离散工具
try:
    from nuscenes.map_expansion import arcline_path_utils as arcline_path_utils
except Exception:
    arcline_path_utils = None

from config import config
from entity.enums import TrafficLightState, CollisionType
from entity.types import VehicleInfo, SurroundingVehicles, LaneInfo, TrafficLightInfo
from environment.state import RawState

CONVERT_TO_CARLA_FRAME = True  # y 取反 + yaw 镜像
ASSUME_SPEED_KMH = True  # vehicle_monitor.vehicle_speed 常见是 km/h
STEER_SIGN = 1.0
STEER_MAX_DEG = 390.0  # 方向盘角经验值

LANE_QUERY_RADIUS = 2.0  # 找最近车道半径
LANE_RESOLUTION = 1.0  # 中心线离散步长
LANE_LOOKAHEAD_M = 10.0  # 判断阈值


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _safe_scene_lookup(nusc: NuScenes, scene_name: str) -> dict:
    scenes = [s for s in nusc.scene if s.get("name") == scene_name]
    if not scenes:
        avail = [s.get("name") for s in nusc.scene]
        raise ValueError(f"scene_name={scene_name} 不在当前 NuScenes({nusc.version}) 中。可用: {avail}")
    return scenes[0]


def _find_nearest_can_msg(msgs: List[dict], utime: int) -> Optional[dict]:
    if not msgs:
        return None

    lo, hi = 0, len(msgs) - 1
    target = int(utime)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(msgs[mid].get("utime", 0)) < target:
            lo = mid + 1
        else:
            hi = mid

    cand = [msgs[lo]]
    if lo - 1 >= 0:
        cand.append(msgs[lo - 1])
    best = min(cand, key=lambda m: abs(int(m.get("utime", 0)) - target))
    return best


def _quat_to_rotmat(q_list) -> np.ndarray:
    if Quaternion is None:
        raise RuntimeError("pyquaternion 未安装/不可用，请 pip install pyquaternion 或检查 devkit 环境。")
    q = Quaternion(q_list)
    return np.array(q.rotation_matrix, dtype=np.float64)


def _rotmat_to_yaw_pitch_roll(R: np.ndarray) -> Tuple[float, float, float]:
    """
    统一用 ZYX坐标，从旋转矩阵恢复欧拉角。
    yaw: around Z
    pitch: around Y
    roll: around X
    """
    # 参考常见 ZYX 分解
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.asin(-float(np.clip(R[2, 0], -1.0, 1.0)))
    roll = math.atan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def _quat_to_yaw_pitch_roll(q_list) -> Tuple[float, float, float]:
    R = _quat_to_rotmat(q_list)
    return _rotmat_to_yaw_pitch_roll(R)


def _to_carla_frame_xy(x: float, y: float) -> Tuple[float, float]:
    if not CONVERT_TO_CARLA_FRAME:
        return x, y
    return x, -y


def _to_carla_frame_yaw(yaw: float) -> float:
    if not CONVERT_TO_CARLA_FRAME:
        return yaw
    return -yaw


def _maybe_speed_to_mps(speed_val: float) -> float:
    s = float(speed_val)
    return s / 3.6 if ASSUME_SPEED_KMH else s


def _normalize_01_like(v: float, *, name: str) -> float:
    """
    兼容 nuScenes CAN 里常见的几种量纲：
    - 已经是 0..1
    - 0..100 的百分比
    - 极少数异常值做 clip
    """
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        return 0.0
    ax = abs(x)
    # 典型百分比编码
    if ax > 1.5:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def _normalize_steer(raw_steer_angle_deg: float) -> float:
    """
    将 nuScenes CAN 中的方向盘角映射到 [-1, 1]
    """
    steer = (float(raw_steer_angle_deg) / float(STEER_MAX_DEG)) * float(STEER_SIGN)
    return float(np.clip(steer, -1.0, 1.0))


def _build_lane_info(nusc_map: NuScenesMap, x: float, y: float, ego_yaw: float) -> Optional[LaneInfo]:
    lane_token = None
    try:
        lane_token = nusc_map.get_closest_lane(x, y, radius=LANE_QUERY_RADIUS)
    except TypeError:
        try:
            lane_token = nusc_map.get_closest_lane(x, y)
        except Exception:
            lane_token = None
    except Exception:
        lane_token = None

    if not lane_token:
        return None

    li = LaneInfo()
    li.lane_id = int(hash(lane_token) % 10000)

    is_junction = False
    try:
        if hasattr(nusc_map, "lane_connector") and isinstance(nusc_map.lane_connector, list):
            lc_tokens = {r["token"] for r in nusc_map.lane_connector if isinstance(r, dict) and "token" in r}
            is_junction = lane_token in lc_tokens
    except Exception:
        is_junction = False
    li.is_junction = bool(is_junction)

    # 默认不限制
    li.has_next = True
    li.yaw_diff = 0.0
    li.lane_change = True
    li.speed_limit = 999.0

    # 能算就算
    if arcline_path_utils is None:
        return li

    try:
        path = nusc_map.get_arcline_path(lane_token)
        pts = arcline_path_utils.discretize_lane(path, resolution_meters=LANE_RESOLUTION)
        pts = np.array(pts, dtype=np.float32).reshape(-1, 3)

        if len(pts) < 2:
            return li

        # 找最近点
        d2 = np.sum((pts[:, :2] - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
        idx = int(np.argmin(d2))

        # 切线方向
        if idx < len(pts) - 1:
            v = pts[idx + 1, :2] - pts[idx, :2]
        else:
            v = pts[idx, :2] - pts[idx - 1, :2]

        lane_yaw = math.atan2(float(v[1]), float(v[0]))
        yaw_diff = _wrap_to_pi(lane_yaw - float(ego_yaw))
        li.yaw_diff = float(math.degrees(yaw_diff))

        # 剩余长度
        if idx < len(pts) - 1:
            seg = pts[idx:, :2]
            seg_len = float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1)))
            li.has_next = bool(seg_len > LANE_LOOKAHEAD_M)
        else:
            li.has_next = True
    except Exception:
        # 保持默认
        pass

    return li


def build_raw_state_from_sample(
        nusc: NuScenes,
        nusc_can: NuScenesCanBus,
        nusc_map: NuScenesMap,
        sample: dict,
        vehicle_monitor_msgs: List[dict],
) -> RawState:
    rs = RawState()

    #  ego
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    nx, ny, nz = ego_pose["translation"]
    nu_yaw, nu_pitch, nu_roll = _quat_to_yaw_pitch_roll(ego_pose["rotation"])

    # 坐标系转换
    x, y = _to_carla_frame_xy(float(nx), float(ny))
    z = float(nz)
    yaw = _to_carla_frame_yaw(float(nu_yaw))

    rs.ego_info.location_x = x
    rs.ego_info.location_y = y
    rs.ego_info.location_z = z
    rs.ego_info.rotation_yaw = yaw
    rs.ego_info.rotation_pitch = float(nu_pitch)
    rs.ego_info.rotation_roll = float(nu_roll)

    # throttle / brake / steer / speed
    utime = int(sd["timestamp"])
    vm = _find_nearest_can_msg(vehicle_monitor_msgs, utime)

    if vm is not None:
        rs.ego_info.throttle = _normalize_01_like(vm.get("throttle", 0.0), name="throttle")
        rs.ego_info.brake = _normalize_01_like(vm.get("brake", 0.0), name="brake")

        raw_steer_deg = float(vm.get("steering", 0.0))
        rs.ego_info.steer = _normalize_steer(raw_steer_deg)

        speed_mps = _maybe_speed_to_mps(float(vm.get("vehicle_speed", 0.0)))
        rs.ego_info.speed = float(max(speed_mps, 0.0))

        # 用 speed + yaw 合成世界坐标系分量
        rs.ego_info.velocity_x = float(rs.ego_info.speed * math.cos(yaw))
        rs.ego_info.velocity_y = float(rs.ego_info.speed * math.sin(yaw))
        rs.ego_info.velocity_z = 0.0
    else:
        rs.ego_info.throttle = 0.0
        rs.ego_info.brake = 0.0
        rs.ego_info.steer = 0.0
        rs.ego_info.speed = 0.0
        rs.ego_info.velocity_x = 0.0
        rs.ego_info.velocity_y = 0.0
        rs.ego_info.velocity_z = 0.0

    # surrounding vehicles
    sv = SurroundingVehicles()
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not str(ann.get("category_name", "")).startswith("vehicle"):
            continue

        vx, vy, vz = ann["translation"]
        v_yaw, v_pitch, v_roll = _quat_to_yaw_pitch_roll(ann["rotation"])

        # 坐标转换
        vx2, vy2 = _to_carla_frame_xy(float(vx), float(vy))
        v_yaw2 = _to_carla_frame_yaw(float(v_yaw))

        rel_x = vx2 - x
        rel_y = vy2 - y
        rel_z = float(vz) - z
        dist = float(math.hypot(rel_x, rel_y))

        vel = nusc.box_velocity(ann_token)
        vel = np.nan_to_num(vel, nan=0.0).astype(np.float32)
        vvx, vvy, vvz = float(vel[0]), float(vel[1]), float(vel[2])
        if CONVERT_TO_CARLA_FRAME:
            vvy = -vvy

        speed = float(math.hypot(vvx, vvy))

        size_l, size_w, size_h = ann["size"]
        extent_x, extent_y, extent_z = float(size_l) / 2.0, float(size_w) / 2.0, float(size_h) / 2.0

        vi = VehicleInfo()
        vi.vehicle_id = int(hash(ann.get("instance_token", ann_token)) % 10_000_000)
        vi.location_x, vi.location_y, vi.location_z = vx2, vy2, float(vz)
        vi.rotation_yaw, vi.rotation_pitch, vi.rotation_roll = v_yaw2, float(v_pitch), float(v_roll)
        vi.velocity_x, vi.velocity_y, vi.velocity_z = vvx, vvy, vvz
        vi.speed = speed
        vi.distance = dist
        vi.relative_x, vi.relative_y, vi.relative_z = float(rel_x), float(rel_y), float(rel_z)
        vi.extent_x, vi.extent_y, vi.extent_z = extent_x, extent_y, extent_z

        sv.add_vehicle(vi)

    rs.surrounding_vehicles = sv

    # LaneInfo
    # 地图 API 需要 nuScenes 原坐标系 (nx, ny) 做查询
    lane_info = _build_lane_info(nusc_map, float(nx), float(ny), float(nu_yaw))
    if lane_info is not None and CONVERT_TO_CARLA_FRAME:
        lane_info.yaw_diff = -float(lane_info.yaw_diff)
    rs.lane_info = lane_info

    # GNSS / Collision / LaneInvasion / TrafficLight
    rs.gnss.latitude = float(nx)
    rs.gnss.longitude = float(ny)
    rs.gnss.altitude = float(nz)

    # 碰撞：真实离线数据无此事件，设置为不限制
    rs.collision.is_collision = False
    rs.collision.collision_type = CollisionType.NO_COLLISION
    rs.collision.impulse = 0.0

    # 车道侵入：离线数据通常无法可靠判断，默认合法
    rs.lane_invasion.is_invasion = False
    rs.lane_invasion.is_legal = True
    rs.lane_invasion.hold_frames = 0

    # 红绿灯：nuScenes v1.0 本身不提供动态信号灯状态，默认绿灯+远距离
    tf_max = float(getattr(config, "tf_max_dis", 1e9))
    rs.traffic_light = TrafficLightInfo(distance=tf_max, state=TrafficLightState.GREEN)

    return rs


# v1.0-mini测试
def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=config.nuscenes_root, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=config.can_root)

    avail_scenes = [s["name"] for s in nusc.scene]
    print("当前版本可用 scenes:", avail_scenes)

    scene_name = "scene-0916"  # 只能从上面的列表里选
    scene = _safe_scene_lookup(nusc, scene_name)

    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    nusc_map = NuScenesMap(dataroot=config.map_root, map_name=location)
    print(f"[scene] {scene_name} | location={location}")

    vehicle_monitor_msgs = nusc_can.get_messages(scene_name, "vehicle_monitor")
    vehicle_monitor_msgs = sorted(vehicle_monitor_msgs, key=lambda m: int(m.get("utime", 0)))

    print(f"找到 {len(vehicle_monitor_msgs)} 条 vehicle_monitor 数据")
    if vehicle_monitor_msgs:
        print(f"第一条 vehicle_monitor: {vehicle_monitor_msgs[0]}")

    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        rs = build_raw_state_from_sample(
            nusc=nusc,
            nusc_can=nusc_can,
            nusc_map=nusc_map,
            sample=sample,
            vehicle_monitor_msgs=vehicle_monitor_msgs,
        )
        rs.print_raw_state()
        token = sample["next"]


if __name__ == "__main__":
    main()
