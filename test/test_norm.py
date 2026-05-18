from dataclasses import dataclass

import torch

from entity.enums import TrafficLightState, CollisionType
from entity.types import (
    EgoVehicleInfo,
    GnssData,
    ImuData,
    CollisionData,
    LaneInvasionData,
    VehicleInfo,
    SurroundingVehicles,
    TrafficLightInfo,
    LaneInfo,
)
from environment.state import ExtractedState
from util.normalize import clip01, clip11, R

RAW_STATE_DICT = {
    "ego_info": {
        "location_x": -67.25457000732422,
        "location_y": 27.963756561279297,
        "location_z": 0.5074906349182129,
        "rotation_pitch": 0.0,
        "rotation_yaw": 0.15919800102710724,
        "rotation_roll": 0.0,
        "velocity_x": -1.0304680760853743e-36,
        "velocity_y": -2.8631932781424322e-39,
        "velocity_z": -1.305741548538208,
        "speed": 1.305741548538208,
        "throttle": 0.8500000238418579,
        "steer": 0.015735767781734467,
        "brake": 0.0,
        "extent_x": 2.4459850788116455,
        "extent_y": 0.9178230166435242,
        "extent_z": 0.7620594501495361,
    },
    "gnss": {
        "latitude": -0.0002512276634547561,
        "longitude": -0.0005951749288034617,
        "altitude": 2.5469372272491455,
    },
    "imu": {
        "accelerometer_x": -1.1668535633380784e-35,
        "accelerometer_y": 4.96361637311086e-36,
        "accelerometer_z": 9.8100004196167,
        "gyroscope_x": 0.0,
        "gyroscope_y": 0.0,
        "gyroscope_z": 0.0,
        "compass": 1.5735749006271362,
    },
    "collision": {
        "collision_type": "NO_COLLISION",
        "impulse": 0.0,
    },
    "lane_invasion": {
        "is_invasion": False,
        "is_legal": True,
    },
    "surrounding_vehicles": [
        {
            "vehicle_id": 34,
            "location_x": -76.66610717773438,
            "location_y": 24.471010208129883,
            "location_z": 0.5074906349182129,
            "velocity_x": 5.861345050864602e-38,
            "velocity_y": 1.6286030882229456e-40,
            "velocity_z": -1.305741548538208,
            "distance": 4.974375247955322,
            "speed": 1.305741548538208,
            "relative_x": -9.411537170410156,
            "relative_y": -3.492746353149414,
            "relative_z": 0.0,
            "extent_x": 2.6183807849884033,
            "extent_y": 0.9622216820716858,
            "extent_z": 0.8219860196113586,
        }
    ],
    "traffic_light": {
        "state": "RED",
        "distance": 9.188175201416016,
    },
    "lane_info": {
        "lane_id": -2,
        "has_next": True,
        "is_junction": False,
        "yaw_diff": 0.0,
        "lane_change": 2,
        "speed_limit": 30.0,
    },
}


@dataclass
class RawState:
    ego_info: EgoVehicleInfo
    gnss: GnssData
    imu: ImuData
    collision: CollisionData
    lane_invasion: LaneInvasionData
    surrounding_vehicles: SurroundingVehicles
    traffic_light: TrafficLightInfo
    lane_info: LaneInfo
    _offroad_flag: bool = False  # norm_flags 用到这个字段


def build_raw_state_from_dict(data: dict) -> RawState:
    # --- ego_info 直接用 dataclass 解包 ---
    ego = EgoVehicleInfo(**data["ego_info"])

    gnss_data = data["gnss"]
    gnss = GnssData()
    gnss.latitude = gnss_data["latitude"]
    gnss.longitude = gnss_data["longitude"]
    gnss.altitude = gnss_data["altitude"]

    imu_data = data["imu"]
    imu = ImuData()
    imu.accelerometer_x = imu_data["accelerometer_x"]
    imu.accelerometer_y = imu_data["accelerometer_y"]
    imu.accelerometer_z = imu_data["accelerometer_z"]
    imu.gyroscope_x = imu_data["gyroscope_x"]
    imu.gyroscope_y = imu_data["gyroscope_y"]
    imu.gyroscope_z = imu_data["gyroscope_z"]
    imu.compass = imu_data["compass"]

    # --- CollisionData ---
    col_data = data["collision"]
    collision = CollisionData()
    # "NO_COLLISION" -> CollisionType.NO_COLLISION
    collision.collision_type = getattr(CollisionType, col_data["collision_type"])
    collision.impulse = col_data["impulse"]
    collision.is_collision = (collision.collision_type != CollisionType.NO_COLLISION)

    # --- LaneInvasionData ---
    li_data = data["lane_invasion"]
    lane_invasion = LaneInvasionData()
    lane_invasion.is_invasion = li_data["is_invasion"]
    lane_invasion.is_legal = li_data["is_legal"]

    # --- SurroundingVehicles & VehicleInfo ---
    sv = SurroundingVehicles()
    for v in data.get("surrounding_vehicles", []):
        veh = VehicleInfo(
            vehicle_id=v["vehicle_id"],
            location_x=v["location_x"],
            location_y=v["location_y"],
            location_z=v["location_z"],
            velocity_x=v["velocity_x"],
            velocity_y=v["velocity_y"],
            velocity_z=v["velocity_z"],
            distance=v["distance"],
            speed=v["speed"],
            relative_x=v["relative_x"],
            relative_y=v["relative_y"],
            relative_z=v["relative_z"],
            extent_x=v["extent_x"],
            extent_y=v["extent_y"],
            extent_z=v["extent_z"],
            # lane_width / road_id / section_id / lane_id 用默认值
        )
        sv.add_vehicle(veh)

    # --- TrafficLightInfo ---
    tl_raw = data["traffic_light"]
    tl_state = getattr(TrafficLightState, tl_raw["state"])  # "RED" -> TrafficLightState.RED
    traffic_light = TrafficLightInfo(distance=tl_raw["distance"], state=tl_state)

    # --- LaneInfo ---
    lane_raw = data["lane_info"]
    lane_info = LaneInfo()
    lane_info.lane_id = lane_raw["lane_id"]
    lane_info.has_next = lane_raw["has_next"]
    lane_info.is_junction = lane_raw["is_junction"]
    lane_info.yaw_diff = lane_raw["yaw_diff"]
    lane_info.lane_change = bool(lane_raw["lane_change"])
    lane_info.speed_limit = lane_raw["speed_limit"]

    return RawState(
        ego_info=ego,
        gnss=gnss,
        imu=imu,
        collision=collision,
        lane_invasion=lane_invasion,
        surrounding_vehicles=sv,
        traffic_light=traffic_light,
        lane_info=lane_info,
    )


def test_clip_functions_support_scalar_and_tensor():
    # 标量
    s = clip01(100.0, 0.0, R.dist_hi)
    assert 0.0 <= s <= 1.0

    # Tensor
    t = torch.tensor([-10.0, 0.0, 40.0, 100.0])
    nt = clip01(t, 0.0, R.dist_hi)
    assert torch.all(nt >= 0.0) and torch.all(nt <= 1.0)

    u = clip11(torch.tensor([-200.0, 0.0, 200.0]), R.pos_lo, R.pos_hi)
    assert torch.all(u >= -1.0) and torch.all(u <= 1.0)


def test_extracted_state_normalization_shapes_and_ranges():
    raw_state = build_raw_state_from_dict(RAW_STATE_DICT)
    device = torch.device("cpu")

    state = ExtractedState.from_raw(
        raw_state,
        device=device,
        normalize=True,
    )

    # ---- shape 检查 ----
    assert state.ego_vector.shape == (16,)
    assert state.gnss_vector.shape == (3,)
    assert state.tl_vector.shape == (5,)
    assert state.lane_vector.shape == (9,)

    assert state.nearest_vector is not None
    assert state.nearest_vector.shape == (21,)

    assert state.vehicles_tensor is not None
    assert state.vehicles_tensor.shape[0] == 1
    assert state.vehicles_tensor.shape[1] == 21

    # ---- 范围检查：只挑关键几块 ----
    v = state.ego_vector

    # 位置 [-1,1]
    assert torch.all(v[0:2] >= -1.0) and torch.all(v[0:2] <= 1.0)
    # 角度 [-1,1]
    assert torch.all(v[3:6] >= -1.0) and torch.all(v[3:6] <= 1.0)
    # 速度 [-1,1]
    assert torch.all(v[6:9] >= -1.0) and torch.all(v[6:9] <= 1.0)
    # speed [0,1]
    assert 0.0 <= float(v[9]) <= 1.0
    # throttle / steer / brake 合理范围
    assert 0.0 <= float(v[10]) <= 1.0
    assert -1.0 <= float(v[11]) <= 1.0
    assert 0.0 <= float(v[12]) <= 1.0

    # 交通灯：distance [0,1]，one-hot 在 [0,1]
    tv = state.tl_vector
    assert 0.0 <= float(tv[0]) <= 1.0
    assert torch.all(tv[1:] >= 0.0) and torch.all(tv[1:] <= 1.0)

    # 车道：has_next / is_junction / lane_change / turning one-hot ∈[0,1]
    lv = state.lane_vector
    assert torch.all(lv[0:2] >= 0.0) and torch.all(lv[0:2] <= 1.0)
    assert -1.0 <= float(lv[2]) <= 1.0  # yaw_diff_norm
    assert 0.0 <= float(lv[3]) <= 1.0  # lane_change
    assert 0.0 <= float(lv[4]) <= 1.0  # speed_limit_norm
    assert torch.all(lv[5:] >= 0.0) and torch.all(lv[5:] <= 1.0)

    # 最近车辆 + 周围车辆：简单检查整体落在 [-1.1,1.1] 防止浮点误差
    nv = state.nearest_vector
    assert torch.all(nv >= -1.1) and torch.all(nv <= 1.1)

    vt = state.vehicles_tensor
    assert torch.all(vt >= -1.1) and torch.all(vt <= 1.1)
