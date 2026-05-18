import warnings

import torch

from entity.enums import TrafficLightState, DirectionType
from entity.types import EgoVehicleInfo, GnssData, ImuData, VehicleInfo, TrafficLightInfo, LaneInfo


def _warn(name: str, msg: str) -> None:
    warnings.warn(f"[extractor] {name}: {msg}", RuntimeWarning, stacklevel=2)


def ego_extractor(ego_info: EgoVehicleInfo, device: torch.device = 'cpu') -> torch.Tensor:
    if ego_info is None:
        _warn("ego_extractor", "ego_info is None, use all-zero defaults.")
        ego_info = EgoVehicleInfo()
    return torch.tensor([
        ego_info.location_x,
        ego_info.location_y,
        ego_info.location_z,
        ego_info.rotation_pitch,
        ego_info.rotation_yaw,
        ego_info.rotation_roll,
        ego_info.velocity_x,
        ego_info.velocity_y,
        ego_info.velocity_z,
        ego_info.speed,
        ego_info.throttle,
        ego_info.steer,
        ego_info.brake,
        ego_info.extent_x,
        ego_info.extent_y,
        ego_info.extent_z
    ], dtype=torch.float32, device=device)  # [1 * 16]


def gnss_extractor(gnss: GnssData, device: torch.device = 'cpu') -> torch.Tensor:
    if gnss is None:
        _warn("gnss_extractor", "gnss is None, use all-zero defaults.")
        gnss = GnssData(sensor=None)

    return torch.tensor([
        gnss.latitude,
        gnss.longitude,
        gnss.altitude
    ], dtype=torch.float32, device=device)  # [1 * 3]


def imu_extractor(imu: ImuData, device: torch.device = 'cpu') -> torch.Tensor:
    if imu is None:
        _warn("imu_extractor", "imu is None, use all-zero defaults.")
        imu = ImuData(sensor=None)

    return torch.tensor([
        imu.accelerometer_x,
        imu.accelerometer_y,
        imu.accelerometer_z,
        imu.gyroscope_x,
        imu.gyroscope_y,
        imu.gyroscope_z,
        imu.compass
    ], dtype=torch.float32, device=device)  # [1 * 7]


def vehicle_extractor(npc_vehicle: VehicleInfo, device: torch.device = 'cpu') -> torch.Tensor:
    if npc_vehicle is None:
        _warn("vehicle_extractor", "npc_vehicle is None, use defaults.")
        npc_vehicle = VehicleInfo()

    return torch.tensor([
        npc_vehicle.location_x,
        npc_vehicle.location_y,
        npc_vehicle.location_z,
        npc_vehicle.rotation_pitch,
        npc_vehicle.rotation_yaw,
        npc_vehicle.rotation_roll,
        npc_vehicle.velocity_x,
        npc_vehicle.velocity_y,
        npc_vehicle.velocity_z,
        npc_vehicle.distance,
        npc_vehicle.speed,
        npc_vehicle.relative_x,
        npc_vehicle.relative_y,
        npc_vehicle.relative_z,
        npc_vehicle.extent_x,
        npc_vehicle.extent_y,
        npc_vehicle.extent_z,
        npc_vehicle.road_id,
        npc_vehicle.section_id,
        npc_vehicle.lane_id,
        npc_vehicle.lane_width
    ], dtype=torch.float32, device=device)  # [1 * 21]


def tl_extractor(tl: TrafficLightInfo, device: torch.device = 'cpu') -> torch.Tensor:
    if tl is None:
        _warn("tl_extractor", "tl is None, use no_light().")
        tl = TrafficLightInfo.no_light()

    return torch.tensor([
        tl.distance,
        1 if tl.state == TrafficLightState.UNKNOWN else 0,
        1 if tl.state == TrafficLightState.RED else 0,
        1 if tl.state == TrafficLightState.YELLOW else 0,
        1 if tl.state == TrafficLightState.GREEN else 0,
    ], dtype=torch.float32, device=device)  # [1 * 5]


def lane_extractor(lane: LaneInfo, device: torch.device = 'cpu') -> torch.Tensor:
    if lane is None:
        _warn("lane_extractor", "lane is None, use default LaneInfo().")
        lane = LaneInfo()

    direction = lane.turning_direction()
    return torch.tensor([
        1 if lane.has_next else 0,
        1 if lane.is_junction else 0,
        lane.yaw_diff,
        1 if lane.lane_change else 0,
        lane.speed_limit,
        1 if direction == DirectionType.STOP else 0,
        1 if direction == DirectionType.STRAIGHT else 0,
        1 if direction == DirectionType.LEFT else 0,
        1 if direction == DirectionType.RIGHT else 0,
    ], dtype=torch.float32, device=device)  # [1 * 9]
