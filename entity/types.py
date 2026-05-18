from config import config
from dataclasses import dataclass
from typing import List, Optional
from entity.enums import DirectionType, TrafficLightState, CollisionType


@dataclass
class EgoVehicleInfo:
    location_x: float = 0.0
    location_y: float = 0.0
    location_z: float = 0.0

    rotation_pitch: float = 0.0
    rotation_yaw: float = 0.0
    rotation_roll: float = 0.0

    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    speed: float = 0.0

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0

    extent_x: float = 0.0
    extent_y: float = 0.0
    extent_z: float = 0.0


@dataclass
class GnssData:
    sensor: Optional[object] = None
    latitude: float = 0.0  # 纬度(单位: 度)
    longitude: float = 0.0  # 经度(单位: 度)
    altitude: float = 0.0  # 高度(单位: 米)

    def copy_data(self) -> "GnssData":
        return GnssData(
            sensor=None,
            latitude=float(self.latitude),
            longitude=float(self.longitude),
            altitude=float(self.altitude)
        )


@dataclass
class ImuData:
    sensor: Optional[object] = None
    accelerometer_x: float = 0.0  # 沿 X 轴的加速度(单位: m/s²)
    accelerometer_y: float = 0.0
    accelerometer_z: float = 0.0
    gyroscope_x: float = 0.0  # 绕 X 轴的角速度(单位: rad/s)
    gyroscope_y: float = 0.0
    gyroscope_z: float = 0.0
    compass: float = 0.0  # 航向角(单位: 弧度, 0 表示正北)

    def copy_data(self) -> "ImuData":
        return ImuData(
            sensor=None,
            accelerometer_x=float(self.accelerometer_x),
            accelerometer_y=float(self.accelerometer_y),
            accelerometer_z=float(self.accelerometer_z),
            gyroscope_x=float(self.gyroscope_x),
            gyroscope_y=float(self.gyroscope_y),
            gyroscope_z=float(self.gyroscope_z),
            compass=float(self.compass),
        )


@dataclass
class CollisionData:
    sensor: Optional[object] = None
    is_collision: bool = False  # 是否发生碰撞
    collision_type: CollisionType = CollisionType.NO_COLLISION  # 碰撞类型
    impulse: float = 0.0  # 碰撞冲击力

    def copy_data(self) -> "CollisionData":
        return CollisionData(
            sensor=None,
            is_collision=bool(self.is_collision),
            collision_type=self.collision_type,
            impulse=float(self.impulse)
        )


@dataclass
class LaneInvasionData:
    sensor: Optional[object] = None
    is_invasion: bool = False  # 是否发生变道
    is_legal: bool = True  # 是否合法
    hold_frames: int = 0  # 还要保持多少帧（粘滞计数器）
    decay_len: int = 8  # 记忆窗口长度（可调）

    def copy_data(self) -> "LaneInvasionData":
        return LaneInvasionData(
            sensor=None,
            is_invasion=bool(self.is_invasion),
            is_legal=bool(self.is_legal),
            hold_frames=int(self.hold_frames),
            decay_len=int(self.decay_len),
        )


@dataclass
class VehicleInfo:
    # 标识
    vehicle_id: int = 0

    # 位置
    location_x: float = 0.0
    location_y: float = 0.0
    location_z: float = 0.0

    # 方向
    rotation_pitch: float = 0.0
    rotation_yaw: float = 0.0
    rotation_roll: float = 0.0

    # 速度
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    distance: float = config.surrounding_radius  # 相对距离
    speed: float = 0.0  # 速度

    relative_x: float = config.surrounding_radius
    relative_y: float = config.surrounding_radius
    relative_z: float = config.surrounding_radius

    # 包围盒
    extent_x: float = 0.0
    extent_y: float = 0.0
    extent_z: float = 0.0

    # 车道信息
    road_id: float = 0.0
    section_id: float = 0.0
    lane_id: float = 0.0
    lane_width: float = 0.0


class SurroundingVehicles:
    def __init__(self):
        self.vehicles: List[VehicleInfo] = []

    def add_vehicle(self, vehicle_info: VehicleInfo):
        self.vehicles.append(vehicle_info)

    def get_closest(self, n=1) -> List[VehicleInfo]:
        return sorted(self.vehicles, key=lambda v: v.distance)[:n]


@dataclass
class TrafficLightInfo:
    distance: float  # 红绿灯的距离
    state: TrafficLightState  # 0 unknown  1 red  2 yellow  3 green

    @staticmethod
    def no_light():
        return TrafficLightInfo(distance=config.tf_max_dis, state=TrafficLightState.UNKNOWN)


class LaneInfo:
    __slots__ = (
        "lane_id",
        "has_next",
        "is_junction",
        "yaw_diff",
        "lane_change",
        "speed_limit",
    )

    def __init__(self):
        self.lane_id: float = 0  # 车道id
        self.has_next: bool = False  # 前方一定距离还有没有路
        self.is_junction: bool = False  # 是否在路口
        self.yaw_diff: float = 0.0  # 道路夹角
        self.lane_change: bool = False  # 是否支持变道
        self.speed_limit = 999.0  # 车道限速

    # 道路状况的推荐方向, 实际操作还需智能体结合红绿灯状态等做决策, 大转向还是小转向由智能体根据yaw_diff自行判断
    def turning_direction(self):
        if not self.has_next:
            return DirectionType.STOP  # 没路了
        if abs(self.yaw_diff) < 5:  # 可选加上直行判断
            return DirectionType.STRAIGHT
        return DirectionType.LEFT if self.yaw_diff > 0 else DirectionType.RIGHT
