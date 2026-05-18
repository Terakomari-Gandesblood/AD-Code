import copy
from dataclasses import dataclass
from typing import Optional

import torch

from entity.types import EgoVehicleInfo, GnssData, VehicleInfo, TrafficLightInfo, LaneInfo, CollisionData, \
    LaneInvasionData, SurroundingVehicles
from util.feature_extraction import ego_extractor, tl_extractor, gnss_extractor, lane_extractor, \
    vehicle_extractor
from util.normalize import clip11, clip01, R


class RawState:
    def __init__(self):
        self.ego_info: EgoVehicleInfo = EgoVehicleInfo()
        self.gnss: GnssData = GnssData()
        # self.imu: ImuData = ImuData()  # 容易引起carla崩溃暂时注释掉后期再优化
        self.collision: CollisionData = CollisionData()
        self.lane_invasion: LaneInvasionData = LaneInvasionData()
        self.surrounding_vehicles: Optional[SurroundingVehicles] = None
        self.traffic_light: TrafficLightInfo = TrafficLightInfo.no_light()
        self.lane_info: Optional[LaneInfo] = None

    def copy_data(self) -> "RawState":
        out = RawState()

        out.gnss = self.gnss.copy_data()
        out.collision = self.collision.copy_data()
        out.lane_invasion = self.lane_invasion.copy_data()

        out.ego_info = copy.deepcopy(self.ego_info)
        out.traffic_light = copy.deepcopy(self.traffic_light)
        out.lane_info = copy.deepcopy(self.lane_info)
        out.surrounding_vehicles = copy.deepcopy(self.surrounding_vehicles)

        return out

    def print_raw_state(self):
        def fmt(val, fmt_str="{:.1f}"):
            """数值格式化，None 安全"""
            if val is None:
                return "None"
            try:
                return fmt_str.format(val)
            except (TypeError, ValueError):
                return str(val)

        def print_none(name):
            print(f"  {name} is None")

        print("------ RawState ------")

        # ===== Ego =====
        ego = self.ego_info
        if ego is None:
            print_none("ego_info")
        else:
            print(
                f"  ego: "
                f"loc_x={fmt(ego.location_x)} "
                f"pitch={fmt(ego.rotation_pitch)} "
                f"vel_x={fmt(ego.velocity_x)} "
                f"speed={fmt(ego.speed)} "
                f"throttle={fmt(ego.throttle)} "
                f"steer={fmt(ego.steer)} "
                f"brake={fmt(ego.brake)} "
                f"extent_x={fmt(ego.extent_x)}"
            )

        # ===== GNSS =====
        if self.gnss is None:
            print_none("gnss")
        else:
            print(
                f"  gnss: "
                f"lat={fmt(self.gnss.latitude, '{:.6f}')} "
                f"lon={fmt(self.gnss.longitude, '{:.6f}')}"
            )

        # ===== Collision =====
        if self.collision is None:
            print_none("collision")
        else:
            collision_type = (
                self.collision.collision_type.name
                if self.collision.collision_type is not None
                else "Unknown"
            )
            print(
                f"  collision: {collision_type} "
                f"(force={fmt(self.collision.impulse)})"
            )

        # ===== Lane Invasion =====
        if self.lane_invasion is None:
            print_none("lane_invasion")
        else:
            status = "INVASION" if self.lane_invasion.is_invasion else "OK"
            legality = "Legal" if self.lane_invasion.is_legal else "Illegal"
            print(f"  lane_invasion: {status} ({legality})")

        # ===== Surrounding Vehicles =====
        sv = self.surrounding_vehicles
        if sv is None or not sv.vehicles:
            print("  surrounding_vehicles: None")
        else:
            closest = sv.get_closest(1)
            closest = closest[0] if closest else None
            if closest is None:
                print(f"  surrounding_vehicles: {len(sv.vehicles)} vehicles | Closest: None")
            else:
                print(
                    f"  surrounding_vehicles: {len(sv.vehicles)} vehicles | "
                    f"Closest: {fmt(closest.distance)}m"
                )

        # ===== Traffic Light =====
        if self.traffic_light is None:
            print_none("traffic_light")
        else:
            state = (
                self.traffic_light.state.name
                if self.traffic_light.state is not None
                else "Unknown"
            )
            print(
                f"  traffic_light: {state} "
                f"@ {fmt(self.traffic_light.distance)}m"
            )

        # ===== Lane Info =====
        if self.lane_info is None:
            print_none("lane_info")
        else:
            turn = (
                self.lane_info.turning_direction().name
                if self.lane_info.turning_direction() is not None
                else "Unknown"
            )
            print(
                f"  lane_info: "
                f"id={self.lane_info.lane_id} "
                f"turn={turn} "
                f"limit={fmt(self.lane_info.speed_limit, '{:.0f}')}km/h"
            )


@dataclass
class ExtractedState:
    ego_vector: torch.Tensor  # 自车特征向量 [1 * 16]
    gnss_vector: torch.Tensor  # 全球导航卫星系统数据特征 [1 * 3]
    # imu_vector: torch.Tensor  # 惯性测量单元特征 [1 * 7]
    tl_vector: torch.Tensor  # 交通信号灯特征 [1 * 5]
    lane_vector: torch.Tensor  # 车道信息特征 [1 * 9]
    nearest_vector: torch.Tensor  # 最近邻目标特征 [1 * 21]
    vehicles_tensor: torch.Tensor  # 周围所有车辆特征矩阵 [vehicles_num * 21]

    @classmethod
    def from_raw(cls, raw_state, device: torch.device, normalize: bool = True) -> "ExtractedState":
        # 单体信息提取
        ego_vector = ego_extractor(raw_state.ego_info, device).reshape(-1)
        gnss_vector = gnss_extractor(raw_state.gnss, device).reshape(-1)
        # imu_vector = imu_extractor(raw_state.imu, device).reshape(-1)
        tl_vector = tl_extractor(raw_state.traffic_light, device).reshape(-1)
        lane_vector = lane_extractor(raw_state.lane_info, device).reshape(-1)

        # 最近车辆
        sv = getattr(raw_state, "surrounding_vehicles", None)
        if sv and getattr(sv, "vehicles", None):
            nearest = sv.get_closest()[0]
        else:
            nearest = VehicleInfo()
        nearest_vector = vehicle_extractor(nearest, device).reshape(-1)

        # 周围车辆
        vehicles_src = sv.vehicles if (sv and sv.vehicles) else []
        if len(vehicles_src) == 0:
            vehicles_tensor = None
        else:
            encoded_list = [vehicle_extractor(v, device).reshape(-1)
                            for v in vehicles_src]
            vehicles_tensor = torch.stack(encoded_list, dim=0).to(device)  # [N, D]

        state = cls(
            ego_vector=ego_vector,
            gnss_vector=gnss_vector,
            # imu_vector=imu_vector,
            tl_vector=tl_vector,
            lane_vector=lane_vector,
            nearest_vector=nearest_vector,
            vehicles_tensor=vehicles_tensor,
        )

        if normalize:
            state.normalize_()

        return state

    @staticmethod
    def _norm_vehicle_vec(v: torch.Tensor) -> torch.Tensor:
        v = v.clone()

        # 位置
        v[0:2] = clip11(v[0:2], R.pos_lo, R.pos_hi)
        v[2] = clip11(v[2], -5.0, 10.0)

        # 欧拉角
        v[3] = clip11(v[3], -45.0, 45.0)  # pitch
        v[4] = clip11(v[4], -180.0, 180.0)  # yaw
        v[5] = clip11(v[5], -45.0, 45.0)  # roll

        # 速度
        v[6:9] = clip11(v[6:9], R.vel_lo, R.vel_hi)  # vx,vy,vz
        v[9] = clip01(v[9], 0.0, R.dist_hi)  # distance
        v[10] = clip01(v[10], 0.0, R.speed_hi)  # speed

        # 相对位置
        v[11] = clip11(v[11], -80.0, 80.0)  # rel_x
        v[12] = clip11(v[12], -10.0, 10.0)  # rel_y
        v[13] = clip11(v[13], -5.0, 5.0)  # rel_z

        # 车体尺寸
        v[14:17] = clip01(v[14:17], 0.0, R.size_hi)

        # road / section / lane id
        v[17] = clip11(v[17], -2000.0, 2000.0)
        v[18] = clip11(v[18], -2000.0, 2000.0)
        v[19] = clip11(v[19], -10.0, 10.0)

        # 车道宽度
        v[20] = clip11(v[20], R.lane_w_lo, R.lane_w_hi)

        return v

    @classmethod
    def _norm_vehicle_batch(cls, vs: torch.Tensor) -> torch.Tensor:
        out = [cls._norm_vehicle_vec(vs[i]) for i in range(vs.shape[0])]
        return torch.stack(out, dim=0)

    def normalize_(self) -> "ExtractedState":
        # ego
        v = self.ego_vector

        # 位置
        v[0:2] = clip11(v[0:2], R.pos_lo, R.pos_hi)
        v[2] = clip11(v[2], -5.0, 10.0)

        # 角度
        v[3] = clip11(v[3], -45.0, 45.0)  # pitch
        v[4] = clip11(v[4], -180.0, 180.0)  # yaw
        v[5] = clip11(v[5], -45.0, 45.0)  # roll

        # 速度向量和标量速度
        v[6:9] = clip11(v[6:9], R.vel_lo, R.vel_hi)
        v[9] = clip01(v[9], 0.0, R.speed_hi)

        # 控制量只做 clamp
        v[10] = v[10].clamp(0.0, 1.0)  # throttle
        v[11] = v[11].clamp(-1.0, 1.0)  # steer
        v[12] = v[12].clamp(0.0, 1.0)  # brake

        # 车体尺寸
        v[13:16] = clip01(v[13:16], 0.0, R.size_hi)

        # tl_vector:
        tv = self.tl_vector
        tv[0] = clip01(tv[0], 0.0, R.dist_hi)
        tv[1:] = tv[1:].clamp(0.0, 1.0)

        # lane_vector
        lv = self.lane_vector
        lv[0:2] = lv[0:2].clamp(0.0, 1.0)  # flags
        lv[2] = clip11(lv[2], -60.0, 60.0)  # yaw_diff
        lv[3] = lv[3].clamp(0.0, 1.0)  # lane_change flag

        # 限速
        speed_limit_ms = (lv[4] / 3.6).clamp(0.0, R.speed_hi)
        lv[4] = speed_limit_ms / R.speed_hi
        lv[5:] = lv[5:].clamp(0.0, 1.0)  # turning one-hot

        # 最近车辆和周围车辆
        if self.nearest_vector is not None:
            self.nearest_vector = self._norm_vehicle_vec(self.nearest_vector)

        if self.vehicles_tensor is not None:
            self.vehicles_tensor = self._norm_vehicle_batch(self.vehicles_tensor)

        # gnss_vector
        gv = self.gnss_vector
        gv[2] = clip11(gv[2], -100.0, 500.0)  # altitude

        return self

    def __repr__(self):
        return (
            f"EncodedState(\n"
            f"  ego_vector: {tuple(self.ego_vector.shape)}, "
            f"mean={self.ego_vector.mean():.3f}, std={self.ego_vector.std():.3f}\n"
            f"  gnss_vector: {tuple(self.gnss_vector.shape)}, "
            f"mean={self.gnss_vector.mean():.3f}\n"
            # f"  imu_vector: {tuple(self.imu_vector.shape)}, "
            # f"mean={self.imu_vector.mean():.3f}\n"
            f"  tl_vector: {tuple(self.tl_vector.shape)}, "
            f"mean={self.tl_vector.mean():.3f}\n"
            f"  lane_vector: {tuple(self.lane_vector.shape)}, "
            f"mean={self.lane_vector.mean():.3f}\n"
            f"  nearest_vector: {tuple(self.nearest_vector.shape)}, "
            f"mean={self.nearest_vector.mean():.3f}\n"
            f"  vehicles_tensor: {tuple(self.vehicles_tensor.shape)}"
        )
