import gym

from perturbation.attack import attack
from util.carla_utils import *
from typing import Dict, Optional


def _shape_speed_gate(v_mps: float):
    """
    返回 (gate_bool, scale_float)
    - gate_bool: 是否认为“正在真实行驶”
    - scale_float: 用速度缩放正奖，速度越慢缩放越小
    """
    gate = v_mps > 1.0  # >1m/s 视为在行驶
    scale = min(1.0, v_mps / 6.0)  # 0~6m/s 线性缩放到 0~1
    return gate, scale


class CarlaEnv(gym.Env):
    def __init__(self, no_rendering_mode=False, npc_vehicle_num=10):
        # 连接服务
        self.client, self.world = get_client()

        # 设置渲染模式
        set_render_mode(self.world, no_rendering_mode=no_rendering_mode, synchronous_mode=True)

        self.ego = None
        self.raw_state = RawState()
        self.raw_obs = RawState()  # 当前汽车观测到的环境
        self.callback_id = None
        self.npc_vehicle_num = npc_vehicle_num

        self.prev_action = (0.0, 0.0, 0.0)
        self.step_count = 0
        self.stuck_counter = 0
        self.max_steps = config.env_max_step

        self.stuck_patience = getattr(config, "stuck_patience", 80)

        self._last_heading_diff_deg = 0.0

        # === 奖励权重相关 ===
        self.w_v = getattr(config, "w_v", 0.5)  # 【速度跟随权重】
        # 控制车辆速度是否接近期望速度的重要程度
        # 越大 → 更强调限速与红灯减速等规则驾驶
        # 越小 → 允许更自由/激进的加速与探索

        self.w_lane = getattr(config, "w_lane", 5.0)  # 【车道保持权重】
        # 车道侵入惩罚的权重，越大越重视不越线
        # 建议 >1，防止模型在训练初期频繁压线

        self.w_tlv = getattr(config, "w_tlv", 10.0)  # 【闯红灯惩罚权重】
        # 对红灯下继续前进的严重惩罚
        # 值过低会让模型忽略红灯

        self.w_tls = getattr(config, "w_tls", 2.0)  # 【红灯停车奖励权重】
        # 鼓励在红灯前准确停下
        # 与 w_tlv 配合使用形成“停下奖励 + 闯灯惩罚”对称机制

        self.w_safety = getattr(config, "w_safety", 1.0)  # 【跟车安全权重】
        # 过近跟车惩罚的强度
        # 提高此值可让模型更稳健地保持安全距离

        self.w_smooth = getattr(config, "w_smooth", 0.05)  # 【控制平滑性权重】
        # 对转向、油门、刹车变化幅度的惩罚
        # 值越大，行为越平稳但反应更慢

        self.w_alive = getattr(config, "w_alive", 0.01)  # 【存活步惩罚权重】
        # 每步固定小负值，避免智能体无意义地拖时间；
        # 一般保持很小（0.001~0.05）。

        self.w_collision = getattr(config, "w_collision", 30.0)  # 【碰撞惩罚权重】
        # 碰撞的极大负反馈
        # 通常是最大权重项，代表“安全高于一切”

        # === 环境物理/规则参数 ===
        self.speed_limit_ratio = getattr(config, "speed_limit_ratio", 0.9)
        # 车道限速折扣比例
        # 保留一定安全余量

        self.red_dist_trigger = getattr(config, "red_dist_trigger", 30.0)
        # 距离红灯触发减速的距离阈值（米）
        # 小于此距离时开始规划刹车

        self.red_stop_margin = getattr(config, "red_stop_margin", 3.5)
        # 红灯停车预留距离（米），防止车辆压线或越界停车

        self.k_tl = getattr(config, "k_tl", 1.0)  # 【红灯线性制动系数】
        # 控制红灯减速曲线的斜率
        # 值大 → 更快减速，值小 → 更平缓

        self.safe_dist = getattr(config, "safe_dist", 8.0)
        # 跟车安全距离（米）
        # 小于此距离会触发安全惩罚

        self.k_gap = getattr(config, "k_gap", 0.8)
        # 跟车速度调节系数
        # 决定允许的跟车速度随距离的线性增长程度

        self.follow_delta = getattr(config, "follow_delta", 1.0)
        # 跟车速度上限的松弛量
        # 允许 Ego 车比前车稍快一点（米/秒）

        # === 方向对齐相关权重/阈值 ===
        self.w_heading = getattr(config, "w_heading", 2.0)
        # 【方向对齐惩罚权重】
        # 在非路口时若车辆朝向与车道方向差异过大，将给予负反馈
        # 过大会导致车辆频繁调整方向，过小则无法矫正偏航

        self.heading_tol_deg = getattr(config, "heading_tol_deg", 12.0)
        # 【方向容差角（度）】
        # 车辆与道路方向在该角度以内视为对齐
        # 小于此角度不惩罚

        self.heading_max_deg = getattr(config, "heading_max_deg", 90.0)
        # 【最大惩罚角（度）】
        # 当角度差 ≥ 该值时按满惩罚 −1 处理
        # 例如逆向行驶时即达到最大惩罚

        self.w_center = getattr(config, "w_center", 2.0)  # 中心线偏移惩罚权重
        self.center_tol_m = getattr(config, "center_tol_m", 0.3)  # 容差
        self.center_max_m = getattr(config, "center_max_m", 1.5)  # 达到满惩罚的偏移

        self.w_offroad = getattr(config, "w_offroad", 5.0)  # 离开可行驶车道的持续惩罚权重
        self.offroad_patience = getattr(config, "offroad_patience", 40)  # 连续 off-road 多少步后结束
        self.offroad_counter = 0  # 计数器

    # === NEW: 角度工具 ===
    @staticmethod
    def _to_deg(angle):
        """传入弧度或度，返回度"""
        if abs(angle) <= 2 * math.pi:
            return math.degrees(angle)
        return angle

    @staticmethod
    def _angle_diff_deg(a_deg, b_deg):
        """返回两方向的最小夹角（度）∈ [0, 180]"""
        d = abs(a_deg - b_deg) % 360.0
        return d if d <= 180.0 else 360.0 - d

    @staticmethod
    def _kmh_to_ms(kmh: float) -> float:
        return kmh / 3.6

    def _nearest_vehicle(self) -> Optional[VehicleInfo]:
        sv = self.raw_state.surrounding_vehicles
        if sv and sv.vehicles:
            return sv.get_closest(1)[0]
        return None

    def _ego_forward_xy(self):
        """基于 ego yaw 得到 2D 前向单位向量"""
        yaw_deg = float(self.raw_state.ego_info.rotation_yaw)
        yaw = math.radians(yaw_deg) if abs(yaw_deg) > 2 * math.pi else yaw_deg
        return math.cos(yaw), math.sin(yaw)

    def _is_ahead(self, vx: float, vy: float, half_fov_deg: float = 45.0) -> bool:
        """判断相对位移(vx,vy)是否在 Ego 前方扇区内"""
        fx, fy = self._ego_forward_xy()
        rel_norm = math.hypot(vx, vy)
        if rel_norm < 1e-6:
            return True
        cosang = (fx * vx + fy * vy) / rel_norm
        return cosang >= math.cos(math.radians(half_fov_deg))

    def _nearest_vehicle_ahead(self, lateral_tol_m: float = 3.5, allow_adjacent: bool = False) -> Optional[VehicleInfo]:
        """取前方扇区内的最近车辆"""
        sv = self.raw_state.surrounding_vehicles
        li = self.raw_state.lane_info
        if not (sv and sv.vehicles and li):
            return None

        ego_lane_id = getattr(li, "lane_id", None)
        ego_wp = self.world.get_map().get_waypoint(self.ego.get_location(), project_to_road=True)
        ego_road_id = getattr(ego_wp, "road_id", None)
        ego_section_id = getattr(ego_wp, "section_id", None)

        cand = []
        for v in sv.vehicles:
            if not self._is_ahead(v.relative_x, v.relative_y):
                continue

            if v.lane_id is not None and ego_lane_id is not None:
                lane_ok = (v.lane_id == ego_lane_id) or (allow_adjacent and abs(v.lane_id - ego_lane_id) == 1)
                road_ok = (v.road_id is None or ego_road_id is None) or (v.road_id == ego_road_id)
                sec_ok = (v.section_id is None or ego_section_id is None) or (v.section_id == ego_section_id)
                same_lane_ok = lane_ok and road_ok and sec_ok
            else:
                same_lane_ok = abs(v.relative_y) <= lateral_tol_m

            if same_lane_ok:
                cand.append(v)

        if not cand:
            return None
        cand.sort(key=lambda x: x.distance)
        return cand[0]

    def _lane_heading_deg(self) -> Optional[float]:
        """
        返回当前所在 driving lane 的航向角
        """
        carla_map = self.world.get_map()
        wp = carla_map.get_waypoint(self.ego.get_location(),
                                    project_to_road=False,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        return float(wp.transform.rotation.yaw)

    def _center_align_reward(self, off: Optional[float]) -> float:
        """
        在中心 ≈ +1；偏离到容差边界降到 0；超出容差线性到 -1。
        非路口才计算；无法计算或在路口返回 0。
        """
        if off is None:
            return 0.0
        li = getattr(self.raw_state, "lane_info", None)
        if not li or li.is_junction:
            return 0.0

        tol = float(self.center_tol_m)  # 容差
        mx = float(self.center_max_m)  # 达到满罚的偏移
        if off <= tol:
            # 正奖
            return 1.0 - (off / max(1e-6, tol))
        # 负奖
        span = max(1e-6, mx - tol)
        return -min(1.0, (off - tol) / span)

    def _heading_align_reward(self,
                              ego_yaw_deg: Optional[float],
                              lane_yaw_deg: Optional[float]) -> float:
        """
        非路口时计算；小角度内给正奖，超过容差后线性到 -1。
        """
        li = getattr(self.raw_state, "lane_info", None)
        if li is None or li.is_junction:
            return 0.0
        if ego_yaw_deg is None or lane_yaw_deg is None:
            return 0.0

        # 最小夹角 [0,180]
        diff = self._angle_diff_deg(float(ego_yaw_deg), float(lane_yaw_deg))
        self._last_heading_diff_deg = diff

        # 在容差以内给正奖：diff=0 → +1；diff=tol → 0
        tol = float(self.heading_tol_deg)
        hmax = float(self.heading_max_deg)
        if diff <= tol:
            return 1.0 - diff / max(1e-6, tol)

        # 超过容差 → 线性变为负分：diff=tol → 0；diff=hmax → -1
        span = max(1e-6, hmax - tol)

        return -min(1.0, (diff - tol) / span)

    # 计算中心线横向偏移（米）
    def _lane_center_offset(self) -> Optional[float]:
        """返回 ego 到当前 lane center 的横向偏移 |y|（米）"""
        carla_map = self.world.get_map()
        ego_tf = self.ego.get_transform()
        wp = carla_map.get_waypoint(ego_tf.location, project_to_road=False,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        # waypoint 局部坐标：x前 y左；把 ego 位置投影到 wp 坐标求横向偏移
        dx = ego_tf.location.x - wp.transform.location.x
        dy = ego_tf.location.y - wp.transform.location.y
        yaw = math.radians(wp.transform.rotation.yaw)
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return abs(local_y)

    def _is_off_drivable_lane(self) -> bool:
        """
        True = 不在可行驶车道上
        注意：不用 project_to_road， 会把车吸回最近道路
        直接判定当前位置 waypoint 是否存在且为 Driving
        """
        carla_map = self.world.get_map()
        wp = carla_map.get_waypoint(self.ego.get_location(),
                                    project_to_road=False,
                                    lane_type=carla.LaneType.Driving)
        return wp is None

    def desired_speed(self) -> float:
        """综合限速/交通灯/跟车，返回 m/s"""
        # 车道限速
        v_lim = self._kmh_to_ms(self.raw_state.lane_info.speed_limit) if self.raw_state.lane_info else 30 / 3.6
        v_lim *= self.speed_limit_ratio

        # 红/黄灯减速规划
        tl = self.raw_state.traffic_light
        v_tl = float("inf")
        if tl and tl.state in (TrafficLightState.RED, TrafficLightState.YELLOW) and tl.distance < self.red_dist_trigger:
            stop_dist = max(0.0, tl.distance - self.red_stop_margin)
            v_tl = max(0.0, self.k_tl * stop_dist)  # 线性降到 0

        # 跟车约束
        v_gap = float("inf")
        nv = self._nearest_vehicle_ahead()
        if nv is not None:
            d = max(0.0, nv.distance)
            # 允许的跟驰速度 ~ k_gap * (d - d_min)
            v_gap = max(0.0, self.k_gap * (d - self.safe_dist)) if d < 30.0 else float("inf")
            v_gap = min(v_gap, nv.speed + self.follow_delta)  # 不快于前车太多

        # 取最小
        v_des = min(v_lim, v_tl, v_gap)
        if not math.isfinite(v_des):
            v_des = v_lim
        return max(0.0, v_des)

    # 奖励相关
    def _reward_components(self, action) -> Dict[str, float]:
        rs: Dict[str, float] = {}
        ego = self.raw_state.ego_info
        v = max(0.0, float(ego.speed))  # m/s

        # 速度项：接近 v_des 越好；需要停下时鼓励 v≈0
        v_des = self.desired_speed()
        if v_des <= 1e-3:
            r_speed = 1.0 if v < 0.5 else 0.0
        else:
            r_speed = max(0.0, 1.0 - abs(v - v_des) / max(v_des, 1.0))
        rs["speed"] = r_speed

        # 车道侵入（违法越线为 -1）
        rs["lane"] = -1.0 if (
                self.raw_state.lane_invasion.is_invasion and not self.raw_state.lane_invasion.is_legal
        ) else 0.0

        # 交通灯：闯红灯 -1；正确停下 +1
        rs["tl_violation"] = 0.0
        rs["tl_stop_bonus"] = 0.0
        tl = self.raw_state.traffic_light
        if tl and tl.state == TrafficLightState.RED and tl.distance < self.red_dist_trigger:
            if tl.distance < 2.5 and v > 1.0:
                rs["tl_violation"] = -1.0
            if tl.distance < 4.0 and v < 0.5:
                rs["tl_stop_bonus"] = 1.0

        # 跟车安全：过近为负
        rs["safety"] = 0.0
        nv = self._nearest_vehicle()
        if nv is not None and nv.distance < self.safe_dist:
            rs["safety"] = - (self.safe_dist - nv.distance) / self.safe_dist  # [-1,0]

        # 平滑项（动作差分为负）
        st, thr, brk = action
        pst, pthr, pbrk = self.prev_action
        rs["smooth"] = - (abs(st - pst) + abs(thr - pthr) + abs(brk - pbrk))

        # 存活小罚
        rs["alive"] = -1.0

        # 碰撞
        imp = float(getattr(self.raw_state.collision, "impulse", 0.0))
        if imp > 2.0:  # 低于 2 认为噪声
            rs["collision"] = -min(1.0, (imp - 2.0) / 50.0)  # 50 为满罚强度
        else:
            rs["collision"] = 0.0

        # 方向对齐
        gate, scale = _shape_speed_gate(v)

        # 中心线
        off = self._lane_center_offset()
        r_c = self._center_align_reward(off)  # ∈[-1,+1]
        if r_c > 0.0 and not gate:
            r_c = 0.0  # 原地不刷正奖
        r_c = r_c * scale  # 速度缩放
        r_c = max(-1.0, min(r_c, 0.2))
        rs["center"] = r_c

        # 方向
        ego_yaw_deg = float(self.raw_state.ego_info.rotation_yaw)
        lane_yaw_deg = self._lane_heading_deg()
        r_h = self._heading_align_reward(ego_yaw_deg, lane_yaw_deg)  # ∈[-1,+1]
        if r_h > 0.0 and not gate:
            r_h = 0.0
        r_h = r_h * scale
        r_h = max(-1.0, min(r_h, 0.2))
        rs["heading"] = r_h

        # 离开可行驶车道
        rs["offroad"] = -1.0 if self._is_off_drivable_lane() else 0.0

        # 记录调试信息
        self._last_center_off_m = off
        self._last_speed_gate = gate
        self._last_shape_scale = scale

        return rs

    def _aggregate_reward(self, comps: Dict[str, float]) -> float:
        return (
                self.w_v * comps["speed"]
                + self.w_lane * comps["lane"]
                + self.w_tlv * comps["tl_violation"]
                + self.w_tls * comps["tl_stop_bonus"]
                + self.w_safety * comps["safety"]
                + self.w_smooth * comps["smooth"]
                + self.w_alive * comps["alive"]
                + self.w_collision * comps["collision"]
                + self.w_heading * comps["heading"]
                + self.w_center * comps.get("center", 0.0)
                + self.w_offroad * comps.get("offroad", 0.0)
        )

    def _check_done(self):  # done, reason, offroad_terminate, stuck_terminate
        if self.raw_state.collision.is_collision:
            return True, "collision", False, False

        if self._is_off_drivable_lane():
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0
        if self.offroad_counter >= self.offroad_patience:
            return True, "offroad", True, False

        # 卡滞逻辑
        v = float(self.raw_state.ego_info.speed)
        tl = self.raw_state.traffic_light
        waiting_red = (tl and tl.state == TrafficLightState.RED and tl.distance < self.red_dist_trigger)
        nv = self._nearest_vehicle_ahead()
        following_close = (nv is not None and nv.distance < max(1.5 * self.safe_dist, 12.0))

        if v < 0.1:
            if waiting_red or following_close or self._is_off_drivable_lane():
                # 在红灯、跟车或 off-road 情况下的低速不再累计卡滞
                # off-road 的结束已经由 offroad_counter 控制
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.stuck_patience:
            return True, "stuck", False, True

        if self.step_count >= self.max_steps:
            return True, "max_steps", False, False

        return False, "running", False, False

    def step(self, action=None):
        """
        对 ego 车辆执行一个控制动作, 如果未提供action则随机
        control = carla.VehicleControl(
            throttle=0.5,     # 油门 0~1
            steer=-0.2,       # 转向 -1~1（左负右正）
            brake=0.0,        # 刹车 0~1
            hand_brake=False, # 手刹, 暂时不做
            reverse=False     # 是否倒车, 暂时不做
        )
        """
        if action is None:
            steer = random.uniform(-1.0, 1.0)
            throttle = random.uniform(0.0, 1.0)
            brake = random.uniform(0.0, 0.3)
        else:
            steer, throttle, brake = action

        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))  # type: ignore
        self.ego.apply_control(control)

        self.world.tick()

        self.step_count += 1

        comps = self._reward_components((steer, throttle, brake))
        reward = float(self._aggregate_reward(comps))
        done, done_reason, offroad_term, stuck_term = self._check_done()

        if config.enable_perturbation:
            self.raw_obs = attack(self.raw_state)
        else:
            self.raw_obs = self.raw_state

        info = {
            "r_speed": comps["speed"],
            "r_lane": comps["lane"],
            "r_tl_violation": comps["tl_violation"],
            "r_tl_stop_bonus": comps["tl_stop_bonus"],
            "r_safety": comps["safety"],
            "r_smooth": comps["smooth"],
            "collision": self.raw_state.collision.is_collision,
            "speed_mps": float(self.raw_state.ego_info.speed),
            "v_des": float(self.desired_speed()),
            "nearest_dist": float(self._nearest_vehicle().distance) if self._nearest_vehicle() else None,
            "tl_state": self.raw_state.traffic_light.state.name if self.raw_state.traffic_light else "NONE",
            "tl_dist": float(self.raw_state.traffic_light.distance) if self.raw_state.traffic_light else None,
            "r_heading": comps.get("heading", 0.0),
            "heading_diff_deg": self._last_heading_diff_deg,
            "center_offset_m": self._last_center_off_m,
            "offroad_flag": bool(self._is_off_drivable_lane()),
            "done_reason": done_reason,
            "offroad_terminate": offroad_term,
            "stuck_terminate": stuck_term
        }

        # 更新 prev_action
        self.prev_action = (float(steer), float(throttle), float(brake))

        return self.raw_obs, reward, done, info

    def reset(self, seed=None, return_info=False, options=None) -> RawState:

        # 清理上一回合注册的回调, 并重置世界
        self.world = reset_world(self.client, self.callback_id)

        # 在新回合里, 将原来的 callback_id 清空
        self.callback_id = None

        # 生成一些 NPC 车辆
        spawn_vehicles(self.world, vehicle_num=self.npc_vehicle_num)
        self.world.tick()

        # 启动 Traffic Manager（让 NPC 车辆开启自动驾驶）
        enable_traffic_manager(self.client, self.world)

        # 生成 Ego 车辆, 并记录引用
        self.ego = spawn_ego_vehicle(self.world)
        self.world.tick()

        # 为 Ego 车辆附加传感器, 并保存传感器字典
        self.raw_state = attach_sensors(self.world, self.ego)
        self.world.tick()

        # 注册世界的 Tick 回调
        callbacks = [
            create_follow_tick(self.ego),  # 让观众视角跟随 Ego, 不渲染不会执行
            reset_events_tick(self.raw_state),  # 每帧重置部分传感器事件
            update_state(self.world, self.ego, self.raw_state)
        ]
        self.callback_id = register_tick_callbacks(self.world, callbacks)

        # 做一次 Tick, 确保场景与状态更新
        self.world.tick()

        self.prev_action = (0.0, 0.0, 0.0)
        self.step_count = 0
        self.stuck_counter = 0
        self.offroad_counter = 0

        if config.enable_perturbation:
            self.raw_obs = attack(self.raw_state)
        else:
            self.raw_obs = self.raw_state

        # 返回观测
        return self.raw_state
