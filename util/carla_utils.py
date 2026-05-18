import math
import random
import carla
import time
import subprocess
from config import config
import socket

from entity.enums import TrafficLightState, CollisionType, LaneMarkingType
from environment.state import RawState, SurroundingVehicles, VehicleInfo, TrafficLightInfo, LaneInfo


def is_port_open(host, port):
    """检查指定端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # 设置超时时间，避免长时间阻塞
        return s.connect_ex((host, port)) == 0  # 端口可连接返回 True


def start_carla(low_quality=False, off_screen=False, max_wait=config.max_wait_time):
    """启动 Carla 模拟器"""
    carla_path = config.carla_path
    # 构造启动参数
    args = []
    if low_quality:
        args.append("-quality-level=Low")  # 低负载渲染
    if off_screen:
        args.append("-RenderOffScreen")  # 无窗口模式
    # 拼接命令
    cmd = [carla_path] + args
    # 运行命令
    subprocess.Popen(cmd, shell=True)
    print(f"Carla 启动命令: {' '.join(cmd)}")

    # 等待 Carla 服务器启动
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if is_port_open(config.carla_service, config.port):
            print("Carla 服务器已启动")
            return True
        time.sleep(1)  # 每秒检查一次
    print("错误: Carla 启动超时！")
    return False  # 如果超时返回 False


def get_client(max_retries=config.max_wait_time, base_timeout=config.connect_timeout):
    """连接 Carla服务器 """
    if not is_port_open(config.carla_service, config.port):
        print('无服务, 启动本地服务')
        start_carla(True)
    client = carla.Client(config.carla_service, config.port)

    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (attempt + 1)  # 超时时间逐渐增加
            client.set_timeout(timeout)
            world = client.get_world()  # type: ignore
            print(f"Carla 连接成功!")
            return client, world
        except Exception as e:
            print(f"连接失败，重试 {attempt + 1}/{max_retries}：{e}")
            time.sleep(2)  # 等待 2 秒后重试

    raise RuntimeError("Carla 连接失败，请检查服务器是否启动")


def limit_fps(world, fps=config.default_fps):
    """设置 Carla 的最大帧率"""
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / fps  # 设置固定时间步长
    world.apply_settings(settings)
    print(f"已将 FPS 限制为 {fps}")


def set_render_mode(world, no_rendering_mode=False, synchronous_mode=True, fps=config.default_fps):
    """设置渲染模式"""
    settings = world.get_settings()
    settings.no_rendering_mode = no_rendering_mode  # 是否关闭渲染
    settings.synchronous_mode = synchronous_mode  # 是否同步渲染
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    print("渲染模式已修改")


def reset_world(client, callback_id=None):
    """重置世界"""
    world = client.get_world()
    settings = world.get_settings()

    # 暂停仿真避免冲突
    was_sync = settings.synchronous_mode
    if was_sync:
        settings.synchronous_mode = False
        world.apply_settings(settings)

    # 移除视角绑定回调
    if callback_id is not None:
        try:
            world.remove_on_tick(callback_id)
            print(f"已移除回调 ID: {callback_id}")
        except Exception as e:
            print(f"移除回调失败: {e}")

    # 获取所有 actor
    actors = world.get_actors()
    to_destroy = [a for a in actors if a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.')]

    # 清理所有非 ego 的动态实体
    if to_destroy:
        print(f"销毁 {len(to_destroy)} 个 NPC")
        for actor in to_destroy:
            actor.destroy()

    # 重置天气
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 重置仿真时间
    world.tick()  # 触发一次更新
    print("世界已重置完毕")

    # 恢复同步模式
    if was_sync:
        settings.synchronous_mode = True
        world.apply_settings(settings)

    world.tick()
    time.sleep(1)

    return client.get_world()


def reload_world(client, new_world='/Game/Carla/Maps/Town10HD_Opt', timeout=config.max_wait_time):
    """渲染新的地图"""
    maps = client.get_available_maps()  # type: ignore
    if not maps:
        raise RuntimeError("无可用地图")

    if new_world not in maps:
        print(f"地图 {new_world} 不存在，返回当前世界")
        return client.get_world()
    else:
        print(f"正在加载地图: {new_world}")
        client.set_timeout(timeout)  # 增加超时时间，避免超时
        client.load_world(new_world)

        # 等待 CARLA 服务器稳定运行
        time.sleep(config.connect_timeout)

        world = client.get_world()
        print(f"新地图已加载: {world.get_map().name}")
        return world


def blueprint_info(world):
    """打印所有的蓝图信息"""
    blueprint_library = world.get_blueprint_library()
    print(f"CARLA 蓝图库共包含 {len(blueprint_library)} 个蓝图")

    vehicles = blueprint_library.filter('*vehicle*')
    print(f"可用车辆模型: {len(vehicles)}")
    for vehicle in vehicles:
        print("-", vehicle.id)

    walkers = blueprint_library.filter('*walker*')
    print(f"可用行人模型: {len(walkers)}")
    for walker in walkers:
        print("-", walker.id)

    sensors = blueprint_library.filter('*sensor*')
    print(f"可用传感器模型: {len(sensors)}")
    for sensor in sensors:
        print("-", sensor.id)

    static_objects = blueprint_library.filter('*static*')
    print(f"可用静态物体: {len(static_objects)}")
    for obj in static_objects:
        print("-", obj.id)


def spawn_vehicles(world, vehicle_num=config.vehicle_num):
    """生成汽车"""
    blueprint_library = world.get_blueprint_library()

    # 获取所有车辆和行人蓝图
    vehicles = blueprint_library.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()  # 获取所有可用的出生点

    # 计算可用的最大生成数
    max_spawn_num = min(len(spawn_points), vehicle_num)
    selected_spawn_points = random.sample(spawn_points, max_spawn_num)

    spawned_vehicles = []
    for i in range(max_spawn_num):
        blueprint = random.choice(vehicles)  # 允许车型重复
        spawn_point = selected_spawn_points[i]

        vehicle = world.try_spawn_actor(blueprint, spawn_point)

        if vehicle:
            spawned_vehicles.append(vehicle)

    return spawned_vehicles


def spawn_ego_vehicle(world, spawn_index=None):
    """生成自主车辆"""
    # 获取车辆蓝图并设置 role_name
    ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz')
    ego_bp.set_attribute('role_name', 'hero')

    # 获取所有可用出生点
    spawn_points = world.get_map().get_spawn_points()

    # 获取已使用的位置（已有车辆的 transform）
    occupied_transforms = [v.get_transform() for v in world.get_actors().filter('vehicle.*')]

    # 定义是否使用指定位置
    if spawn_index is not None and 0 <= spawn_index < len(spawn_points):
        candidate_points = [spawn_points[spawn_index]]
    else:
        # 随机选择未被占用的出生点
        candidate_points = random.sample(spawn_points, len(spawn_points))

    ego = None
    for point in candidate_points:
        # 判断是否和其他车辆位置重叠（距离太近）
        if all(point.location.distance(o.location) > 2.5 for o in occupied_transforms):
            ego = world.try_spawn_actor(ego_bp, point)
            if ego:
                break

    if ego is None:
        raise RuntimeError("无法找到可用出生点，Ego 车辆未生成")

    print(f"Ego 车辆生成于: {ego.get_location()}")

    return ego


def enable_traffic_manager(client, world, synchronous_mode=True):
    """应用交通管理自动驾驶其他车辆"""
    # 获取 Traffic Manager
    tm = client.get_trafficmanager(config.traffic_manager_port)
    if synchronous_mode:
        tm.set_synchronous_mode(True)

    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True, tm.get_port())

    return tm


def create_follow_tick(vehicle, distance=6.0, height=2.5):
    """ 视角跟随 """
    spectator = vehicle.get_world().get_spectator()

    def follow():
        transform = vehicle.get_transform()
        location = transform.location - transform.get_forward_vector() * distance
        location.z += height
        spectator.set_transform(carla.Transform(location, transform.rotation))

    return follow


def register_tick_callbacks(world, callbacks):
    """将多个 tick 逻辑合并注册到 world 的 tick 回调中"""

    def tick_handler(_):
        for callback in callbacks:
            callback()

    return world.on_tick(tick_handler)


def update_gnss(raw_state, data):
    raw_state.gnss.latitude = data.latitude
    raw_state.gnss.longitude = data.longitude
    raw_state.gnss.altitude = data.altitude


def update_imu(raw_state, data):
    raw_state.imu.accelerometer_x = data.accelerometer.x
    raw_state.imu.accelerometer_y = data.accelerometer.y
    raw_state.imu.accelerometer_z = data.accelerometer.z
    raw_state.imu.gyroscope_x = data.gyroscope.x
    raw_state.imu.gyroscope_y = data.gyroscope.y
    raw_state.imu.gyroscope_z = data.gyroscope.z
    raw_state.imu.compass = data.compass


def update_collision(raw_state, event):
    print("collision")
    raw_state.collision.is_collision = True

    # 判断碰撞对象类型
    actor_type = event.other_actor.type_id
    if actor_type.startswith("vehicle."):
        raw_state.collision.collision_type = CollisionType.VEHICLE
    elif actor_type.startswith("walker."):
        raw_state.collision.collision_type = CollisionType.WALKER
    else:
        raw_state.collision.collision_type = CollisionType.STATIC

    # 计算冲击力大小
    impulse = event.normal_impulse
    impulse_norm = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5
    raw_state.collision.impulse = impulse_norm


def is_marking_legal(lane_type: LaneMarkingType) -> bool:
    # return marking_type in [
    #     LaneMarkingType.BROKEN,
    #     LaneMarkingType.UNKNOWN,
    # ]
    # 单/双虚线一般允许跨越
    if lane_type in {LaneMarkingType.BROKEN}:
        return True
    # 明确禁止：纯实线、双实线
    if lane_type in {LaneMarkingType.SOLID, LaneMarkingType.SOLID_SOLID}:
        return False
    # 实虚组合（SingleSolid+Broken）：方向相关；不知道方向就保守按不合法
    if lane_type in {LaneMarkingType.BROKEN_SOLID, LaneMarkingType.SOLID_BROKEN}:
        return False
    # 其他/未知：默认合法，避免噪声误伤
    return True


def update_lane_invasion(raw_state: RawState, event):
    # raw_state.lane_invasion.is_invasion = True
    # raw_state.lane_invasion.is_legal = True
    # for marking in event.crossed_lane_markings:
    #     try:
    #         lane_type = LaneMarkingType(marking.type.name)
    #     except ValueError:
    #         lane_type = LaneMarkingType.UNKNOWN
    #
    #     # 不合法线设置为 False
    #     if not is_marking_legal(lane_type):
    #         raw_state.lane_invasion.is_legal = False
    #         break
    inv = raw_state.lane_invasion
    inv.is_invasion = True
    inv.is_legal = True
    inv.hold_frames = inv.decay_len

    inv.last_markings = []
    for marking in event.crossed_lane_markings:
        try:
            lane_type = LaneMarkingType(marking.type.name)
            inv.last_markings.append(lane_type.name)
        except ValueError:
            lane_type = LaneMarkingType.UNKNOWN
            inv.last_markings.append("UNKNOWN")

        if not is_marking_legal(lane_type):
            inv.is_legal = False  # 只要混入不合法线，整段窗口都视作不合法


def attach_sensors(world, ego) -> RawState:
    raw_state = RawState()

    blueprint_library = world.get_blueprint_library()
    pos_high = carla.Transform(carla.Location(x=1.0, z=2.0))  # type: ignore
    pos_flat = carla.Transform()  # type: ignore

    # GNSS
    gnss_bp = blueprint_library.find('sensor.other.gnss')

    gnss_bp.set_attribute('sensor_tick', '0.05')

    gnss = world.spawn_actor(gnss_bp, pos_high, attach_to=ego)
    raw_state.gnss.sensor = gnss
    gnss.listen(lambda data: update_gnss(raw_state, data))

    # IMU
    # imu_bp = blueprint_library.find('sensor.other.imu')
    # imu_bp.set_attribute('sensor_tick', '0.05')
    # imu = world.spawn_actor(imu_bp, pos_high, attach_to=ego)
    # raw_state.imu.sensor = imu
    # imu.listen(lambda data: update_imu(raw_state, data))

    # Collision
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, pos_flat, attach_to=ego)
    raw_state.collision.sensor = collision
    collision.listen(lambda event: update_collision(raw_state, event))

    # Lane Invasion
    lane_bp = blueprint_library.find('sensor.other.lane_invasion')
    lane = world.spawn_actor(lane_bp, pos_flat, attach_to=ego)
    raw_state.lane_invasion.sensor = lane
    lane.listen(lambda event: update_lane_invasion(raw_state, event))

    return raw_state


def reset_events_tick(raw_state):
    def reset():
        # 变道情况
        inv = raw_state.lane_invasion
        if inv.hold_frames > 0:
            inv.hold_frames -= 1
            inv.is_invasion = True  # 保持点亮
            # inv.is_legal 保持上次回调判定
        else:
            inv.is_invasion = False
            inv.is_legal = True
            inv.last_markings = None

        # 碰撞情况
        # 如果碰撞是结束条件则需要注释否则会在判断前重置
        # 如果不是结束条件则可以取消注释
        # raw_state.collision.is_collision = False
        # raw_state.collision.collision_type = CollisionType.NO_COLLISION
        # raw_state.collision.impulse = 0.0

    return reset


def create_vehicle_info(actor: carla.Actor, ego, distance) -> VehicleInfo:
    ego_location = ego.get_location()

    transform = actor.get_transform()  # type: ignore
    velocity = actor.get_velocity()  # type: ignore

    location = transform.location
    rotation = transform.rotation

    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    relative_x = location.x - ego_location.x
    relative_y = location.y - ego_location.y
    relative_z = location.z - ego_location.z

    ego_extent = ego.bounding_box.extent
    extent = actor.bounding_box.extent

    world = actor.get_world()  # type: ignore
    carla_map = world.get_map()
    wp_other = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_width = getattr(wp_other, "lane_width", None)
    road_id = getattr(wp_other, "road_id", None)
    section_id = getattr(wp_other, "section_id", None)
    lane_id = getattr(wp_other, "lane_id", None)

    return VehicleInfo(
        vehicle_id=actor.id,

        location_x=location.x,
        location_y=location.y,
        location_z=location.z,

        rotation_pitch=rotation.pitch,
        rotation_yaw=rotation.yaw,
        rotation_roll=rotation.roll,

        velocity_x=velocity.x,
        velocity_y=velocity.y,
        velocity_z=velocity.z,

        distance=distance - ego_extent.x - extent.x,
        speed=speed,

        relative_x=relative_x,
        relative_y=relative_y,
        relative_z=relative_z,

        extent_x=extent.x,
        extent_y=extent.y,
        extent_z=extent.z,

        road_id=road_id,
        section_id=section_id,
        lane_id=lane_id,
        lane_width=lane_width
    )


def get_surrounding_vehicles(world, ego, radius=config.surrounding_radius) -> SurroundingVehicles:
    ego_loc = ego.get_location()
    actors = world.get_actors().filter("vehicle.*")

    result = SurroundingVehicles()

    for actor in actors:
        if actor.id == ego.id:
            continue

        distance = actor.get_location().distance(ego_loc)
        if distance <= radius:
            vehicle_info = create_vehicle_info(actor, ego, distance)
            result.add_vehicle(vehicle_info)

    return result


def get_nearest_traffic_light(world, ego, max_dist=config.tf_max_dis) -> TrafficLightInfo:
    """
    返回与 ego 当前车道相关联的红绿灯及到停止线的距离。
    找不到则返回 UNKNOWN + max_dist。
    """
    carla_map = world.get_map()
    ego_tf = ego.get_transform()
    ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return TrafficLightInfo.no_light()

    ego_road_id = ego_wp.road_id
    ego_section_id = ego_wp.section_id
    ego_lane_id = ego_wp.lane_id

    tl_actors = world.get_actors().filter('traffic.traffic_light*')
    best_dist = float('inf')
    best_state = None

    # Ego前向单位向量（用于判断“前方”）
    yaw = math.radians(ego_tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)

    for tl in tl_actors:
        # 关键：用停止线 waypoints 来判断是否“控制了我的车道”
        try:
            stop_wps = tl.get_stop_waypoints()  # List[carla.Waypoint]
        except Exception:
            stop_wps = []

        if not stop_wps:
            continue

        # 选择与 ego 同 road/section，且 lane_id 相同
        cand_wps = []
        for wp in stop_wps:
            same_road = (wp.road_id == ego_road_id)
            same_section = (wp.section_id == ego_section_id)
            # 你也可以放开相邻车道：abs(wp.lane_id - ego_lane_id) <= 1
            same_lane = (wp.lane_id == ego_lane_id)
            if same_road and same_section and same_lane:
                cand_wps.append(wp)

        if not cand_wps:
            continue

        # 取与 ego 最近的停止线点
        # 注意：距离用 几何直线距离 + “是否在前方”的过滤
        for wp in cand_wps:
            sp = wp.transform.location
            dx, dy = sp.x - ego_tf.location.x, sp.y - ego_tf.location.y

            # 在车前方
            dot = dx * fx + dy * fy
            if dot <= 0:
                continue

            dist = math.hypot(dx, dy)
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_state = tl.state  # carla.TrafficLightState

    if best_state is None:
        return TrafficLightInfo.no_light()

    # 映射到你的枚举
    if best_state == carla.TrafficLightState.Red:
        state = TrafficLightState.RED
    elif best_state == carla.TrafficLightState.Yellow:
        state = TrafficLightState.YELLOW
    elif best_state == carla.TrafficLightState.Green:
        state = TrafficLightState.GREEN
    else:
        state = TrafficLightState.UNKNOWN

    return TrafficLightInfo(distance=best_dist, state=state)


def signed_angle_to_target(ego, target_location):
    ego_tf = ego.get_transform()

    # 车辆前向向量
    fwd = ego_tf.get_forward_vector()  # carla.Vector3D

    # 从车指向目标点的向量（只考虑平面上的 x,y）
    to_target = target_location - ego_tf.location
    to_target.z = 0.0

    # 归一化
    fwd_norm = math.sqrt(fwd.x ** 2 + fwd.y ** 2) + 1e-6
    tgt_norm = math.sqrt(to_target.x ** 2 + to_target.y ** 2) + 1e-6

    fx, fy = fwd.x / fwd_norm, fwd.y / fwd_norm
    tx, ty = to_target.x / tgt_norm, to_target.y / tgt_norm

    # 计算有符号夹角 [-180, 180]
    dot = fx * tx + fy * ty
    dot = max(-1.0, min(1.0, dot))  # 防止数值误差
    cross = fx * ty - fy * tx  # 2D 叉积 z 分量（符号代表左右）

    angle_rad = math.atan2(cross, dot)
    angle_deg = math.degrees(angle_rad)
    return angle_deg  # 正负号可以约定：>0 表示目标在左，需要往左打方向


def get_lane_info(world, ego, lookahead=config.lookahead):
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(ego.get_location())  # 获取 Ego 车辆当前位置的车道点
    lane_info = LaneInfo()

    if wp is None:
        lane_info.has_next = False
        return lane_info

    lane_info.lane_id = wp.lane_id
    lane_info.is_junction = wp.is_junction
    lane_info.lane_change = wp.lane_change
    lane_info.speed_limit = ego.get_speed_limit()

    next_wp_list = wp.next(lookahead)  # 在指定距离之后，查找下一个车道点
    # 如果没有下一个点, 说明道路尽头
    if not next_wp_list:
        lane_info.has_next = False
    else:
        lane_info.has_next = True

    next_wp = next_wp_list[0]

    # 计算下一个车道点与当前车道点的航向差（角度差），单位为度
    # raw_diff = next_wp.transform.rotation.yaw - wp.transform.rotation.yaw
    # yaw_diff = (raw_diff + 180) % 360 - 180  # 转换到 [-180, 180] 范围
    # lane_info.yaw_diff = yaw_diff

    # ego_yaw = ego.get_transform().rotation.yaw
    # lane_yaw = wp.transform.rotation.yaw
    # raw_heading_diff = lane_yaw - ego_yaw
    # lane_info.yaw_diff = (raw_heading_diff + 180) % 360 - 180

    raw_diff = next_wp.transform.rotation.yaw - wp.transform.rotation.yaw
    lane_info.yaw_diff = (raw_diff + 180) % 360 - 180
    lane_info.yaw_diff = signed_angle_to_target(ego, next_wp.transform.location)

    return lane_info


def update_ego_info(raw_state: RawState, ego):
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    control = ego.get_control()
    extent = ego.bounding_box.extent

    raw_state.ego_info.location_x = transform.location.x
    raw_state.ego_info.location_y = transform.location.y
    raw_state.ego_info.location_z = transform.location.z

    raw_state.ego_info.rotation_pitch = transform.rotation.pitch
    raw_state.ego_info.rotation_yaw = transform.rotation.yaw
    raw_state.ego_info.rotation_roll = transform.rotation.roll

    raw_state.ego_info.velocity_x = velocity.x
    raw_state.ego_info.velocity_y = velocity.y
    raw_state.ego_info.velocity_z = velocity.z
    raw_state.ego_info.speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    raw_state.ego_info.throttle = control.throttle
    raw_state.ego_info.steer = control.steer
    raw_state.ego_info.brake = control.brake

    raw_state.ego_info.extent_x = extent.x
    raw_state.ego_info.extent_y = extent.y
    raw_state.ego_info.extent_z = extent.z


def update_state(world, ego, raw_state):
    def update():
        update_ego_info(raw_state, ego)  # 更新自我车辆的信息
        raw_state.surrounding_vehicles = get_surrounding_vehicles(world, ego)
        raw_state.traffic_light = get_nearest_traffic_light(world, ego)
        raw_state.lane_info = get_lane_info(world, ego)

    return update


def simulate_adversarial_behavior(ego, tick_count):
    """模拟被攻击后的驾驶行为，加入随机扰动"""
    control = carla.VehicleControl()  # type: ignore

    # 匀速前进
    control.throttle = 0.4
    control.brake = 0.0

    # 基础扰动：带随机相位的正弦函数
    # base_steer = 0.3 * math.sin(tick_count * 0.1 + random.uniform(-0.5, 0.5))

    # 叠加随机高频扰动（更突兀的失控感）
    noise = random.uniform(-0.15, 0.15)

    # 总转向角 = 基础扰动 + 随机扰动
    control.steer = max(-1.0, min(1.0, noise))

    # 偶尔突然加速（每50 tick 触发一次）
    if tick_count % 50 == 0 and random.random() < 0.5:
        control.throttle = random.uniform(0.6, 1.0)
        print(f"[!] Adversarial Spike: sudden throttle at tick {tick_count:.0f}")

    ego.apply_control(control)
