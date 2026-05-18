import random

from config import config
from entity.enums import TrafficLightState, CollisionType
from entity.types import EgoVehicleInfo, GnssData, TrafficLightInfo, LaneInfo, SurroundingVehicles, \
    VehicleInfo, CollisionData, LaneInvasionData
from environment.state import RawState


def attack(raw_state: RawState) -> RawState:
    if not getattr(config, "enable_perturbation", False):
        return raw_state

    pert = raw_state.copy_data()

    if getattr(config, "attack_ego", False):
        disturb_ego(pert.ego_info)

    if getattr(config, "attack_gnss", False):
        disturb_gnss(pert.gnss)

    if getattr(config, "attack_tl", False) and getattr(pert, "traffic_light", None) is not None:
        disturb_traffic_light(pert.traffic_light, raw_state)

    if getattr(config, "attack_lane", False) and getattr(pert, "lane_info", None) is not None:
        disturb_lane(pert.lane_info, raw_state)

    if getattr(config, "attack_vehicles", False) and getattr(pert, "surrounding_vehicles", None) is not None:
        disturb_vehicles(pert.surrounding_vehicles, raw_state)

    if getattr(config, "attack_collision", False):
        disturb_collision(pert.collision)

    if getattr(config, "attack_lane_invasion", False):
        disturb_lane_invasion(pert.lane_invasion)

    return pert


def disturb_ego(ego: EgoVehicleInfo):
    p = getattr(config, "ego_speed_attack_prob", 0.5)
    if random.random() < p:
        mode = getattr(config, "ego_speed_attack_mode", "scale")
        if mode == "scale":
            k_max = getattr(config, "ego_speed_scale_max", 0.3)
            k = random.uniform(-k_max, k_max)
            ego.speed = max(0.0, ego.speed * (1.0 + k))
        else:
            delta = getattr(config, "ego_speed_offset_max", 3.0)
            d = random.uniform(-delta, delta)
            ego.speed = max(0.0, ego.speed + d)

    q = getattr(config, "ego_yaw_attack_prob", 0.5)
    if random.random() < q:
        max_deg = getattr(config, "ego_yaw_offset_deg", 10.0)
        d = random.uniform(-max_deg, max_deg)
        ego.rotation_yaw = ego.rotation_yaw + d


def disturb_gnss(gnss: GnssData):
    lat_sigma = getattr(config, "gnss_lat_sigma", 1e-5)
    lon_sigma = getattr(config, "gnss_lon_sigma", 1e-5)
    alt_sigma = getattr(config, "gnss_alt_sigma", 0.5)

    gnss.latitude += random.gauss(0.0, lat_sigma)
    gnss.longitude += random.gauss(0.0, lon_sigma)
    gnss.altitude += random.gauss(0.0, alt_sigma)

    jump_prob = getattr(config, "gnss_jump_prob", 0.01)
    if random.random() < jump_prob:
        max_jump_deg = getattr(config, "gnss_jump_deg", 1e-3)
        gnss.latitude += random.uniform(-max_jump_deg, max_jump_deg)
        gnss.longitude += random.uniform(-max_jump_deg, max_jump_deg)


def disturb_traffic_light(tl: TrafficLightInfo, raw_state: RawState):
    if tl.state == TrafficLightState.UNKNOWN:
        return

    max_dist = getattr(config, "tl_attack_max_distance", 60.0)
    if tl.distance > max_dist:
        return

    ego_speed = raw_state.ego_info.speed
    min_speed = getattr(config, "tl_attack_min_speed", 1.0)
    if ego_speed < min_speed:
        return

    if random.random() > getattr(config, "tl_attack_prob", 0.3):
        return

    mode = getattr(config, "tl_attack_mode", "red_to_green")

    if mode == "red_to_green":
        if tl.state == TrafficLightState.RED:
            tl.state = TrafficLightState.GREEN
    elif mode == "red_to_unknown":
        if tl.state == TrafficLightState.RED:
            tl.state = TrafficLightState.UNKNOWN
    elif mode == "random_flip":
        mapping = {
            TrafficLightState.RED: TrafficLightState.GREEN,
            TrafficLightState.GREEN: TrafficLightState.RED,
            TrafficLightState.YELLOW: TrafficLightState.GREEN,
        }
        tl.state = mapping.get(tl.state, tl.state)
    elif mode == "dist_noise":
        sigma = getattr(config, "tl_dist_sigma", 5.0)
        tl.distance = max(0.0, tl.distance + random.gauss(0.0, sigma))


def disturb_lane(lane: LaneInfo, raw_state: RawState):
    if random.random() < getattr(config, "lane_yaw_attack_prob", 0.3):
        mode = getattr(config, "lane_yaw_attack_mode", "bias")
        if mode == "bias":
            max_deg = getattr(config, "lane_yaw_bias_deg", 20.0)
            d = random.uniform(-max_deg, max_deg)
            lane.yaw_diff = lane.yaw_diff + d
        elif mode == "flip":
            lane.yaw_diff = -lane.yaw_diff

    if random.random() < getattr(config, "lane_has_next_attack_prob", 0.1):
        if getattr(config, "lane_has_next_attack_mode", "drop") == "drop":
            lane.has_next = False
        else:
            lane.has_next = True


def disturb_vehicles(sv: SurroundingVehicles, raw_state: RawState):
    if not sv.vehicles:
        return

    if random.random() < getattr(config, "veh_hide_prob", 0.2):
        n_hide = getattr(config, "veh_hide_topk", 1)
        sv.vehicles.sort(key=lambda vehicle: vehicle.distance)
        del sv.vehicles[:min(n_hide, len(sv.vehicles))]

    if random.random() < getattr(config, "veh_fake_prob", 0.1):
        fake = VehicleInfo()
        fake.distance = getattr(config, "veh_fake_distance", 15.0)
        fake.speed = getattr(config, "veh_fake_speed", 0.0)
        fake.relative_x = fake.distance
        fake.relative_y = 0.0
        sv.vehicles.append(fake)

    dist_sigma = getattr(config, "veh_dist_sigma", 2.0)
    for v in sv.vehicles:
        if random.random() < getattr(config, "veh_dist_noise_prob", 0.5):
            v.distance = max(0.0, v.distance + random.gauss(0.0, dist_sigma))


def disturb_collision(col: CollisionData):
    fn_prob = getattr(config, "collision_fn_prob", 0.2)
    fp_prob = getattr(config, "collision_fp_prob", 0.05)

    if col.is_collision and random.random() < fn_prob:
        col.is_collision = False
        col.collision_type = CollisionType.NO_COLLISION
        col.impulse = 0.0
    elif (not col.is_collision) and random.random() < fp_prob:
        col.is_collision = True
        col.collision_type = CollisionType.OTHER
        col.impulse = getattr(config, "collision_fp_impulse", 10.0)


def disturb_lane_invasion(li: LaneInvasionData):
    fn_prob = getattr(config, "lane_inv_fn_prob", 0.2)
    fp_prob = getattr(config, "lane_inv_fp_prob", 0.05)

    if li.is_invasion and random.random() < fn_prob:
        li.is_invasion = False
        li.is_legal = True
        li.hold_frames = 0
    elif (not li.is_invasion) and random.random() < fp_prob:
        li.is_invasion = True
        li.is_legal = False
        li.hold_frames = getattr(config, "lane_inv_fp_hold_frames", 5)