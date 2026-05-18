import json
from util.carla_utils import get_client, set_render_mode, reset_world, spawn_vehicles, spawn_ego_vehicle, \
    enable_traffic_manager, create_follow_tick, register_tick_callbacks, attach_sensors, reset_events_tick, update_state


def save_data(data, file_index):
    filename = f"../save/dataset/expert_data/data_{file_index}.json"
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Data saved to {filename}")


def serialize_raw_state(raw_state):
    return {
        "ego_info": {
            "location_x": raw_state.ego_info.location_x,
            "location_y": raw_state.ego_info.location_y,
            "location_z": raw_state.ego_info.location_z,
            "rotation_pitch": raw_state.ego_info.rotation_pitch,
            "rotation_yaw": raw_state.ego_info.rotation_yaw,
            "rotation_roll": raw_state.ego_info.rotation_roll,
            "velocity_x": raw_state.ego_info.velocity_x,
            "velocity_y": raw_state.ego_info.velocity_y,
            "velocity_z": raw_state.ego_info.velocity_z,
            "speed": raw_state.ego_info.speed,
            "throttle": raw_state.ego_info.throttle,
            "steer": raw_state.ego_info.steer,
            "brake": raw_state.ego_info.brake,
            "extent_x": raw_state.ego_info.extent_x,
            "extent_y": raw_state.ego_info.extent_y,
            "extent_z": raw_state.ego_info.extent_z
        },
        "gnss": {
            "latitude": raw_state.gnss.latitude,
            "longitude": raw_state.gnss.longitude,
            "altitude": raw_state.gnss.altitude
        },
        # "imu": {
        #     "accelerometer_x": raw_state.imu.accelerometer_x,
        #     "accelerometer_y": raw_state.imu.accelerometer_y,
        #     "accelerometer_z": raw_state.imu.accelerometer_z,
        #     "gyroscope_x": raw_state.imu.gyroscope_x,
        #     "gyroscope_y": raw_state.imu.gyroscope_y,
        #     "gyroscope_z": raw_state.imu.gyroscope_z,
        #     "compass": raw_state.imu.compass
        # },
        "collision": {
            "collision_type": raw_state.collision.collision_type.name,
            "impulse": raw_state.collision.impulse
        },
        "lane_invasion": {
            "is_invasion": raw_state.lane_invasion.is_invasion,
            "is_legal": raw_state.lane_invasion.is_legal
        },
        "surrounding_vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "location_x": v.location_x,
                "location_y": v.location_y,
                "location_z": v.location_z,
                "velocity_x": v.velocity_x,
                "velocity_y": v.velocity_y,
                "velocity_z": v.velocity_z,
                "distance": v.distance,
                "speed": v.speed,
                "relative_x": v.relative_x,
                "relative_y": v.relative_y,
                "relative_z": v.relative_z,
                "extent_x": v.extent_x,
                "extent_y": v.extent_y,
                "extent_z": v.extent_z
            }
            for v in raw_state.surrounding_vehicles.vehicles[:5]
        ],
        "traffic_light": {
            "state": raw_state.traffic_light.state.name,
            "distance": raw_state.traffic_light.distance
        },
        "lane_info": {
            "lane_id": raw_state.lane_info.lane_id,
            "has_next": raw_state.lane_info.has_next,
            "is_junction": raw_state.lane_info.is_junction,
            "yaw_diff": raw_state.lane_info.yaw_diff,
            "lane_change": raw_state.lane_info.lane_change,
            "speed_limit": raw_state.lane_info.speed_limit
        }
    }


def collect():
    client, world = get_client()
    callback_id = None
    set_render_mode(world, no_rendering_mode=False, synchronous_mode=True)
    world = reset_world(client, callback_id)
    spawn_vehicles(world, vehicle_num=10)

    ego = spawn_ego_vehicle(world, 1)
    world.tick()

    tm = enable_traffic_manager(client, world)
    ego.set_autopilot(True, tm.get_port())
    world.tick()

    raw_state = attach_sensors(world, ego)
    world.tick()

    data_buffer = []
    file_index = 1

    callbacks = [
        create_follow_tick(ego),
        reset_events_tick(raw_state),
        update_state(world, ego, raw_state)
    ]
    callback_id = register_tick_callbacks(world, callbacks)

    # 主循环
    while True:
        world.tick()  # 推进世界状态
        serialized_data = serialize_raw_state(raw_state)  # 序列化数据
        data_buffer.append(serialized_data)

        # 每1000帧后保存数据
        if len(data_buffer) >= 1000:
            save_data(data_buffer, file_index)
            file_index += 1
            data_buffer.clear()  # 清空列表


if __name__ == '__main__':
    collect()
