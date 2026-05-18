import math
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import config
from data_collection.date_reader import CarlaBCDataset
from entity.enums import TrafficLightState, CollisionType
from entity.types import LaneInfo, TrafficLightInfo, VehicleInfo, SurroundingVehicles, LaneInvasionData, CollisionData, \
    GnssData, EgoVehicleInfo
from environment.state import RawState
from baseline.behavioral_cloning.bc_agent import BCResidualAgent


def dict_to_raw_state(data_dict) -> RawState:
    """
        将字典数据转换回RawState对象
    """
    raw_state = RawState()

    # ego_info
    if "ego_info" in data_dict:
        ego_data = data_dict["ego_info"]
        raw_state.ego_info = EgoVehicleInfo(
            location_x=ego_data.get("location_x", 0.0),
            location_y=ego_data.get("location_y", 0.0),
            location_z=ego_data.get("location_z", 0.0),
            rotation_pitch=ego_data.get("rotation_pitch", 0.0),
            rotation_yaw=ego_data.get("rotation_yaw", 0.0),
            rotation_roll=ego_data.get("rotation_roll", 0.0),
            velocity_x=ego_data.get("velocity_x", 0.0),
            velocity_y=ego_data.get("velocity_y", 0.0),
            velocity_z=ego_data.get("velocity_z", 0.0),
            speed=ego_data.get("speed", 0.0),
            throttle=ego_data.get("throttle", 0.0),
            steer=ego_data.get("steer", 0.0),
            brake=ego_data.get("brake", 0.0),
            extent_x=ego_data.get("extent_x", 0.0),
            extent_y=ego_data.get("extent_y", 0.0),
            extent_z=ego_data.get("extent_z", 0.0)
        )

    # gnss
    if "gnss" in data_dict:
        gnss_data = data_dict["gnss"]
        raw_state.gnss = GnssData(
            sensor=None,
            latitude=gnss_data.get("latitude", 0.0),
            longitude=gnss_data.get("longitude", 0.0),
            altitude=gnss_data.get("altitude", 0.0)
        )

    # collision
    if "collision" in data_dict:
        collision_data = data_dict["collision"]
        raw_state.collision = CollisionData(
            is_collision=collision_data.get("impulse", 0.0) > 0,
            collision_type=CollisionType[collision_data.get("collision_type", "NO_COLLISION")],
            impulse=collision_data.get("impulse", 0.0)
        )

    # lane_invasion
    if "lane_invasion" in data_dict:
        lane_invasion_data = data_dict["lane_invasion"]
        raw_state.lane_invasion = LaneInvasionData(
            is_invasion=lane_invasion_data.get("is_invasion", False),
            is_legal=lane_invasion_data.get("is_legal", True)
        )

    # surrounding_vehicles
    if "surrounding_vehicles" in data_dict:
        surrounding_vehicles = SurroundingVehicles()
        vehicles_data = data_dict["surrounding_vehicles"]
        for vehicle_data in vehicles_data:
            vehicle_info = VehicleInfo(
                vehicle_id=vehicle_data.get("vehicle_id", 0),
                location_x=vehicle_data.get("location_x", 0.0),
                location_y=vehicle_data.get("location_y", 0.0),
                location_z=vehicle_data.get("location_z", 0.0),
                velocity_x=vehicle_data.get("velocity_x", 0.0),
                velocity_y=vehicle_data.get("velocity_y", 0.0),
                velocity_z=vehicle_data.get("velocity_z", 0.0),
                distance=vehicle_data.get("distance", config.surrounding_radius),
                speed=vehicle_data.get("speed", 0.0),
                relative_x=vehicle_data.get("relative_x", config.surrounding_radius),
                relative_y=vehicle_data.get("relative_y", config.surrounding_radius),
                relative_z=vehicle_data.get("relative_z", config.surrounding_radius),
                extent_x=vehicle_data.get("extent_x", 0.0),
                extent_y=vehicle_data.get("extent_y", 0.0),
                extent_z=vehicle_data.get("extent_z", 0.0)
            )
            surrounding_vehicles.add_vehicle(vehicle_info)
        raw_state.surrounding_vehicles = surrounding_vehicles

    # traffic_light
    if "traffic_light" in data_dict:
        traffic_light_data = data_dict["traffic_light"]
        raw_state.traffic_light = TrafficLightInfo(
            distance=traffic_light_data.get("distance", config.tf_max_dis),
            state=TrafficLightState[traffic_light_data.get("state", "UNKNOWN")]
        )

    # lane_info
    if "lane_info" in data_dict:
        lane_info_data = data_dict["lane_info"]
        lane_info = LaneInfo()
        lane_info.lane_id = lane_info_data.get("lane_id", 0)
        lane_info.has_next = lane_info_data.get("has_next", False)
        lane_info.is_junction = lane_info_data.get("is_junction", False)
        lane_info.yaw_diff = lane_info_data.get("yaw_diff", 0.0)
        lane_info.lane_change = lane_info_data.get("lane_change", False)
        lane_info.speed_limit = lane_info_data.get("speed_limit", 999.0)
        raw_state.lane_info = lane_info

    return raw_state


def compute_errors_from_record(rec: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    lane_info = rec["lane_info"]
    ego_info = rec["ego_info"]

    yaw_diff_deg = float(lane_info.get("yaw_diff", 0.0) or 0.0)
    yaw_err = torch.tensor([math.radians(yaw_diff_deg)], dtype=torch.float32, device=device)

    v_des = float(lane_info.get("speed_limit", 0.0) or 0.0)
    v_cur = float(ego_info.get("speed", 0.0) or 0.0)
    v_err = torch.tensor([v_des - v_cur], dtype=torch.float32, device=device)

    return yaw_err, v_err


def get_expert_action_from_record(rec: dict, device: torch.device) -> torch.Tensor:
    ego = rec["ego_info"]
    steer = float(ego["steer"])
    throttle = float(ego["throttle"])
    brake = float(ego["brake"])
    return torch.tensor([steer, throttle, brake], dtype=torch.float32, device=device)  # (3,)


def train_bc_residual(data_dir: str,
                      num_epochs: int = 10,
                      batch_size: int = 64,
                      lr: float = 3e-4,
                      device: torch.device = None,
                      save_best_only: bool = True,
                      save_last: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CarlaBCDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: x
    )

    agent = BCResidualAgent(device=device)
    actor = agent.actor
    optimizer = torch.optim.Adam(agent.actor_param_list, lr=lr)

    save_dir = config.behavioral_cloning_save_path
    os.makedirs(save_dir, exist_ok=True)

    best_path = os.path.join(save_dir, "bc_residual_best.pt")
    last_path = os.path.join(save_dir, "bc_residual_last.pt")

    best_loss = float("inf")

    alpha = config.alpha
    residual_scale = torch.tensor(config.residual_scale, dtype=torch.float32, device=device).view(1, 3)

    for epoch in range(num_epochs):
        actor.train()
        total_loss = 0.0
        total_samples = 0

        for batch_records in dataloader:
            B = len(batch_records)

            feats, yaw_errs, v_errs, residual_labels = [], [], [], []

            for rec in batch_records:
                global_feat = agent.global_feat_from_dict(rec, dict_to_raw_state)
                feats.append(global_feat)

                yaw_err, v_err = compute_errors_from_record(rec, device=device)
                yaw_errs.append(yaw_err)
                v_errs.append(v_err)

                a_exp = get_expert_action_from_record(rec, device=device).view(1, 3)

                with torch.no_grad():
                    tmpl = actor.template_action(yaw_err, v_err, dt=1.0 / config.default_fps)

                denom = alpha * residual_scale
                r_label = (a_exp - tmpl) / denom
                r_label = torch.clamp(r_label, -1.0, 1.0)
                residual_labels.append(r_label)

            feats = torch.cat(feats, dim=0)
            yaw_errs = torch.cat(yaw_errs, dim=0)
            v_errs = torch.cat(v_errs, dim=0)
            residual_labels = torch.cat(residual_labels, dim=0)

            _, pred_residual = actor(feats, yaw_errs, v_errs)
            loss = F.mse_loss(pred_residual, residual_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        avg_loss = total_loss / max(total_samples, 1)
        print(f"[BC Residual] Epoch {epoch} | Avg Loss = {avg_loss:.6f} | Samples = {total_samples}")

        if save_best_only and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "best_loss": best_loss,
                "actor_state_dict": agent.actor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "alpha": config.alpha,
                    "residual_scale": config.residual_scale,
                    "default_fps": config.default_fps,
                    "node_feature_dim": config.node_feature_dim,
                }
            }, best_path)
            print(f"[BC Residual] New best saved: loss={best_loss:.6f} -> {best_path}")

        if save_last:
            torch.save({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "actor_state_dict": agent.actor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, last_path)

    final_path = os.path.join(config.behavioral_cloning_save_path, "bc_residual_final.pt")
    torch.save(agent.actor.state_dict(), final_path)
    print(f"[BC Residual] Final actor saved -> {final_path}")


if __name__ == "__main__":
    train_bc_residual(data_dir="../save/dataset/expert_data", num_epochs=10, batch_size=64, lr=3e-4)
