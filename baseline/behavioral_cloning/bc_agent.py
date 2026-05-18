from typing import TypedDict, Optional

import torch
from itertools import chain
from config import config
from baseline.behavioral_cloning.bc_network import ResidualBCActor
from model.encoder_net import EncodingNetwork
from environment.state import RawState, ExtractedState


class EncodedFeatureDict(TypedDict):
    ego: torch.Tensor
    gnss: torch.Tensor
    traffic_light: torch.Tensor
    lane: torch.Tensor
    nearest_vehicle: torch.Tensor
    vehicles: Optional[torch.Tensor]


class BCResidualAgent:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ego_encoder = EncodingNetwork(
            config.encoding_layers, config.ego_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.gnss_encoder = EncodingNetwork(
            config.encoding_layers, config.gnss_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.vehicle_encoder = EncodingNetwork(
            config.encoding_layers, config.vehicle_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.tl_encoder = EncodingNetwork(
            config.encoding_layers, config.tl_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        self.lane_encoder = EncodingNetwork(
            config.encoding_layers, config.lane_feature_dim,
            config.encoding_hidden_dim, config.node_feature_dim
        ).to(self.device)

        input_dim = 6 * config.node_feature_dim
        self.actor = ResidualBCActor(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=3,
            alpha=config.alpha,
            residual_scale=tuple(config.residual_scale),
            action_clamp_steer=tuple(config.action_clamp_steer),
            action_clamp_throttle=tuple(config.action_clamp_throttle),
            action_clamp_brake=tuple(config.action_clamp_brake),
        ).to(self.device)

        self.actor_param_list = list(chain(
            self.ego_encoder.parameters(),
            self.gnss_encoder.parameters(),
            self.vehicle_encoder.parameters(),
            self.tl_encoder.parameters(),
            self.lane_encoder.parameters(),
            self.actor.parameters(),
        ))

    def encoding(self, es: ExtractedState) -> EncodedFeatureDict:
        device = self.device

        ego_v = es.ego_vector.to(device).unsqueeze(0)
        gnss_v = es.gnss_vector.to(device).unsqueeze(0)
        tl_v = es.tl_vector.to(device).unsqueeze(0)
        lane_v = es.lane_vector.to(device).unsqueeze(0)
        nearest_v = es.nearest_vector.to(device).unsqueeze(0)

        ego_feat = self.ego_encoder(ego_v)
        gnss_feat = self.gnss_encoder(gnss_v)
        tl_feat = self.tl_encoder(tl_v)
        lane_feat = self.lane_encoder(lane_v)
        nearest_feat = self.vehicle_encoder(nearest_v)

        if es.vehicles_tensor is not None:
            vehicles_tensor = es.vehicles_tensor.to(device)
            vehicles_feat = self.vehicle_encoder(vehicles_tensor)
        else:
            vehicles_feat = None

        return {
            "ego": ego_feat,
            "gnss": gnss_feat,
            "traffic_light": tl_feat,
            "lane": lane_feat,
            "nearest_vehicle": nearest_feat,
            "vehicles": vehicles_feat,
        }

    def get_global_feat(self, feats: EncodedFeatureDict) -> torch.Tensor:
        device = self.device

        ego = feats["ego"]
        gnss = feats["gnss"]
        tl = feats["traffic_light"]
        lane = feats["lane"]
        nearest = feats["nearest_vehicle"]

        parts = [ego, gnss, tl, lane, nearest]

        veh = feats["vehicles"]
        if veh is not None and veh.numel() > 0:
            veh_mean = veh.mean(dim=0, keepdim=True)
        else:
            veh_mean = torch.zeros_like(ego, device=device)
        parts.append(veh_mean)

        global_feat = torch.cat(parts, dim=-1)
        return global_feat

    def global_feat_from_raw(self, raw_state: RawState) -> torch.Tensor:
        es = ExtractedState.from_raw(raw_state, self.device)
        feats = self.encoding(es)
        return self.get_global_feat(feats)

    def global_feat_from_dict(self, rec: dict, dict_to_raw_fn) -> torch.Tensor:
        raw = dict_to_raw_fn(rec)
        return self.global_feat_from_raw(raw)
