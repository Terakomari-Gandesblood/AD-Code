from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from config import config


@dataclass(frozen=True)
class AttackIntensityLevel:
    level: int
    name: str
    desc: str = ""


INTENSITY_LEVELS: List[AttackIntensityLevel] = [
    AttackIntensityLevel(0, "L0", "off/very weak"),
    AttackIntensityLevel(1, "L1", "weak"),
    AttackIntensityLevel(2, "L2", "medium"),
    AttackIntensityLevel(3, "L3", "strong"),
]

# key 必须与 disturb_* 内 getattr(config, "xxx", default) 完全一致
INTENSITY_PRESETS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "tl_dist": {
        0: {"tl_attack_prob": 0.0, "tl_dist_sigma": 0.0},
        1: {"tl_attack_prob": 0.2, "tl_dist_sigma": 2.0},
        2: {"tl_attack_prob": 0.5, "tl_dist_sigma": 5.0},
        3: {"tl_attack_prob": 0.8, "tl_dist_sigma": 8.0},
    },
    "tl_color": {
        0: {"tl_attack_prob": 0.0},
        1: {"tl_attack_prob": 0.2},
        2: {"tl_attack_prob": 0.5},
        3: {"tl_attack_prob": 0.8},
    },
    "lane_yaw": {
        0: {"lane_yaw_attack_prob": 0.0, "lane_yaw_bias_deg": 0.0},
        1: {"lane_yaw_attack_prob": 0.2, "lane_yaw_bias_deg": 8.0},
        2: {"lane_yaw_attack_prob": 0.5, "lane_yaw_bias_deg": 15.0},
        3: {"lane_yaw_attack_prob": 0.8, "lane_yaw_bias_deg": 25.0},
    },
    "lane_has_next": {
        0: {"lane_has_next_attack_prob": 0.0},
        1: {"lane_has_next_attack_prob": 0.2},
        2: {"lane_has_next_attack_prob": 0.5},
        3: {"lane_has_next_attack_prob": 0.8},
    },
    "veh_noise": {
        0: {"veh_dist_noise_prob": 0.0, "veh_dist_sigma": 0.0},
        1: {"veh_dist_noise_prob": 0.3, "veh_dist_sigma": 1.0},
        2: {"veh_dist_noise_prob": 0.6, "veh_dist_sigma": 2.0},
        3: {"veh_dist_noise_prob": 0.9, "veh_dist_sigma": 4.0},
    },
    "veh_hide": {
        0: {"veh_hide_prob": 0.0},
        1: {"veh_hide_prob": 0.3},
        2: {"veh_hide_prob": 0.6},
        3: {"veh_hide_prob": 0.9},
    },
    "veh_fake": {
        0: {"veh_fake_prob": 0.0},
        1: {"veh_fake_prob": 0.2},
        2: {"veh_fake_prob": 0.5},
        3: {"veh_fake_prob": 0.8},
    },
    "ego": {
        0: {"ego_speed_attack_prob": 0.0, "ego_speed_scale_max": 0.0, "ego_yaw_attack_prob": 0.0, "ego_yaw_offset_deg": 0.0},
        1: {"ego_speed_attack_prob": 0.2, "ego_speed_scale_max": 0.10, "ego_yaw_attack_prob": 0.2, "ego_yaw_offset_deg": 5.0},
        2: {"ego_speed_attack_prob": 0.5, "ego_speed_scale_max": 0.20, "ego_yaw_attack_prob": 0.5, "ego_yaw_offset_deg": 10.0},
        3: {"ego_speed_attack_prob": 0.8, "ego_speed_scale_max": 0.30, "ego_yaw_attack_prob": 0.8, "ego_yaw_offset_deg": 15.0},
    },
    "gnss": {
        0: {"gnss_lat_sigma": 0.0, "gnss_lon_sigma": 0.0, "gnss_alt_sigma": 0.0, "gnss_jump_prob": 0.0, "gnss_jump_deg": 0.0},
        1: {"gnss_lat_sigma": 5e-6, "gnss_lon_sigma": 5e-6, "gnss_alt_sigma": 0.2, "gnss_jump_prob": 0.005, "gnss_jump_deg": 5e-4},
        2: {"gnss_lat_sigma": 1e-5, "gnss_lon_sigma": 1e-5, "gnss_alt_sigma": 0.5, "gnss_jump_prob": 0.01, "gnss_jump_deg": 1e-3},
        3: {"gnss_lat_sigma": 2e-5, "gnss_lon_sigma": 2e-5, "gnss_alt_sigma": 1.0, "gnss_jump_prob": 0.03, "gnss_jump_deg": 2e-3},
    },
    "lane_inv": {
        0: {"lane_inv_fn_prob": 0.0, "lane_inv_fp_prob": 0.0},
        1: {"lane_inv_fn_prob": 0.1, "lane_inv_fp_prob": 0.02},
        2: {"lane_inv_fn_prob": 0.2, "lane_inv_fp_prob": 0.05},
        3: {"lane_inv_fn_prob": 0.3, "lane_inv_fp_prob": 0.10},
    },
    "collision": {
        0: {"collision_fn_prob": 0.0, "collision_fp_prob": 0.0},
        1: {"collision_fn_prob": 0.1, "collision_fp_prob": 0.02},
        2: {"collision_fn_prob": 0.2, "collision_fp_prob": 0.05},
        3: {"collision_fn_prob": 0.3, "collision_fp_prob": 0.10},
    },
}


def apply_intensity_to_config(scenario_key: str, level: int) -> None:
    """把场景和档位 preset 写入 config"""
    preset = INTENSITY_PRESETS.get(scenario_key, {}).get(int(level), None)
    if preset:
        for k, v in preset.items():
            setattr(config, k, v)
    config.attack_intensity_level = int(level)


@dataclass
class AttackScenario:
    name: str
    desc: str = ""
    attack_ego: bool = False
    attack_gnss: bool = False
    attack_tl: bool = False
    attack_lane: bool = False
    attack_vehicles: bool = False
    attack_collision: bool = False
    attack_lane_invasion: bool = False
    extra_cfg: Dict[str, Any] = field(default_factory=dict)

    def apply(self, level: Optional[int] = None) -> None:
        # 模块开关
        config.enable_perturbation = any([
            self.attack_ego, self.attack_gnss, self.attack_tl,
            self.attack_lane, self.attack_vehicles,
            self.attack_collision, self.attack_lane_invasion,
        ])

        config.attack_ego = self.attack_ego
        config.attack_gnss = self.attack_gnss
        config.attack_tl = self.attack_tl
        config.attack_lane = self.attack_lane
        config.attack_vehicles = self.attack_vehicles
        config.attack_collision = self.attack_collision
        config.attack_lane_invasion = self.attack_lane_invasion

        # 场景结构参数
        for k, v in self.extra_cfg.items():
            setattr(config, k, v)

        # 强度参数覆盖
        if level is not None:
            apply_intensity_to_config(self.name, int(level))
        else:
            config.attack_intensity_level = -1


def make_default_scenarios() -> Dict[str, AttackScenario]:
    return {
        "clean": AttackScenario(name="clean", desc="无攻击"),

        "ego": AttackScenario(name="ego", desc="Ego 自身状态扰动", attack_ego=True),

        "gnss": AttackScenario(name="gnss", desc="GNSS 噪声 + 偶发大跳变", attack_gnss=True),

        "tl_dist": AttackScenario(
            name="tl_dist",
            desc="红绿灯距离扰动",
            attack_tl=True,
            extra_cfg={
                "tl_attack_mode": "dist_noise",
                "tl_attack_max_distance": 60.0,
                "tl_attack_min_speed": 1.0,
            },
        ),

        "tl_color": AttackScenario(
            name="tl_color",
            desc="红绿灯灯色扰动（red->green 等）",
            attack_tl=True,
            extra_cfg={
                "tl_attack_mode": "red_to_green",
                "tl_attack_max_distance": 60.0,
                "tl_attack_min_speed": 1.0,
            },
        ),

        "lane_yaw": AttackScenario(
            name="lane_yaw",
            desc="车道方向 yaw_diff 扰动",
            attack_lane=True,
            extra_cfg={
                "lane_yaw_attack_mode": "bias",
                "lane_has_next_attack_prob": 0.0,
            },
        ),

        "lane_has_next": AttackScenario(
            name="lane_has_next",
            desc="车道 has_next 错报",
            attack_lane=True,
            extra_cfg={
                "lane_yaw_attack_prob": 0.0,
                "lane_has_next_attack_mode": "drop",
            },
        ),

        "veh_noise": AttackScenario(
            name="veh_noise",
            desc="车辆距离噪声",
            attack_vehicles=True,
            extra_cfg={
                "veh_hide_prob": 0.0,
                "veh_fake_prob": 0.0,
            },
        ),

        "veh_hide": AttackScenario(
            name="veh_hide",
            desc="隐藏最近车辆（漏检）",
            attack_vehicles=True,
            extra_cfg={
                "veh_hide_topk": 1,
                "veh_fake_prob": 0.0,
                "veh_dist_noise_prob": 0.0,
            },
        ),

        "veh_fake": AttackScenario(
            name="veh_fake",
            desc="加入幻影车辆（虚假感知）",
            attack_vehicles=True,
            extra_cfg={
                "veh_hide_prob": 0.0,
                "veh_fake_distance": 15.0,
                "veh_fake_speed": 0.0,
            },
        ),

        "lane_inv": AttackScenario(
            name="lane_inv",
            desc="车道入侵传感器误报/漏报",
            attack_lane_invasion=True,
        ),

        "collision": AttackScenario(
            name="collision",
            desc="碰撞传感器误报/漏报",
            attack_collision=True,
        ),
    }
