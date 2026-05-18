from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class StabilityWeights:
    """各子指标在总稳定性得分中的权重"""
    w_safety: float = 0.4  # 安全类指标
    w_lane: float = 0.3  # 轨迹质量类指标
    w_smooth: float = 0.1  # 平滑性类指标
    w_performance: float = 0.2  # 表现类指标


@dataclass
class StabilityNormConfig:
    # 车道偏移：期望 mean_center_offset < 0.2m 属于高分区间
    lane_offset_scale: float = 1.5

    # 航向偏差：期望 mean_heading_diff < 10° 属于高分区间
    heading_scale: float = 3.0

    # 动作变化：期望平均 Δ action < 0.1 属于高分区间
    action_delta_scale: float = 5.0

    # 速度跟随误差：期望 mean_speed_error < 1.0 m/s 属于高分区间
    speed_error_scale: float = 1.0


@dataclass
class EpisodeMetrics:
    # === 基本信息 ===
    steps: int = 0
    total_reward: float = 0.0

    # === 安全/规则相关 ===
    collision: bool = False
    tl_violation_steps: int = 0  # 有闯红灯的步数
    offroad_steps: int = 0  # 在 offroad 上的步数
    stuck_terminate: bool = False  # 是否因为卡死结束
    offroad_terminate: bool = False  # 是否因为持续 offroad 结束

    # === 车道/轨迹质量 ===
    center_offsets: List[float] = field(default_factory=list)  # 每步 |center_offset|
    heading_diffs: List[float] = field(default_factory=list)  # 每步 |heading_diff_deg|

    # === 平滑性 ===
    delta_steers: List[float] = field(default_factory=list)
    delta_throttles: List[float] = field(default_factory=list)
    delta_brakes: List[float] = field(default_factory=list)

    # === 任务完成效率 ===
    success: bool = False  # 是否完成任务
    time_to_goal: Optional[float] = None  # 完成任务耗时（秒），未完成可以是 None
    avg_speed: Optional[float] = None  # episode 平均速度
    speed_errors: List[float] = field(default_factory=list)  # |v - v_des| 每步

    # === 预计算的均值 ===
    mean_center_offset: float = 0.0
    mean_heading_diff: float = 0.0
    mean_delta_action: float = 0.0
    mean_speed_error: float = 0.0

    def finalize_means(self) -> None:
        """在 end_episode 时调用，统一算一次均值。"""

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        self.mean_center_offset = _mean(self.center_offsets)
        self.mean_heading_diff = _mean(self.heading_diffs)
        self.mean_delta_action = _mean(
            [*self.delta_steers, *self.delta_throttles, *self.delta_brakes]
        ) if (self.delta_steers or self.delta_throttles or self.delta_brakes) else 0.0
        self.mean_speed_error = _mean(self.speed_errors)

    def safety_score(self) -> float:
        if self.collision:
            return 0.0

        score = 1.0
        # 有闯红灯，打 0.8 折
        if self.tl_violation_steps > 0:
            score *= 0.8

        # offroad 多则继续打折，假设 offroad_steps / steps 是比率
        if self.steps > 0 and self.offroad_steps > 0:
            ratio = self.offroad_steps / self.steps
            score *= max(0.0, 1.0 - ratio)  # offroad 越多越低

        # 卡死 / 持续 offroad 终止再打一次折扣
        if self.stuck_terminate:
            score *= 0.7
        if self.offroad_terminate:
            score *= 0.5

        return max(0.0, min(1.0, score))

    def lane_score(self, norm: StabilityNormConfig) -> float:
        """根据中心线偏移 + 航向偏差做一个 [0,1] 的车道保持得分。"""
        # 采用 exp(-k * x) 形式，x 越大惩罚越强
        lane_part = math.exp(-norm.lane_offset_scale * self.mean_center_offset)
        heading_part = math.exp(-norm.heading_scale * (self.mean_heading_diff / 180.0))
        score = lane_part * heading_part
        return max(0.0, min(1.0, score))

    def smooth_score(self, norm: StabilityNormConfig) -> float:
        """动作变化越小，得分越高。"""
        score = math.exp(-norm.action_delta_scale * self.mean_delta_action)
        return max(0.0, min(1.0, score))

    def performance_score(self, norm: StabilityNormConfig) -> float:
        """
        任务完成 + 速度跟随。
        - 成功：基础分 1.0
        - 失败：基础分 0.2
        """
        base = 1.0 if self.success else 0.2
        # 速度跟随误差越小，得分越高
        follow = math.exp(-norm.speed_error_scale * self.mean_speed_error)
        score = base * follow
        return max(0.0, min(1.0, score))

    def stability_score(
            self,
            weights: StabilityWeights,
            norm: StabilityNormConfig
    ) -> Dict[str, float]:
        """
        返回包含各子得分和总稳定性分的字典。
        """
        self.finalize_means()

        s_safety = self.safety_score()
        s_lane = self.lane_score(norm)
        s_smooth = self.smooth_score(norm)
        s_perf = self.performance_score(norm)

        # 归一化权重
        w_sum = (
                weights.w_safety
                + weights.w_lane
                + weights.w_smooth
                + weights.w_performance
        )
        ws = StabilityWeights(
            w_safety=weights.w_safety / w_sum,
            w_lane=weights.w_lane / w_sum,
            w_smooth=weights.w_smooth / w_sum,
            w_performance=weights.w_performance / w_sum,
        )

        total = (
                ws.w_safety * s_safety +
                ws.w_lane * s_lane +
                ws.w_smooth * s_smooth +
                ws.w_performance * s_perf
        )

        return {
            "safety": s_safety,
            "lane": s_lane,
            "smooth": s_smooth,
            "performance": s_perf,
            "total": total,
        }


class StabilityEvaluator:

    def __init__(
            self,
            weights: Optional[StabilityWeights] = None,
            norm_cfg: Optional[StabilityNormConfig] = None,
    ):
        self.weights = weights or StabilityWeights()
        self.norm_cfg = norm_cfg or StabilityNormConfig()

        self._metrics = EpisodeMetrics()
        self._prev_action = None  # type: Optional[tuple]

        # 外部可以设置
        self._success_flag = False
        self._goal_time = None  # type: Optional[float]

    def start_episode(self):
        self._metrics = EpisodeMetrics()
        self._prev_action = None
        self._success_flag = False
        self._goal_time = None

    def mark_success(self, success: bool, time_to_goal: Optional[float] = None):
        """在任务完成时由外部调用。"""
        self._success_flag = success
        self._goal_time = time_to_goal

    def end_episode(
            self,
            done_reason: Optional[str] = None,
    ) -> EpisodeMetrics:
        """
        Episode 结束时调用。
        """
        if done_reason is not None:
            if done_reason == "stuck":
                self._metrics.stuck_terminate = True
            elif done_reason == "offroad":
                self._metrics.offroad_terminate = True

        self._metrics.success = self._success_flag
        self._metrics.time_to_goal = self._goal_time

        # 均值在 compute_scores 时会自动 finalize
        self._metrics.finalize_means()
        return self._metrics

    def step(
            self,
            info: Dict,
            action: Optional[tuple] = None,
            reward: Optional[float] = None,
    ):
        """
        每次 env.step 后调用一次。
        - info：来自 env.step(...)[3]，需要包含一些约定好的 key。
        - action：本步执行的动作 (steer, throttle, brake)。
        - reward：本步 RL 奖励，可选。
        """
        m = self._metrics

        # 步数 & 总 reward
        m.steps += 1
        if reward is not None:
            m.total_reward += float(reward)

        # 碰撞
        if bool(info.get("collision", False)):
            m.collision = True

        # 闯红灯
        if info.get("r_tl_violation", 0.0) < 0.0:
            m.tl_violation_steps += 1

        # offroad
        if info.get("offroad_flag", False):
            m.offroad_steps += 1

        # 车道/方向
        center_off = info.get("center_offset_m", None)
        if center_off is not None:
            m.center_offsets.append(abs(float(center_off)))

        heading_diff = info.get("heading_diff_deg", None)
        if heading_diff is not None:
            m.heading_diffs.append(abs(float(heading_diff)))

        # 速度跟随误差
        v = info.get("speed_mps", None)
        v_des = info.get("v_des", None)
        if v is not None and v_des is not None:
            m.speed_errors.append(abs(float(v) - float(v_des)))

        # 平滑性：动作差分
        if action is not None:
            if self._prev_action is not None:
                st, thr, brk = action
                pst, pthr, pbrk = self._prev_action
                m.delta_steers.append(abs(float(st) - float(pst)))
                m.delta_throttles.append(abs(float(thr) - float(pthr)))
                m.delta_brakes.append(abs(float(brk) - float(pbrk)))
            self._prev_action = tuple(map(float, action))

    def compute_scores(self) -> Dict[str, float]:
        """
        生成当前 episode 的各项得分。
        需要在 end_episode 之后调用（否则 success/终止原因可能不完整）。
        """
        return self._metrics.stability_score(self.weights, self.norm_cfg)


def robustness_ratio(clean_score: float, attacked_score: float, eps: float = 1e-6) -> float:
    """
    返回 attacked / clean，作为简单的鲁棒性指标。
    """
    return attacked_score / (clean_score + eps)
