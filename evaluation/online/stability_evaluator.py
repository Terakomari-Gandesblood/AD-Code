from typing import Dict, Tuple, Optional

from stability_metrics import EpisodeMetrics, StabilityWeights


class StabilityEvaluator:

    def __init__(self, weights: Optional[StabilityWeights] = None):
        self.weights = weights or StabilityWeights()
        self._metrics: Optional[EpisodeMetrics] = None
        self._prev_action: Optional[Tuple[float, float, float]] = None

    # Episode 生命周期
    def start_episode(self, max_steps: int) -> None:
        """在每个 episode 开始时调用"""
        self._metrics = EpisodeMetrics(max_steps=max_steps)
        self._prev_action = None

    def step(self, info: Dict, action: Tuple[float, float, float]) -> None:
        """
        每个 env.step() 后调用一次，用于累积统计信息。
        """
        if self._metrics is None:
            raise RuntimeError("Call start_episode() before step().")

        m = self._metrics
        m.steps += 1

        # 安全规则
        collision = bool(info.get("collision", False))
        if collision:
            m.collision = True

        if info.get("offroad_flag", False):
            m.offroad_steps += 1

        if info.get("r_tl_violation", 0.0) < 0.0:
            m.tl_violation_count += 1

        # 轨迹质量
        center_off = info.get("center_offset_m", None)
        if center_off is not None:
            m.sum_center_offset += abs(center_off)
            m.max_center_offset = max(m.max_center_offset, abs(center_off))

        heading_diff = info.get("heading_diff_deg", None)
        if heading_diff is not None:
            m.sum_heading_diff_deg += abs(heading_diff)
            m.max_heading_diff_deg = max(m.max_heading_diff_deg, abs(heading_diff))

        # 平滑性
        st, thr, brk = action
        if self._prev_action is not None:
            pst, pthr, pbrk = self._prev_action
            m.sum_delta_steer += abs(st - pst)
            m.sum_delta_throttle += abs(thr - pthr)
            m.sum_delta_brake += abs(brk - pbrk)
        self._prev_action = (st, thr, brk)

        # 性能
        speed = info.get("speed_mps", None)
        if speed is not None:
            m.sum_speed += float(speed)

        v_des = info.get("v_des", None)
        if speed is not None and v_des is not None:
            m.sum_speed_error += abs(float(speed) - float(v_des))

        # 其他 debug 信息
        for k in ("r_speed", "r_lane", "r_smooth", "r_heading"):
            if k in info:
                m.extras.setdefault(k + "_sum", 0.0)
                m.extras[k + "_sum"] += float(info[k])

    def end_episode(self, done_reason: str, offroad_terminate: bool, stuck_terminate: bool) -> EpisodeMetrics:
        if self._metrics is None:
            raise RuntimeError("No episode to end. Call start_episode() first.")

        m = self._metrics
        m.done_reason = done_reason
        m.offroad_terminate = bool(offroad_terminate)
        m.stuck_terminate = bool(stuck_terminate)

        # 计算各子分项和总分
        m.compute_scores(self.weights)

        # 清理内部状态
        self._metrics = None
        self._prev_action = None

        return m
