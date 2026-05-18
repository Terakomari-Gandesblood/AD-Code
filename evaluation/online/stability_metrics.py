from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StabilityWeights:
    w_safety: float = 0.4   # 安全部分（碰撞、offroad、红灯）
    w_lane: float = 0.2     # 车道保持部分（中心偏移 / 航向差）
    w_smooth: float = 0.2   # 控制平滑性部分（动作差分）
    w_perf: float = 0.2     # 性能


@dataclass
class EpisodeMetrics:
    # 基本信息
    steps: int = 0
    max_steps: int = 0  # 计算生存比例

    collision: bool = False  # 是否发生碰撞
    offroad_terminate: bool = False  # 是否因 offroad 终止
    stuck_terminate: bool = False  # 是否因 stuck 终止
    done_reason: str = "unknown"  # 终止原因

    # 安全 / 规则
    offroad_steps: int = 0  # offroad 状态的累计步数
    tl_violation_count: int = 0  # 红灯违规计数

    # -- 轨迹质量 --
    # 名称：车道中心偏移绝对值累计和（米）
    # 公式：\sum |d_c| = \sum_{t=1}^{T} \left|d_{c,t}\right|
    sum_center_offset: float = 0.0

    # 名称：车道中心偏移绝对值最大值（米）（统计/诊断用，当前 lane_score 用均值）
    # 公式：\max |d_c| = \max_{t \in [1,T]} \left|d_{c,t}\right|
    max_center_offset: float = 0.0

    # 名称：航向差绝对值累计和（度）
    # 公式：\sum |\Delta\psi| = \sum_{t=1}^{T} \left|\Delta\psi_t\right|
    sum_heading_diff_deg: float = 0.0

    # 名称：航向差绝对值最大值（度）（统计/诊断用，当前 lane_score 用均值）
    # 公式：\max |\Delta\psi| = \max_{t \in [1,T]} \left|\Delta\psi_t\right|
    max_heading_diff_deg: float = 0.0

    # -- 平滑性 --
    # 名称：方向盘动作差分绝对值累计和
    # 公式：\sum |\Delta s| = \sum_{t=2}^{T} \left|s_t - s_{t-1}\right|
    sum_delta_steer: float = 0.0

    # 名称：油门动作差分绝对值累计和
    # 公式：\sum |\Delta u| = \sum_{t=2}^{T} \left|u_t - u_{t-1}\right|
    sum_delta_throttle: float = 0.0

    # 名称：刹车动作差分绝对值累计和
    # 公式：\sum |\Delta b| = \sum_{t=2}^{T} \left|b_t - b_{t-1}\right|
    sum_delta_brake: float = 0.0

    # -- 速度/简单性能 --
    # 名称：速度累计和（m/s）
    # 公式：\sum v = \sum_{t=1}^{T} v_t
    sum_speed: float = 0.0

    # 名称：速度误差绝对值累计和
    # 公式：\sum |e_v| = \sum_{t=1}^{T} \left|v_t - v^{\text{des}}_t\right|
    sum_speed_error: float = 0.0  # |v - v_des|

    # 名称：episode “成功”标志
    # 公式：
    # \mathbb{I}_{\text{succ}} =
    # \mathbb{I}\Big[\neg \mathbb{I}_{\text{coll}} \wedge \neg \mathbb{I}_{\text{offroad-term}}
    # \wedge \frac{T}{T_{\max}} \ge 0.9 \wedge \bar{v} \ge v_{\min}\Big]
    success_flag: bool = False    # 没有目的地则用安全生存+速度正常定义

    # 名称：安全得分 S_{\text{safety}} \in [0,1]
    # 公式：
    # S_{\text{safety}} = \max\Big(0,\ 1
    # -0.6\mathbb{I}_{\text{coll}}
    # -0.3\mathbb{I}_{\text{offroad-term}}
    # -0.1\mathbb{I}_{\text{stuck-term}}
    # -\min(0.3,\ 0.05N_{\text{tl}})\Big)
    safety_score: float = 0.0

    # 名称：车道保持得分 S_{\text{lane}} \in [0,1]
    # 公式：
    # \bar{d}_c = \frac{1}{T}\sum_{t=1}^{T}|d_{c,t}|,\quad
    # \overline{\Delta\psi} = \frac{1}{T}\sum_{t=1}^{T}|\Delta\psi_t|
    # lane\_pos = e^{-1.0\bar{d}_c},\quad lane\_dir = e^{-0.02\overline{\Delta\psi}}
    # S_{\text{lane}} = \mathrm{clip}_{[0,1]}\Big(0.5\cdot lane\_pos + 0.5\cdot lane\_dir\Big)
    lane_score: float = 0.0

    # 名称：控制平滑得分 S_{\text{smooth}} \in [0,1]
    # 公式（T>1 时）：
    # \overline{\Delta s} = \frac{1}{T-1}\sum_{t=2}^{T}|s_t-s_{t-1}|
    # \overline{\Delta u} = \frac{1}{T-1}\sum_{t=2}^{T}|u_t-u_{t-1}|
    # \overline{\Delta b} = \frac{1}{T-1}\sum_{t=2}^{T}|b_t-b_{t-1}|
    # \overline{\Delta a} = \frac{\overline{\Delta s}+\overline{\Delta u}+\overline{\Delta b}}{3}
    # S_{\text{smooth}} = \mathrm{clip}_{[0,1]}\big(e^{-2.0\overline{\Delta a}}\big)
    # 边界：T \le 1 时，S_{\text{smooth}}=1
    smooth_score: float = 0.0

    # 名称：性能/效率得分 S_{\text{perf}} \in [0,1]
    # 公式：
    # r_{\text{survive}} = \frac{T}{T_{\max}},\quad \bar{v}=\frac{1}{T}\sum_{t=1}^{T} v_t
    # v_{\text{norm}} = \min(1,\ \bar{v}/15)
    # base = 0.5r_{\text{survive}} + 0.5v_{\text{norm}}
    # 若 \mathbb{I}_{\text{succ}}=1:\ base=\min(1,\ base+0.1)
    # S_{\text{perf}}=\mathrm{clip}_{[0,1]}(base)
    # 注：当前 eval 中可能设置 w_{\text{perf}}=0，因此该项可不影响总分但仍计算。
    perf_score: float = 0.0

    # 名称：最终稳定性得分 S_{ep} \in [0,1]
    # 公式：
    # S_{ep} = w_s S_{\text{safety}} + w_l S_{\text{lane}} + w_{sm} S_{\text{smooth}} + w_p S_{\text{perf}}
    # 其中 (w_s,w_l,w_{sm},w_p) 来自 StabilityWeights
    stability_score: float = 0.0  # 最终 S_ep

    # 附加 debug 信息
    extras: Dict[str, float] = field(default_factory=dict)

    # 评分计算逻辑
    def compute_scores(self, weights: StabilityWeights) -> None:

        # 1) 安全部分：越少事故越好
        self.safety_score = self._compute_safety_score()

        # 2) 车道保持：中心偏移和航向差越小越好
        self.lane_score = self._compute_lane_score()

        # 3) 平滑度：动作变化越小越好
        self.smooth_score = self._compute_smooth_score()

        # 4) 性能/效率：在没有目标点时，用“生存比例 + 平均速度”近似
        self.perf_score = self._compute_perf_score()

        # 5) 总稳定性分
        w = weights
        self.stability_score = (
            w.w_safety * self.safety_score
            + w.w_lane * self.lane_score
            + w.w_smooth * self.smooth_score
            + w.w_perf * self.perf_score
        )

    # 各子分项的默认实现
    def _compute_safety_score(self) -> float:
        # 基于是否碰撞、是否 offroad 终止、是否 stuck 终止、red-light 次数
        score = 1.0

        if self.collision:
            score -= 0.6
        if self.offroad_terminate:
            score -= 0.3
        if self.stuck_terminate:
            score -= 0.1

        # 红灯违规次数越多，扣得越多
        score -= min(0.3, 0.05 * self.tl_violation_count)

        return max(0.0, score)

    def _compute_lane_score(self) -> float:
        if self.steps <= 0:
            return 0.0
        mean_center = self.sum_center_offset / self.steps
        mean_heading = self.sum_heading_diff_deg / self.steps

        # 简单的指数衰减映射到 (0,1]
        import math
        lane_pos = math.exp(-1.0 * mean_center)  # 中心偏移越大越差
        lane_dir = math.exp(-0.02 * mean_heading)  # 航向差越大越差
        score = 0.5 * lane_pos + 0.5 * lane_dir
        return max(0.0, min(1.0, score))

    def _compute_smooth_score(self) -> float:
        if self.steps <= 1:
            return 1.0  # 单步没法看平滑度，先给满分

        mean_d_steer = self.sum_delta_steer / (self.steps - 1)
        mean_d_thr = self.sum_delta_throttle / (self.steps - 1)
        mean_d_brk = self.sum_delta_brake / (self.steps - 1)

        avg_delta = (mean_d_steer + mean_d_thr + mean_d_brk) / 3.0

        import math
        score = math.exp(-2.0 * avg_delta)  # 0→1; delta大→趋近0
        return max(0.0, min(1.0, score))

    def _compute_perf_score(self) -> float:
        if self.steps <= 0 or self.max_steps <= 0:
            return 0.0

        survival_ratio = self.steps / float(self.max_steps)
        mean_speed = self.sum_speed / self.steps

        # 假设 0~15 m/s 是有效区间
        speed_norm = min(1.0, mean_speed / 15.0)

        # 成功条件：在 90% 以上步数存活且无碰撞/offroad 终止/严重违规，且平均速度大于 v_min
        v_min = 2.0  # m/s，可调
        self.success_flag = (
            not self.collision
            and not self.offroad_terminate
            and survival_ratio >= 0.9
            and mean_speed >= v_min
        )

        # success 给一部分奖励，生存+速度给连续奖励
        base = 0.5 * survival_ratio + 0.5 * speed_norm
        if self.success_flag:
            base = min(1.0, base + 0.1)

        return max(0.0, min(1.0, base))
