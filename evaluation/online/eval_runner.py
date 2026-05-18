import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Protocol, Optional

from config import config
from environment.carla_env import CarlaEnv
from evaluation.algorithm.behavioral_cloning import BCEvalPolicy
from evaluation.algorithm.gnn_flat_ppo import GnnFlatPPOEvalPolicy
from evaluation.algorithm.gnn_hierarchical_ppo import GnnHierarchicalPPOEvalPolicy
from evaluation.algorithm.hierarchical_ppo import HierarchicalPPOEvalPolicy
from evaluation.algorithm.pure_ppo import PurePPOEvalPolicy
from evaluation.algorithm.rule_pid import RulePIDEvalPolicy
from evaluation.algorithm.sac import SACEvalPolicy
from evaluation.online.stability_evaluator import StabilityEvaluator
from evaluation.online.stability_metrics import StabilityWeights
from perturbation.attack_scenarios import AttackScenario, make_default_scenarios, INTENSITY_LEVELS


class EvalPolicy(Protocol):
    def act(self, env: CarlaEnv) -> Tuple[float, float, float]: ...


AlgoBuilder = Callable[[], EvalPolicy]


@dataclass
class ScenarioEvalSummary:
    episodes: int

    # 主指标
    mean_stab: float
    mean_safety: float
    mean_lane: float
    mean_smooth: float
    mean_perf: float

    # 公认指标
    dsr: float  # Defense Success Rate
    collision_rate: float  # Collision Rate（CR）

    # 轨迹/控制偏差
    mean_mld: float  # Mean Lateral Deviation（MLD）= mean(|center_offset|)
    mean_mhd: float  # Mean Heading Deviation（MHD）= mean(|heading_diff|)
    mean_mad: float  # Mean Action Delta（MAD）= mean(|Δsteer|,|Δthr|,|Δbrk|) 的平均

    # 诊断指标
    mean_speed: float
    mean_speed_error: float  # mean(|v - v_des|)

    def to_row(self, scenario_key: str, desc: str) -> list:
        return [
            scenario_key,
            desc,
            self.episodes,
            f"{self.mean_stab:.6f}",
            f"{self.mean_safety:.6f}",
            f"{self.mean_lane:.6f}",
            f"{self.mean_smooth:.6f}",
            f"{self.mean_perf:.6f}",
            f"{self.dsr:.6f}",
            f"{self.collision_rate:.6f}",
            f"{self.mean_mld:.6f}",
            f"{self.mean_mhd:.6f}",
            f"{self.mean_mad:.6f}",
            f"{self.mean_speed:.6f}",
            f"{self.mean_speed_error:.6f}",
        ]


def _make_run_dir(base_out: str) -> Tuple[str, str]:
    """
    返回 (run_dir, run_id)，run_dir = base_out/YYYYmmdd_HHMMSS
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_out, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def _safe_get(info: dict, key: str, default=None):
    try:
        return info.get(key, default) if isinstance(info, dict) else default
    except Exception:
        return default


def _write_episode_row(
        writer: csv.writer,
        run_id: str,
        method: str,
        scenario_key: str,
        scenario_desc: str,
        seed: int,
        ep: int,
        metrics,  # EpisodeMetrics
        extra_info: Optional[dict] = None,
):
    """
    episodes.csv：每行一个 episode，存可复现定位信息 + 核心评价指标。
    """
    extra_info = extra_info or {}
    timestamp = datetime.now().isoformat(timespec="seconds")

    steps = int(getattr(metrics, "steps", 0) or 0)

    # episode 内均值
    mean_center = (metrics.sum_center_offset / steps) if steps > 0 else 0.0
    mean_heading = (metrics.sum_heading_diff_deg / steps) if steps > 0 else 0.0
    mean_speed = (metrics.sum_speed / steps) if steps > 0 else 0.0
    mean_speed_err = (metrics.sum_speed_error / steps) if steps > 0 else 0.0

    # 动作差分均值
    if steps > 1:
        denom = float(steps - 1)
        mean_d_steer = metrics.sum_delta_steer / denom
        mean_d_thr = metrics.sum_delta_throttle / denom
        mean_d_brk = metrics.sum_delta_brake / denom
        mean_mad = (mean_d_steer + mean_d_thr + mean_d_brk) / 3.0
    else:
        mean_d_steer = mean_d_thr = mean_d_brk = mean_mad = 0.0

    writer.writerow([
        run_id,
        timestamp,
        _safe_get(extra_info, "map", ""),
        method,
        scenario_key,
        scenario_desc,
        seed,
        ep,
        steps,
        getattr(metrics, "done_reason", "unknown"),
        int(bool(getattr(metrics, "collision", False))),
        int(bool(getattr(metrics, "offroad_terminate", False))),
        int(bool(getattr(metrics, "stuck_terminate", False))),
        int(bool(getattr(metrics, "success_flag", False))),

        f"{getattr(metrics, 'stability_score', 0.0):.6f}",
        f"{getattr(metrics, 'safety_score', 0.0):.6f}",
        f"{getattr(metrics, 'lane_score', 0.0):.6f}",
        f"{getattr(metrics, 'smooth_score', 0.0):.6f}",
        f"{getattr(metrics, 'perf_score', 0.0):.6f}",

        f"{mean_center:.6f}",
        f"{mean_heading:.6f}",
        f"{mean_speed:.6f}",
        f"{mean_speed_err:.6f}",

        f"{mean_d_steer:.6f}",
        f"{mean_d_thr:.6f}",
        f"{mean_d_brk:.6f}",
        f"{mean_mad:.6f}",
        int(getattr(config, "attack_intensity_level", -1)),
    ])


def build_algo_builders() -> Dict[str, AlgoBuilder]:
    return {
        "rule_pid": lambda: RulePIDEvalPolicy(),
        "bc": lambda: BCEvalPolicy("bc_residual_best.pt"),
        "gnn_hier_ppo": lambda: GnnHierarchicalPPOEvalPolicy("gnn_hierarchical_ppo_best.pt"),
        "gnn_flat_ppo": lambda: GnnFlatPPOEvalPolicy("gnn_flat_ppo_best.pt"),
        "hier_ppo": lambda: HierarchicalPPOEvalPolicy("hierarchical_ppo_best.pt"),
        "pure_ppo": lambda: PurePPOEvalPolicy("pure_ppo_best.pt"),
        "sac": lambda: SACEvalPolicy("sac_template_best.pt"),
    }


def eval_one_algo_one_scenario(
        env: CarlaEnv,
        algo_builder: AlgoBuilder,
        scenario: AttackScenario,
        episodes: int = 10,
        level: Optional[int] = None,

        episodes_writer: Optional[csv.writer] = None,
        run_id: str = "",
        method_name: str = "",
        scenario_key: str = "",
        scenario_desc: str = "",
        seed: int = 0,
        extra_env_info: Optional[dict] = None,

        close_mode: str = "per_scenario",  # "per_scenario" | "end_only"
) -> ScenarioEvalSummary:
    # 应用攻击配置
    scenario.apply(level=level)

    policy = algo_builder()

    # 稳定性评估器
    weights = StabilityWeights(
        w_safety=0.5,
        w_lane=0.3,
        w_smooth=0.2,
        w_perf=0.0,
    )
    stab_eval = StabilityEvaluator(weights)

    # 累积统计
    sum_stab = 0.0
    sum_safety = 0.0
    sum_lane = 0.0
    sum_smooth = 0.0
    sum_perf = 0.0

    collision_episodes = 0
    safe_episodes = 0

    sum_mld = 0.0
    sum_mhd = 0.0
    sum_mad = 0.0
    sum_mean_speed = 0.0
    sum_mean_speed_error = 0.0

    valid_mld = 0
    valid_mhd = 0
    valid_mad = 0
    valid_speed = 0
    valid_speed_error = 0

    for ep in range(episodes):
        env.reset()
        stab_eval.start_episode(max_steps=env.max_steps)

        done = False
        info = {}

        while not done:
            action = policy.act(env)
            raw_state, reward, done, info = env.step(action)
            stab_eval.step(info, action)

        metrics = stab_eval.end_episode(
            done_reason=info.get("done_reason", "unknown"),
            offroad_terminate=info.get("offroad_terminate", False),
            stuck_terminate=info.get("stuck_terminate", False),
        )

        if episodes_writer is not None:
            _write_episode_row(
                writer=episodes_writer,
                run_id=run_id,
                method=method_name,
                scenario_key=scenario_key,
                scenario_desc=scenario_desc,
                seed=seed,
                ep=ep,
                metrics=metrics,
                extra_info=extra_env_info,
            )

        # 主指标
        sum_stab += metrics.stability_score
        sum_safety += metrics.safety_score
        sum_lane += metrics.lane_score
        sum_smooth += metrics.smooth_score
        sum_perf += metrics.perf_score

        if metrics.collision:
            collision_episodes += 1

        if metrics.success_flag:
            safe_episodes += 1

        # 轨迹偏差
        if metrics.steps > 0:
            mld_ep = metrics.sum_center_offset / metrics.steps
            mhd_ep = metrics.sum_heading_diff_deg / metrics.steps
            mean_speed_ep = metrics.sum_speed / metrics.steps
            mean_speed_err_ep = metrics.sum_speed_error / metrics.steps

            sum_mld += mld_ep
            sum_mhd += mhd_ep
            sum_mean_speed += mean_speed_ep
            sum_mean_speed_error += mean_speed_err_ep

            valid_mld += 1
            valid_mhd += 1
            valid_speed += 1
            valid_speed_error += 1

        if metrics.steps > 1:
            denom = (metrics.steps - 1)
            mean_d_steer = metrics.sum_delta_steer / denom
            mean_d_thr = metrics.sum_delta_throttle / denom
            mean_d_brk = metrics.sum_delta_brake / denom
            mad_ep = (mean_d_steer + mean_d_thr + mean_d_brk) / 3.0

            sum_mad += mad_ep
            valid_mad += 1

    # 汇总
    E = max(1, episodes)

    mean_stab = sum_stab / E
    mean_safety = sum_safety / E
    mean_lane = sum_lane / E
    mean_smooth = sum_smooth / E
    mean_perf = sum_perf / E

    collision_rate = collision_episodes / float(E)
    dsr = safe_episodes / float(E)

    mean_mld = (sum_mld / valid_mld) if valid_mld > 0 else 0.0
    mean_mhd = (sum_mhd / valid_mhd) if valid_mhd > 0 else 0.0
    mean_mad = (sum_mad / valid_mad) if valid_mad > 0 else 0.0
    mean_speed = (sum_mean_speed / valid_speed) if valid_speed > 0 else 0.0
    mean_speed_error = (sum_mean_speed_error / valid_speed_error) if valid_speed_error > 0 else 0.0

    if close_mode == "per_scenario":
        env.close()

    return ScenarioEvalSummary(
        episodes=E,
        mean_stab=mean_stab,
        mean_safety=mean_safety,
        mean_lane=mean_lane,
        mean_smooth=mean_smooth,
        mean_perf=mean_perf,
        dsr=dsr,
        collision_rate=collision_rate,
        mean_mld=mean_mld,
        mean_mhd=mean_mhd,
        mean_mad=mean_mad,
        mean_speed=mean_speed,
        mean_speed_error=mean_speed_error,
    )


def run_all_and_save_csv(
        episodes: int = 1,
        base_out: str = "sim_eval_runs",
        seed: int = 0,
        close_mode: str = "per_scenario",
):
    algo_builders = build_algo_builders()
    scenarios = make_default_scenarios()

    scenario_keys = [
        "ego", "gnss", "tl_dist", "tl_color",
        "lane_has_next", "lane_yaw",
        "veh_noise", "veh_hide", "veh_fake", "lane_inv",
    ]

    env = CarlaEnv()

    extra_env_info = {}
    try:
        extra_env_info["map"] = getattr(env, "map_name", "") or getattr(env, "town", "")
    except Exception:
        pass

    run_root, run_id = _make_run_dir(base_out)

    for lvl in INTENSITY_LEVELS:
        level = lvl.level
        # if level < 2:
        #     continue
        out_dir = os.path.join(run_root, f"intensity_{lvl.name}")
        os.makedirs(out_dir, exist_ok=True)

        # 原始数据表
        episodes_csv = os.path.join(out_dir, "episodes.csv")

        # 拆分后的三张全局表
        global_stab_csv = os.path.join(out_dir, "global_stability.csv")
        global_dsr_csv = os.path.join(out_dir, "global_dsr.csv")
        global_cr_csv = os.path.join(out_dir, "global_cr.csv")

        total_algos = len(algo_builders)
        total_scen = 1 + len(scenario_keys)

        f_ep = open(episodes_csv, "w", newline="", encoding="utf-8")
        ep_writer = csv.writer(f_ep)
        ep_writer.writerow([
            "run_id", "timestamp", "map",
            "method", "scenario", "desc",
            "seed", "episode", "steps",
            "done_reason", "collision", "offroad_terminate", "stuck_terminate", "success_flag",
            "stability_score", "safety_score", "lane_score", "smooth_score", "perf_score",
            "mean_center_offset_m", "mean_heading_diff_deg", "mean_speed_mps", "mean_speed_error_mps",
            "mean_d_steer", "mean_d_throttle", "mean_d_brake", "mean_mad",
            "attack_intensity_level",
        ])

        # 全局表
        with open(global_stab_csv, "w", newline="", encoding="utf-8") as f_gs, \
                open(global_dsr_csv, "w", newline="", encoding="utf-8") as f_gd, \
                open(global_cr_csv, "w", newline="", encoding="utf-8") as f_gc:

            writer_gs = csv.writer(f_gs)
            writer_gd = csv.writer(f_gd)
            writer_gc = csv.writer(f_gc)

            # --- headers ---
            writer_gs.writerow(
                ["method", "clean_stab"] + [f"{k}_stab_ratio" for k in scenario_keys]
            )
            writer_gd.writerow(
                ["method", "clean_dsr"] + [f"{k}_dsr" for k in scenario_keys]
            )
            writer_gc.writerow(
                ["method", "clean_cr"] + [f"{k}_cr" for k in scenario_keys]
            )

            for algo_i, (algo_name, builder) in enumerate(algo_builders.items(), start=1):
                print(f"\n=== [{algo_i}/{total_algos}] Evaluating algo: {algo_name} ===", flush=True)

                per_algo_path = os.path.join(out_dir, f"{algo_name}_detail.csv")
                with open(per_algo_path, "w", newline="", encoding="utf-8") as f_algo:
                    writer_algo = csv.writer(f_algo)
                    writer_algo.writerow([
                        "scenario", "desc", "episodes",
                        "mean_stab", "mean_safety", "mean_lane", "mean_smooth", "mean_perf",
                        "dsr", "collision_rate",
                        "mean_mld", "mean_mhd", "mean_mad",
                        "mean_speed", "mean_speed_error"
                    ])

                    # clean
                    print(f"[{algo_name}] -> Scenario [1/{total_scen}]: clean (no attack) | episodes={episodes}",
                          flush=True)
                    clean_sum = eval_one_algo_one_scenario(
                        env=env,
                        algo_builder=builder,
                        scenario=scenarios["clean"],
                        episodes=episodes,
                        level=None,
                        episodes_writer=ep_writer,
                        run_id=run_id,
                        method_name=algo_name,
                        scenario_key="clean",
                        scenario_desc="no attack",
                        seed=seed,
                        extra_env_info=extra_env_info,
                        close_mode=close_mode,
                    )
                    f_ep.flush()
                    writer_algo.writerow(clean_sum.to_row("clean", "no attack"))
                    f_algo.flush()

                    # attacks
                    attack_summaries: Dict[str, ScenarioEvalSummary] = {}

                    for scen_i, key in enumerate(scenario_keys, start=2):
                        sc = scenarios[key]
                        print(
                            f"[{algo_name}] -> Scenario [{scen_i}/{total_scen}]: {key} | desc={sc.desc} | episodes={episodes}",
                            flush=True)

                        s = eval_one_algo_one_scenario(
                            env=env,
                            algo_builder=builder,
                            scenario=sc,
                            episodes=episodes,
                            level=level,
                            episodes_writer=ep_writer,
                            run_id=run_id,
                            method_name=algo_name,
                            scenario_key=key,
                            scenario_desc=sc.desc,
                            seed=seed,
                            extra_env_info=extra_env_info,
                            close_mode=close_mode,
                        )
                        f_ep.flush()

                        attack_summaries[key] = s
                        writer_algo.writerow(s.to_row(key, sc.desc))
                        f_algo.flush()

                # 三张 global 表各自的一行
                eps = max(1e-6, float(clean_sum.mean_stab))

                # stability: clean_stab + stab_ratio
                row_gs = [algo_name, f"{clean_sum.mean_stab:.6f}"]
                for k in scenario_keys:
                    s = attack_summaries[k]
                    row_gs.append(f"{(s.mean_stab / eps):.6f}")
                writer_gs.writerow(row_gs)
                f_gs.flush()

                # dsr: clean_dsr + per-scenario dsr
                row_gd = [algo_name, f"{clean_sum.dsr:.6f}"]
                for k in scenario_keys:
                    row_gd.append(f"{attack_summaries[k].dsr:.6f}")
                writer_gd.writerow(row_gd)
                f_gd.flush()

                # cr: clean_cr + per-scenario cr
                row_gc = [algo_name, f"{clean_sum.collision_rate:.6f}"]
                for k in scenario_keys:
                    row_gc.append(f"{attack_summaries[k].collision_rate:.6f}")
                writer_gc.writerow(row_gc)
                f_gc.flush()

                print(f"[{algo_name}] written: {per_algo_path}", flush=True)

        print(
            f"\n评估完成：\n"
            f"- run_dir: {run_root}\n"
            f"- global_stability: {global_stab_csv}\n"
            f"- global_dsr: {global_dsr_csv}\n"
            f"- global_cr: {global_cr_csv}\n"
            f"- episodes: {episodes_csv}\n"
            f"- per-algo: {run_root}/*_detail.csv",
            flush=True
        )

        f_ep.close()

    if close_mode == "end_only":
        env.close()


if __name__ == '__main__':
    run_all_and_save_csv(close_mode="end_only")
