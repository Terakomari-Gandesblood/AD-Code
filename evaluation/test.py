import random
from typing import Tuple

from environment.carla_env import CarlaEnv
from evaluation.online.stability_metrics import StabilityWeights
from evaluation.online.stability_evaluator import StabilityEvaluator


def act() -> Tuple[float, float, float]:
    # 简单随机控制：[-1,1]转向， [0,1]油门，[0,0.3]刹车
    steer = random.uniform(-1.0, 1.0)
    throttle = random.uniform(0.0, 1.0)
    brake = random.uniform(0.0, 0.3)
    return steer, throttle, brake


if __name__ == "__main__":
    # 创建环境和 Agent
    env = CarlaEnv()

    # 创建稳定性评估器
    weights = StabilityWeights(
        w_safety=0.4,
        w_lane=0.2,
        w_smooth=0.2,
        w_perf=0.2,
    )
    evaluator = StabilityEvaluator(weights)

    NUM_EPISODES = 3

    for ep in range(NUM_EPISODES):
        print(f"\n===== Episode {ep} =====")

        # 重置环境
        obs = env.reset()

        # 告诉评估器本局的最大步数
        evaluator.start_episode(max_steps=env.max_steps)

        done = False
        step = 0
        info = None

        while not done:
            # 用策略选动作
            action = act()

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 喂给评估器做统计
            evaluator.step(info, action)

            step += 1

            # 打印一下调试信息
            if step % 100 == 0:
                print(f"[Ep {ep} | step {step}] reward={reward:.3f}, "
                      f"speed={info.get('speed_mps', 0.0):.2f}, "
                      f"collision={info.get('collision', False)}")

        # 2.4 Episode 结束，生成稳定性指标
        metrics = evaluator.end_episode(
            done_reason=info.get("done_reason", "unknown"),
            offroad_terminate=info.get("offroad_terminate", False),
            stuck_terminate=info.get("stuck_terminate", False),
        )

        # 3) 打印本局的评分结果
        print(f"Episode {ep} done. reason = {metrics.done_reason}")
        print(f"  steps            = {metrics.steps}/{metrics.max_steps}")
        print(f"  collision        = {metrics.collision}")
        print(f"  offroad_terminate= {metrics.offroad_terminate}")
        print(f"  stuck_terminate  = {metrics.stuck_terminate}")
        print(f"  tl_violations    = {metrics.tl_violation_count}")
        print(f"  mean_center_off  = {metrics.sum_center_offset / max(1, metrics.steps):.3f} m")
        print(f"  mean_heading_diff= {metrics.sum_heading_diff_deg / max(1, metrics.steps):.2f} deg")
        print(f"  mean_speed       = {metrics.sum_speed / max(1, metrics.steps):.2f} m/s")
        print(f"  safety_score     = {metrics.safety_score:.3f}")
        print(f"  lane_score       = {metrics.lane_score:.3f}")
        print(f"  smooth_score     = {metrics.smooth_score:.3f}")
        print(f"  perf_score       = {metrics.perf_score:.3f}")
        print(f"  stability_score  = {metrics.stability_score:.3f}")
        print(f"  success_flag     = {metrics.success_flag}")

    print("\n测试完成")
