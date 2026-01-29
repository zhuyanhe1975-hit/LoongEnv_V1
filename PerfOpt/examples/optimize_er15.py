#!/usr/bin/env python3
"""
ER15-1400机器人参数优化示例

演示如何使用PerfOpt进行多目标参数优化
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# 添加perfopt到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from perfopt import ParameterOptimizer, RobotModel, TrajectoryPoint


def create_test_trajectory(n_joints=6, duration=2.0, dt=0.01):
    """
    创建测试轨迹
    
    Args:
        n_joints: 关节数量
        duration: 轨迹持续时间
        dt: 时间步长
    
    Returns:
        轨迹点列表
    """
    trajectory = []
    n_points = int(duration / dt)
    
    for i in range(n_points):
        t = i * dt
        
        # 多频率正弦轨迹
        position = np.array([
            0.8 * np.sin(2 * np.pi * 0.5 * t),
            0.6 * np.cos(2 * np.pi * 0.3 * t),
            0.4 * np.sin(2 * np.pi * 0.7 * t),
            0.3 * np.cos(2 * np.pi * 0.4 * t),
            0.2 * np.sin(2 * np.pi * 0.6 * t),
            0.1 * np.cos(2 * np.pi * 0.8 * t),
        ])
        
        velocity = np.array([
            0.8 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t),
            -0.6 * 2 * np.pi * 0.3 * np.sin(2 * np.pi * 0.3 * t),
            0.4 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t),
            -0.3 * 2 * np.pi * 0.4 * np.sin(2 * np.pi * 0.4 * t),
            0.2 * 2 * np.pi * 0.6 * np.cos(2 * np.pi * 0.6 * t),
            -0.1 * 2 * np.pi * 0.8 * np.sin(2 * np.pi * 0.8 * t),
        ])
        
        acceleration = np.array([
            -0.8 * (2 * np.pi * 0.5)**2 * np.sin(2 * np.pi * 0.5 * t),
            -0.6 * (2 * np.pi * 0.3)**2 * np.cos(2 * np.pi * 0.3 * t),
            -0.4 * (2 * np.pi * 0.7)**2 * np.sin(2 * np.pi * 0.7 * t),
            -0.3 * (2 * np.pi * 0.4)**2 * np.cos(2 * np.pi * 0.4 * t),
            -0.2 * (2 * np.pi * 0.6)**2 * np.sin(2 * np.pi * 0.6 * t),
            -0.1 * (2 * np.pi * 0.8)**2 * np.cos(2 * np.pi * 0.8 * t),
        ])
        
        point = TrajectoryPoint(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=np.zeros(n_joints),
            time=t,
            path_parameter=i / (n_points - 1)
        )
        trajectory.append(point)
    
    return trajectory


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ER15-1400参数优化')
    parser.add_argument('--iterations', type=int, default=50,
                        help='优化迭代次数 (默认: 50)')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'basin_hopping', 'gradient_descent'],
                        help='优化算法 (默认: differential_evolution)')
    parser.add_argument('--population', type=int, default=10,
                        help='种群大小 (默认: 10)')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ER15-1400 机器人参数优化")
    print("=" * 60)
    print(f"优化算法: {args.method}")
    print(f"迭代次数: {args.iterations}")
    print(f"种群大小: {args.population}")
    print("=" * 60)
    print()
    
    # 1. 创建机器人模型
    print("[1/4] 创建机器人模型...")
    robot = RobotModel.create_test_model(n_joints=6)
    print(f"✓ 机器人模型: {robot.name}, {robot.n_joints}个关节")
    print()
    
    # 2. 创建测试轨迹
    print("[2/4] 生成测试轨迹...")
    trajectory = create_test_trajectory(n_joints=6, duration=2.0)
    print(f"✓ 轨迹生成完成: {len(trajectory)}个点, 持续时间{trajectory[-1].time:.2f}秒")
    print()
    
    # 3. 创建优化器
    print("[3/4] 初始化参数优化器...")
    optimizer = ParameterOptimizer(
        robot_model=robot,
        max_iterations=args.iterations,
        population_size=args.population,
        method=args.method,
        verbose=args.verbose
    )
    print("✓ 优化器初始化完成")
    print()
    
    # 4. 运行优化
    print("[4/4] 开始参数优化...")
    print("-" * 60)
    
    results = optimizer.optimize_control_gains(
        reference_trajectory=trajectory
    )
    
    print("-" * 60)
    print()
    
    # 5. 显示结果
    print("=" * 60)
    print("优化结果")
    print("=" * 60)
    print(f"优化成功: {'是' if results.success else '否'}")
    print(f"最优性能: {results.best_performance:.6f}")
    print(f"计算时间: {results.computation_time:.2f}秒")
    print(f"迭代次数: {len(results.optimization_history)}")
    print()
    
    print("最优参数:")
    for param_name, param_value in results.optimal_parameters.items():
        if isinstance(param_value, np.ndarray):
            print(f"  {param_name}: {param_value}")
        else:
            print(f"  {param_name}: {param_value}")
    print()
    
    print("性能改进:")
    if len(results.optimization_history) > 1:
        initial_perf = results.optimization_history[0]
        final_perf = results.best_performance
        improvement = (initial_perf - final_perf) / initial_perf * 100
        print(f"  初始性能: {initial_perf:.6f}")
        print(f"  最终性能: {final_perf:.6f}")
        print(f"  改进幅度: {improvement:.2f}%")
    print()
    
    print("=" * 60)
    print("✓ 优化完成！")
    print("=" * 60)
    print()
    print("报告已保存到 reports/ 目录")
    print()


if __name__ == "__main__":
    main()
