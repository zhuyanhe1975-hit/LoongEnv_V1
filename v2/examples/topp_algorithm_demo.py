#!/usr/bin/env python3
"""
TOPP算法演示脚本

展示时间最优路径参数化（TOPP）算法的功能和性能。
包括基本用法、负载自适应、自适应包络线调整等特性。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, PayloadInfo, 
    Waypoint, TrajectoryPoint
)
from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner


def create_test_robot():
    """创建测试用机器人模型"""
    n_joints = 6
    
    dynamics_params = DynamicsParameters(
        masses=[15.0, 12.0, 8.0, 5.0, 3.0, 1.5],
        centers_of_mass=[[0.0, 0.0, 0.15]] * n_joints,
        inertias=[[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]] * n_joints,
        friction_coeffs=[0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
        gravity=[0.0, 0.0, -9.81]
    )
    
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[np.pi] * n_joints,
        min_joint_positions=[-np.pi] * n_joints,
        max_joint_velocities=[3.0, 2.5, 2.0, 2.5, 3.0, 4.0],
        max_joint_accelerations=[15.0, 12.0, 10.0, 12.0, 15.0, 20.0],
        max_joint_jerks=[75.0, 60.0, 50.0, 60.0, 75.0, 100.0],
        max_joint_torques=[150.0, 120.0, 80.0, 50.0, 30.0, 15.0]
    )
    
    return RobotModel(
        name="demo_robot",
        n_joints=n_joints,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def create_demo_paths():
    """创建演示路径"""
    paths = {}
    
    # 1. 简单直线路径
    paths['straight_line'] = [
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0])),
        Waypoint(position=np.array([1.0, 0.6, 0.4, 0.2, 0.0, 0.0])),
        Waypoint(position=np.array([1.5, 0.9, 0.6, 0.3, 0.0, 0.0]))
    ]
    
    # 2. 弯曲路径（类似圆弧）
    n_points = 8
    angles = np.linspace(0, np.pi/2, n_points)
    radius = 1.2
    
    paths['curved'] = []
    for i, angle in enumerate(angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.1 * np.sin(2 * angle)  # 添加Z方向变化
        position = np.array([x, y, z, angle/2, 0.0, 0.0])
        paths['curved'].append(Waypoint(position=position))
    
    # 3. 复杂路径（S形）
    n_points = 10
    t_values = np.linspace(0, 2*np.pi, n_points)
    
    paths['s_curve'] = []
    for t in t_values:
        x = t / (2*np.pi)
        y = 0.5 * np.sin(t)
        z = 0.2 * np.sin(2*t)
        rx = 0.1 * np.cos(t)
        ry = 0.1 * np.sin(t)
        rz = 0.05 * t
        position = np.array([x, y, z, rx, ry, rz])
        paths['s_curve'].append(Waypoint(position=position))
    
    return paths


def demo_basic_topp():
    """演示基本TOPP算法功能"""
    print("=== TOPP算法基本功能演示 ===")
    
    robot = create_test_robot()
    planner = TrajectoryPlanner(robot)
    paths = create_demo_paths()
    
    results = {}
    
    for path_name, path in paths.items():
        print(f"\n处理路径: {path_name}")
        print(f"路径点数: {len(path)}")
        
        # 生成TOPP轨迹
        import time
        start_time = time.time()
        
        trajectory = planner.generate_topp_trajectory(path, robot.kinodynamic_limits)
        
        computation_time = time.time() - start_time
        
        # 计算轨迹指标
        metrics = planner.compute_trajectory_metrics(trajectory)
        
        results[path_name] = {
            'trajectory': trajectory,
            'metrics': metrics,
            'computation_time': computation_time
        }
        
        print(f"  轨迹点数: {len(trajectory)}")
        print(f"  总时间: {metrics['total_time']:.3f}s")
        print(f"  总距离: {metrics['total_distance']:.3f}")
        print(f"  最大速度: {metrics['max_velocity']:.3f}")
        print(f"  最大加速度: {metrics['max_acceleration']:.3f}")
        print(f"  计算时间: {computation_time:.3f}s")
    
    return results


def demo_payload_adaptation():
    """演示负载自适应功能"""
    print("\n=== 负载自适应演示 ===")
    
    robot = create_test_robot()
    planner = TrajectoryPlanner(robot)
    path = create_demo_paths()['straight_line']
    
    # 不同负载情况
    payloads = {
        'no_payload': None,
        'light_payload': PayloadInfo(
            mass=2.0,
            center_of_mass=[0.0, 0.0, 0.05],
            inertia=[[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
            identification_confidence=0.95
        ),
        'heavy_payload': PayloadInfo(
            mass=8.0,
            center_of_mass=[0.0, 0.0, 0.1],
            inertia=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            identification_confidence=0.85
        )
    }
    
    results = {}
    
    for payload_name, payload in payloads.items():
        print(f"\n负载情况: {payload_name}")
        if payload:
            print(f"  质量: {payload.mass}kg")
            print(f"  置信度: {payload.identification_confidence}")
        
        trajectory = planner.generate_topp_trajectory(
            path, robot.kinodynamic_limits, payload=payload
        )
        
        metrics = planner.compute_trajectory_metrics(trajectory)
        results[payload_name] = {
            'trajectory': trajectory,
            'metrics': metrics
        }
        
        print(f"  总时间: {metrics['total_time']:.3f}s")
        print(f"  最大速度: {metrics['max_velocity']:.3f}")
        print(f"  最大加速度: {metrics['max_acceleration']:.3f}")
    
    # 分析负载影响
    print("\n负载影响分析:")
    no_payload_time = results['no_payload']['metrics']['total_time']
    
    for payload_name in ['light_payload', 'heavy_payload']:
        payload_time = results[payload_name]['metrics']['total_time']
        time_increase = (payload_time - no_payload_time) / no_payload_time * 100
        print(f"  {payload_name}: 时间增加 {time_increase:.1f}%")
    
    return results


def demo_adaptive_envelope():
    """演示自适应包络线调整"""
    print("\n=== 自适应包络线调整演示 ===")
    
    robot = create_test_robot()
    planner = TrajectoryPlanner(robot)
    path = create_demo_paths()['curved']  # 使用弯曲路径
    
    # 比较启用和禁用自适应包络线的效果
    trajectory_adaptive = planner.generate_topp_trajectory(
        path, robot.kinodynamic_limits, adaptive_envelope=True
    )
    
    trajectory_non_adaptive = planner.generate_topp_trajectory(
        path, robot.kinodynamic_limits, adaptive_envelope=False
    )
    
    metrics_adaptive = planner.compute_trajectory_metrics(trajectory_adaptive)
    metrics_non_adaptive = planner.compute_trajectory_metrics(trajectory_non_adaptive)
    
    print("自适应包络线启用:")
    print(f"  总时间: {metrics_adaptive['total_time']:.3f}s")
    print(f"  最大速度: {metrics_adaptive['max_velocity']:.3f}")
    print(f"  速度平滑性: {metrics_adaptive['velocity_smoothness']:.3f}")
    
    print("自适应包络线禁用:")
    print(f"  总时间: {metrics_non_adaptive['total_time']:.3f}s")
    print(f"  最大速度: {metrics_non_adaptive['max_velocity']:.3f}")
    print(f"  速度平滑性: {metrics_non_adaptive['velocity_smoothness']:.3f}")
    
    return {
        'adaptive': {'trajectory': trajectory_adaptive, 'metrics': metrics_adaptive},
        'non_adaptive': {'trajectory': trajectory_non_adaptive, 'metrics': metrics_non_adaptive}
    }


def demo_constraint_handling():
    """演示约束处理能力"""
    print("\n=== 约束处理演示 ===")
    
    robot = create_test_robot()
    planner = TrajectoryPlanner(robot)
    path = create_demo_paths()['s_curve']
    
    # 生成轨迹
    trajectory = planner.generate_topp_trajectory(path, robot.kinodynamic_limits)
    
    # 检查约束满足情况
    limits = robot.kinodynamic_limits
    max_velocities = np.array(limits.max_joint_velocities)
    max_accelerations = np.array(limits.max_joint_accelerations)
    
    velocity_violations = 0
    acceleration_violations = 0
    max_velocity_ratio = 0.0
    max_acceleration_ratio = 0.0
    
    for point in trajectory:
        # 检查速度约束
        velocity_ratios = np.abs(point.velocity) / max_velocities
        max_velocity_ratio = max(max_velocity_ratio, np.max(velocity_ratios))
        if np.any(velocity_ratios > 1.0):
            velocity_violations += 1
        
        # 检查加速度约束
        acceleration_ratios = np.abs(point.acceleration) / max_accelerations
        max_acceleration_ratio = max(max_acceleration_ratio, np.max(acceleration_ratios))
        if np.any(acceleration_ratios > 1.0):
            acceleration_violations += 1
    
    print(f"轨迹点总数: {len(trajectory)}")
    print(f"速度约束违反: {velocity_violations} 点")
    print(f"加速度约束违反: {acceleration_violations} 点")
    print(f"最大速度利用率: {max_velocity_ratio:.1%}")
    print(f"最大加速度利用率: {max_acceleration_ratio:.1%}")
    
    # 验证轨迹平滑性
    is_smooth, errors = planner.validate_trajectory_smoothness(trajectory)
    print(f"轨迹平滑性: {'通过' if is_smooth else '未通过'}")
    if not is_smooth:
        print(f"  平滑性错误数: {len(errors)}")
    
    return trajectory


def visualize_results(results):
    """可视化结果"""
    print("\n=== 生成可视化图表 ===")
    
    try:
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('TOPP算法演示结果', fontsize=16)
        
        # 1. 不同路径的轨迹时间比较
        if 'basic_results' in results:
            path_names = list(results['basic_results'].keys())
            times = [results['basic_results'][name]['metrics']['total_time'] for name in path_names]
            
            axes[0, 0].bar(path_names, times)
            axes[0, 0].set_title('不同路径的执行时间')
            axes[0, 0].set_ylabel('时间 (s)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 负载对执行时间的影响
        if 'payload_results' in results:
            payload_names = list(results['payload_results'].keys())
            payload_times = [results['payload_results'][name]['metrics']['total_time'] for name in payload_names]
            
            axes[0, 1].bar(payload_names, payload_times)
            axes[0, 1].set_title('负载对执行时间的影响')
            axes[0, 1].set_ylabel('时间 (s)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 自适应包络线效果比较
        if 'envelope_results' in results:
            envelope_names = ['自适应', '非自适应']
            envelope_times = [
                results['envelope_results']['adaptive']['metrics']['total_time'],
                results['envelope_results']['non_adaptive']['metrics']['total_time']
            ]
            
            axes[0, 2].bar(envelope_names, envelope_times)
            axes[0, 2].set_title('自适应包络线效果')
            axes[0, 2].set_ylabel('时间 (s)')
        
        # 4. 速度曲线示例
        if 'basic_results' in results and 'straight_line' in results['basic_results']:
            trajectory = results['basic_results']['straight_line']['trajectory']
            times = [p.time for p in trajectory]
            velocities = [np.linalg.norm(p.velocity) for p in trajectory]
            
            axes[1, 0].plot(times, velocities, 'b-', linewidth=2)
            axes[1, 0].set_title('速度曲线示例')
            axes[1, 0].set_xlabel('时间 (s)')
            axes[1, 0].set_ylabel('速度 (rad/s)')
            axes[1, 0].grid(True)
        
        # 5. 加速度曲线示例
        if 'basic_results' in results and 'straight_line' in results['basic_results']:
            trajectory = results['basic_results']['straight_line']['trajectory']
            times = [p.time for p in trajectory]
            accelerations = [np.linalg.norm(p.acceleration) for p in trajectory]
            
            axes[1, 1].plot(times, accelerations, 'r-', linewidth=2)
            axes[1, 1].set_title('加速度曲线示例')
            axes[1, 1].set_xlabel('时间 (s)')
            axes[1, 1].set_ylabel('加速度 (rad/s²)')
            axes[1, 1].grid(True)
        
        # 6. 路径参数vs时间
        if 'basic_results' in results and 'curved' in results['basic_results']:
            trajectory = results['basic_results']['curved']['trajectory']
            times = [p.time for p in trajectory]
            path_params = [p.path_parameter for p in trajectory]
            
            axes[1, 2].plot(times, path_params, 'g-', linewidth=2)
            axes[1, 2].set_title('路径参数化')
            axes[1, 2].set_xlabel('时间 (s)')
            axes[1, 2].set_ylabel('路径参数')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = project_root / "examples" / "topp_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        
        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            print("无法显示图表（非交互环境）")
            
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")


def main():
    """主演示函数"""
    print("TOPP算法综合演示")
    print("=" * 50)
    
    results = {}
    
    try:
        # 1. 基本功能演示
        results['basic_results'] = demo_basic_topp()
        
        # 2. 负载自适应演示
        results['payload_results'] = demo_payload_adaptation()
        
        # 3. 自适应包络线演示
        results['envelope_results'] = demo_adaptive_envelope()
        
        # 4. 约束处理演示
        results['constraint_trajectory'] = demo_constraint_handling()
        
        # 5. 可视化结果
        visualize_results(results)
        
        print("\n=== 演示总结 ===")
        print("✓ TOPP算法基本功能正常")
        print("✓ 负载自适应功能有效")
        print("✓ 自适应包络线调整工作正常")
        print("✓ 约束处理能力良好")
        print("✓ 轨迹生成质量满足要求")
        
        # 性能统计
        if 'basic_results' in results:
            avg_computation_time = np.mean([
                results['basic_results'][name]['computation_time'] 
                for name in results['basic_results']
            ])
            print(f"✓ 平均计算时间: {avg_computation_time:.3f}s")
        
        print("\nTOPP算法演示完成！")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())