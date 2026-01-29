#!/usr/bin/env python3
"""
轨迹插补算法演示

演示七段式S型速度曲线生成和轨迹平滑性验证功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner
from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, Waypoint
)


def create_demo_robot():
    """创建演示用机器人模型"""
    n_joints = 6
    
    dynamics_params = DynamicsParameters(
        masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
        centers_of_mass=[[0.0, 0.0, 0.1]] * n_joints,
        inertias=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * n_joints,
        friction_coeffs=[0.1] * n_joints
    )
    
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[np.pi] * n_joints,
        min_joint_positions=[-np.pi] * n_joints,
        max_joint_velocities=[2.0] * n_joints,
        max_joint_accelerations=[10.0] * n_joints,
        max_joint_jerks=[50.0] * n_joints,
        max_joint_torques=[100.0] * n_joints
    )
    
    return RobotModel(
        name="demo_robot",
        n_joints=n_joints,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def demo_s7_interpolation():
    """演示S7插补算法"""
    print("=== 七段式S型轨迹插补演示 ===")
    
    # 创建机器人模型和轨迹规划器
    robot = create_demo_robot()
    planner = TrajectoryPlanner(robot)
    
    # 创建简单的两点路径
    start = Waypoint(position=np.zeros(6))
    end = Waypoint(position=np.ones(6))
    path = [start, end]
    
    print(f"起始位置: {start.position}")
    print(f"结束位置: {end.position}")
    
    # 生成S型轨迹
    trajectory = planner.interpolate_s7_trajectory(
        path,
        max_velocity=1.0,
        max_acceleration=2.0,
        max_jerk=5.0
    )
    
    print(f"生成轨迹点数: {len(trajectory)}")
    print(f"轨迹总时间: {trajectory[-1].time:.3f}s")
    
    # 验证轨迹平滑性
    is_smooth, errors = planner.validate_trajectory_smoothness(
        trajectory,
        position_tolerance=0.1,
        velocity_tolerance=0.5,
        acceleration_tolerance=2.0
    )
    
    print(f"轨迹平滑性: {'通过' if is_smooth else '未通过'}")
    if not is_smooth:
        print(f"发现 {len(errors)} 个不平滑点")
        if len(errors) <= 5:
            for error in errors[:5]:
                print(f"  - {error}")
    
    # 计算轨迹指标
    metrics = planner.compute_trajectory_metrics(trajectory)
    print("\n轨迹质量指标:")
    print(f"  总时间: {metrics['total_time']:.3f}s")
    print(f"  总距离: {metrics['total_distance']:.3f}")
    print(f"  最大速度: {metrics['max_velocity']:.3f}")
    print(f"  最大加速度: {metrics['max_acceleration']:.3f}")
    print(f"  最大加加速度: {metrics['max_jerk']:.3f}")
    
    return trajectory


def demo_multi_point_interpolation():
    """演示多点轨迹插补"""
    print("\n=== 多点轨迹插补演示 ===")
    
    robot = create_demo_robot()
    planner = TrajectoryPlanner(robot)
    
    # 创建多点路径
    waypoints = [
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])),
        Waypoint(position=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        Waypoint(position=np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5]))
    ]
    
    print(f"路径点数: {len(waypoints)}")
    
    # 生成轨迹
    trajectory = planner.interpolate_s7_trajectory(waypoints)
    
    print(f"生成轨迹点数: {len(trajectory)}")
    print(f"轨迹总时间: {trajectory[-1].time:.3f}s")
    
    # 验证边界条件
    start_match = np.allclose(trajectory[0].position, waypoints[0].position, atol=1e-2)
    end_match = np.allclose(trajectory[-1].position, waypoints[-1].position, atol=1e-2)
    
    print(f"起始位置匹配: {'是' if start_match else '否'}")
    print(f"结束位置匹配: {'是' if end_match else '否'}")
    
    return trajectory


def plot_trajectory(trajectory, title="轨迹曲线"):
    """绘制轨迹曲线"""
    try:
        times = [p.time for p in trajectory]
        positions = np.array([p.position for p in trajectory])
        velocities = np.array([p.velocity for p in trajectory])
        accelerations = np.array([p.acceleration for p in trajectory])
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # 位置曲线
        axes[0].plot(times, positions[:, 0], label='Joint 1')
        axes[0].set_ylabel('Position (rad)')
        axes[0].set_title(f'{title} - 位置')
        axes[0].grid(True)
        axes[0].legend()
        
        # 速度曲线
        axes[1].plot(times, velocities[:, 0], label='Joint 1')
        axes[1].set_ylabel('Velocity (rad/s)')
        axes[1].set_title(f'{title} - 速度')
        axes[1].grid(True)
        axes[1].legend()
        
        # 加速度曲线
        axes[2].plot(times, accelerations[:, 0], label='Joint 1')
        axes[2].set_ylabel('Acceleration (rad/s²)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title(f'{title} - 加速度')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib未安装，跳过绘图")


def main():
    """主函数"""
    print("轨迹插补算法演示程序")
    print("=" * 50)
    
    # 演示S7插补
    trajectory1 = demo_s7_interpolation()
    
    # 演示多点插补
    trajectory2 = demo_multi_point_interpolation()
    
    # 绘制轨迹（如果matplotlib可用）
    plot_trajectory(trajectory1, "S7两点插补")
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()