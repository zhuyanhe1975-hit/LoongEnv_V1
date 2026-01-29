#!/usr/bin/env python3
"""
基本使用示例

演示机器人运动控制系统的基本使用方法，包括模型创建、轨迹规划和控制执行。
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_motion_control import (
    RobotModel, RobotMotionController, 
    DynamicsParameters, KinodynamicLimits
)
from robot_motion_control.core.types import (
    RobotState, Waypoint
)
from robot_motion_control.core.controller import ControllerConfig
from robot_motion_control.simulation import SimulationEnvironment


def create_sample_robot():
    """创建示例机器人模型"""
    n_joints = 6
    
    # 动力学参数
    dynamics_params = DynamicsParameters(
        masses=[15.0, 12.0, 8.0, 5.0, 3.0, 1.5],
        centers_of_mass=[
            [0.0, 0.0, 0.15],   # 基座连杆质心
            [0.0, 0.0, 0.2],    # 大臂质心
            [0.0, 0.0, 0.15],   # 小臂质心
            [0.0, 0.0, 0.1],    # 腕部1质心
            [0.0, 0.0, 0.05],   # 腕部2质心
            [0.0, 0.0, 0.03]    # 腕部3质心
        ],
        inertias=[
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.5]],
            [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.3]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.1]],
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.05]]
        ],
        friction_coeffs=[0.15, 0.12, 0.10, 0.08, 0.06, 0.05]
    )
    
    # 运动学限制
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[np.pi, np.pi/2, np.pi, np.pi, np.pi, np.pi],
        min_joint_positions=[-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi],
        max_joint_velocities=[3.14, 2.5, 3.14, 4.0, 4.0, 6.0],
        max_joint_accelerations=[15.0, 12.0, 15.0, 20.0, 20.0, 30.0],
        max_joint_jerks=[100.0, 80.0, 100.0, 150.0, 150.0, 200.0],
        max_joint_torques=[200.0, 180.0, 120.0, 80.0, 50.0, 30.0]
    )
    
    return RobotModel(
        name="ER15-1400",
        n_joints=n_joints,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def demonstrate_trajectory_planning():
    """演示轨迹规划功能"""
    print("=== 轨迹规划演示 ===")
    
    # 创建机器人模型
    robot = create_sample_robot()
    print(f"创建机器人模型: {robot.name}, {robot.n_joints}轴")
    
    # 创建控制器
    controller = RobotMotionController(robot)
    
    # 定义路径点
    waypoints = [
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0])),
        Waypoint(position=np.array([1.0, 0.5, 0.4, 0.2, 0.1, 0.0])),
        Waypoint(position=np.array([0.8, 0.2, 0.1, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ]
    
    print(f"定义了 {len(waypoints)} 个路径点")
    
    # 规划轨迹
    print("正在规划轨迹...")
    trajectory = controller.plan_trajectory(waypoints, optimize_time=True)
    
    print(f"轨迹规划完成，生成了 {len(trajectory)} 个轨迹点")
    print(f"轨迹总时间: {trajectory[-1].time:.2f} 秒")
    
    return robot, controller, trajectory


def demonstrate_control_execution():
    """演示控制执行功能"""
    print("\n=== 控制执行演示 ===")
    
    robot, controller, trajectory = demonstrate_trajectory_planning()
    
    # 创建仿真环境
    sim_env = SimulationEnvironment(robot)
    
    # 初始化仿真
    initial_state = RobotState(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        joint_accelerations=np.zeros(6),
        joint_torques=np.zeros(6),
        end_effector_pose=np.eye(4),
        timestamp=0.0
    )
    
    sim_env.initialize(initial_state)
    print("仿真环境初始化完成")
    
    # 运行轨迹仿真
    print("正在执行轨迹仿真...")
    result = sim_env.run_trajectory_simulation(trajectory, max_simulation_time=30.0)
    
    if result.success:
        print(f"仿真执行成功!")
        print(f"执行时间: {result.execution_time:.2f} 秒")
        print(f"平均跟踪误差: {result.performance_metrics.tracking_error:.6f} m")
        print(f"振动幅度: {result.performance_metrics.vibration_amplitude:.6f} m")
        print(f"成功率: {result.performance_metrics.success_rate:.2%}")
    else:
        print(f"仿真执行失败: {result.error_message}")
    
    return result


def demonstrate_dynamics_computation():
    """演示动力学计算功能"""
    print("\n=== 动力学计算演示 ===")
    
    robot = create_sample_robot()
    
    # 测试关节配置
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    qd = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    qdd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    print(f"测试关节位置: {q}")
    print(f"测试关节速度: {qd}")
    print(f"测试关节加速度: {qdd}")
    
    # 创建动力学引擎
    from robot_motion_control.algorithms import DynamicsEngine
    dynamics = DynamicsEngine(robot)
    
    # 逆动力学计算
    tau = dynamics.inverse_dynamics(q, qd, qdd)
    print(f"逆动力学计算结果 (力矩): {tau}")
    
    # 正向动力学计算
    qdd_computed = dynamics.forward_dynamics(q, qd, tau)
    print(f"正向动力学计算结果 (加速度): {qdd_computed}")
    
    # 验证一致性
    error = np.linalg.norm(qdd - qdd_computed)
    print(f"动力学一致性误差: {error:.8f}")
    
    # 重力补偿
    g = dynamics.gravity_compensation(q)
    print(f"重力补偿力矩: {g}")
    
    # 雅可比矩阵
    J = dynamics.jacobian(q)
    print(f"雅可比矩阵形状: {J.shape}")
    print(f"雅可比矩阵条件数: {np.linalg.cond(J):.2f}")


def plot_trajectory_results(result):
    """绘制轨迹结果"""
    if not result.success or not result.trajectory_executed:
        print("没有可绘制的轨迹数据")
        return
    
    print("\n=== 绘制轨迹结果 ===")
    
    # 提取数据
    times = [state.timestamp for state in result.trajectory_executed]
    positions = np.array([state.joint_positions for state in result.trajectory_executed])
    velocities = np.array([state.joint_velocities for state in result.trajectory_executed])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制关节位置
    for i in range(positions.shape[1]):
        ax1.plot(times, positions[:, i], label=f'关节 {i+1}')
    
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('关节位置 (rad)')
    ax1.set_title('关节位置轨迹')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制关节速度
    for i in range(velocities.shape[1]):
        ax2.plot(times, velocities[:, i], label=f'关节 {i+1}')
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('关节速度 (rad/s)')
    ax2.set_title('关节速度轨迹')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_results.png', dpi=150, bbox_inches='tight')
    print("轨迹结果已保存到 trajectory_results.png")
    
    # 显示图形（如果在交互环境中）
    try:
        plt.show()
    except:
        pass


def main():
    """主函数"""
    print("机器人运动控制系统演示")
    print("=" * 50)
    
    try:
        # 演示动力学计算
        demonstrate_dynamics_computation()
        
        # 演示控制执行
        result = demonstrate_control_execution()
        
        # 绘制结果
        plot_trajectory_results(result)
        
        print("\n演示完成!")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()