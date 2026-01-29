#!/usr/bin/env python3
"""
高精度路径跟踪控制器演示

演示PathController的各种控制模式和功能：
- PID控制
- 计算力矩控制
- 滑模控制
- 自适应控制
- 前馈控制
- 性能监控
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, RobotState, 
    TrajectoryPoint, ControlCommand
)
from robot_motion_control.algorithms.path_control import PathController, ControlMode


def create_test_robot_model():
    """创建测试用机器人模型"""
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


def generate_test_trajectory(duration=2.0, dt=0.01):
    """生成测试轨迹"""
    t = np.arange(0, duration, dt)
    trajectory = []
    
    for time_step in t:
        # 正弦波轨迹
        amplitude = 0.5
        frequency = 1.0
        
        position = amplitude * np.sin(2 * np.pi * frequency * time_step) * np.ones(6)
        velocity = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * time_step) * np.ones(6)
        acceleration = -amplitude * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * time_step) * np.ones(6)
        jerk = -amplitude * (2 * np.pi * frequency)**3 * np.cos(2 * np.pi * frequency * time_step) * np.ones(6)
        
        point = TrajectoryPoint(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            time=time_step,
            path_parameter=time_step / duration
        )
        trajectory.append(point)
    
    return trajectory


def simulate_robot_response(command, current_state, dt=0.01):
    """简化的机器人响应仿真"""
    new_state = RobotState(
        joint_positions=current_state.joint_positions.copy(),
        joint_velocities=current_state.joint_velocities.copy(),
        joint_accelerations=current_state.joint_accelerations.copy(),
        joint_torques=current_state.joint_torques.copy(),
        end_effector_pose=current_state.end_effector_pose.copy(),
        timestamp=current_state.timestamp + dt
    )
    
    if command.control_mode == "position" and command.joint_positions is not None:
        # 位置控制：简化的一阶系统响应
        position_error = command.joint_positions - current_state.joint_positions
        position_change = np.clip(position_error * 0.5, -0.1, 0.1)  # 限制变化速度
        
        new_state.joint_positions += position_change
        new_state.joint_velocities = position_change / dt
        
    elif command.control_mode == "torque" and command.joint_torques is not None:
        # 力矩控制：简化的二阶系统响应
        # 假设简单的质量-阻尼系统
        mass = 1.0  # 简化质量
        damping = 0.5  # 阻尼系数
        
        # F = ma + cv，求解加速度
        acceleration = (command.joint_torques - damping * current_state.joint_velocities) / mass
        acceleration = np.clip(acceleration, -10.0, 10.0)  # 限制加速度
        
        new_state.joint_accelerations = acceleration
        new_state.joint_velocities += acceleration * dt
        new_state.joint_positions += new_state.joint_velocities * dt
        new_state.joint_torques = command.joint_torques
    
    return new_state


def test_control_mode(robot_model, trajectory, control_mode, enable_feedforward=True):
    """测试特定控制模式"""
    print(f"\n=== 测试 {control_mode.value} 控制模式 ===")
    
    # 创建控制器
    controller = PathController(
        robot_model,
        control_mode=control_mode,
        enable_feedforward=enable_feedforward
    )
    
    # 设置合适的控制增益
    if control_mode == ControlMode.PID:
        controller.set_control_gains(
            kp=np.ones(6) * 100.0,
            ki=np.ones(6) * 10.0,
            kd=np.ones(6) * 20.0
        )
    
    # 初始状态
    current_state = RobotState(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        joint_accelerations=np.zeros(6),
        joint_torques=np.zeros(6),
        end_effector_pose=np.eye(4),
        timestamp=0.0
    )
    
    # 仿真数据记录
    times = []
    actual_positions = []
    reference_positions = []
    tracking_errors = []
    control_efforts = []
    
    # 执行控制仿真
    for i, ref_point in enumerate(trajectory):
        current_state.timestamp = ref_point.time
        
        # 计算控制指令
        try:
            command = controller.compute_control(ref_point, current_state)
            
            # 记录数据
            times.append(ref_point.time)
            actual_positions.append(current_state.joint_positions[0])  # 只记录第一个关节
            reference_positions.append(ref_point.position[0])
            
            tracking_error = np.linalg.norm(ref_point.position - current_state.joint_positions)
            tracking_errors.append(tracking_error)
            
            if command.joint_torques is not None:
                control_efforts.append(np.linalg.norm(command.joint_torques))
            else:
                control_efforts.append(0.0)
            
            # 仿真机器人响应
            current_state = simulate_robot_response(command, current_state)
            
        except Exception as e:
            print(f"控制计算失败: {e}")
            break
    
    # 获取性能指标
    performance = controller.get_tracking_performance()
    
    print(f"平均跟踪误差: {performance['mean_tracking_error']:.6f}")
    print(f"最大跟踪误差: {performance['max_tracking_error']:.6f}")
    print(f"RMS跟踪误差: {performance['rms_tracking_error']:.6f}")
    print(f"平均计算时间: {performance['mean_computation_time']:.6f} s")
    
    return {
        'times': times,
        'actual_positions': actual_positions,
        'reference_positions': reference_positions,
        'tracking_errors': tracking_errors,
        'control_efforts': control_efforts,
        'performance': performance
    }


def plot_results(results_dict):
    """绘制结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('路径控制器性能对比', fontsize=16)
    
    # 位置跟踪对比
    ax1 = axes[0, 0]
    for mode_name, data in results_dict.items():
        ax1.plot(data['times'], data['actual_positions'], 
                label=f'{mode_name} (实际)', linestyle='--')
    
    # 参考轨迹（所有模式相同，只画一次）
    first_data = next(iter(results_dict.values()))
    ax1.plot(first_data['times'], first_data['reference_positions'], 
            'k-', linewidth=2, label='参考轨迹')
    
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('关节位置 (rad)')
    ax1.set_title('位置跟踪对比')
    ax1.legend()
    ax1.grid(True)
    
    # 跟踪误差对比
    ax2 = axes[0, 1]
    for mode_name, data in results_dict.items():
        ax2.plot(data['times'], data['tracking_errors'], label=mode_name)
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('跟踪误差 (rad)')
    ax2.set_title('跟踪误差对比')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')
    
    # 控制努力对比
    ax3 = axes[1, 0]
    for mode_name, data in results_dict.items():
        ax3.plot(data['times'], data['control_efforts'], label=mode_name)
    
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('控制努力')
    ax3.set_title('控制努力对比')
    ax3.legend()
    ax3.grid(True)
    
    # 性能指标对比
    ax4 = axes[1, 1]
    modes = list(results_dict.keys())
    mean_errors = [results_dict[mode]['performance']['mean_tracking_error'] for mode in modes]
    max_errors = [results_dict[mode]['performance']['max_tracking_error'] for mode in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    ax4.bar(x - width/2, mean_errors, width, label='平均误差', alpha=0.8)
    ax4.bar(x + width/2, max_errors, width, label='最大误差', alpha=0.8)
    
    ax4.set_xlabel('控制模式')
    ax4.set_ylabel('跟踪误差 (rad)')
    ax4.set_title('性能指标对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes, rotation=45)
    ax4.legend()
    ax4.grid(True, axis='y')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = project_root / "examples" / "path_controller_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n结果图片已保存到: {output_path}")
    
    plt.show()


def demonstrate_auto_tuning(robot_model, trajectory):
    """演示自动调参功能"""
    print("\n=== 自动调参演示 ===")
    
    controller = PathController(robot_model, control_mode=ControlMode.PID)
    
    # 创建简单的仿真数据用于调参
    reference_trajectory = trajectory[:50]  # 使用前50个点
    simulation_states = []
    
    current_state = RobotState(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        joint_accelerations=np.zeros(6),
        joint_torques=np.zeros(6),
        end_effector_pose=np.eye(4),
        timestamp=0.0
    )
    
    for ref_point in reference_trajectory:
        # 添加一些噪声模拟实际系统
        noise = np.random.normal(0, 0.01, 6)
        current_state.joint_positions = ref_point.position + noise
        simulation_states.append(current_state)
    
    # 执行自动调参
    print("执行自动调参...")
    original_gains = {
        'kp': controller.kp.copy(),
        'ki': controller.ki.copy(),
        'kd': controller.kd.copy()
    }
    
    best_gains = controller.auto_tune_gains(reference_trajectory, simulation_states)
    
    print("调参结果:")
    print(f"原始Kp: {original_gains['kp'][0]:.2f}")
    print(f"优化Kp: {best_gains['kp'][0]:.2f}")
    print(f"原始Ki: {original_gains['ki'][0]:.2f}")
    print(f"优化Ki: {best_gains['ki'][0]:.2f}")
    print(f"原始Kd: {original_gains['kd'][0]:.2f}")
    print(f"优化Kd: {best_gains['kd'][0]:.2f}")


def main():
    """主函数"""
    print("高精度路径跟踪控制器演示")
    print("=" * 50)
    
    # 创建机器人模型
    robot_model = create_test_robot_model()
    print(f"创建机器人模型: {robot_model.name}")
    print(f"关节数量: {robot_model.n_joints}")
    
    # 生成测试轨迹
    trajectory = generate_test_trajectory(duration=2.0, dt=0.01)
    print(f"生成测试轨迹: {len(trajectory)} 个点")
    
    # 测试不同控制模式
    control_modes = [
        ControlMode.PID,
        ControlMode.COMPUTED_TORQUE,
        ControlMode.SLIDING_MODE,
        ControlMode.ADAPTIVE
    ]
    
    results = {}
    
    for mode in control_modes:
        try:
            result = test_control_mode(robot_model, trajectory, mode)
            results[mode.value] = result
        except Exception as e:
            print(f"测试 {mode.value} 模式失败: {e}")
    
    # 绘制结果
    if results:
        plot_results(results)
    
    # 演示自动调参
    try:
        demonstrate_auto_tuning(robot_model, trajectory)
    except Exception as e:
        print(f"自动调参演示失败: {e}")
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()