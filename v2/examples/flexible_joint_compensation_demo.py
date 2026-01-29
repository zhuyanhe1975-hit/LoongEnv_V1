#!/usr/bin/env python3
"""
柔性关节补偿算法演示

演示增强的柔性关节动力学模型、柔性补偿控制算法和末端反馈补偿功能。
展示如何使用这些算法来抑制机器人运动中的振动和提高控制精度。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

from robot_motion_control.algorithms.vibration_suppression import (
    VibrationSuppressor, FlexibleJointParameters, EndEffectorSensorData, VirtualSensorState
)
from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.types import (
    ControlCommand, RobotState, PayloadInfo, DynamicsParameters, KinodynamicLimits
)


def create_sample_robot_model() -> RobotModel:
    """创建示例机器人模型"""
    # 创建动力学参数
    dynamics_params = DynamicsParameters(
        masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
        centers_of_mass=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.15], [0.0, 0.0, 0.12], 
                        [0.0, 0.0, 0.08], [0.0, 0.0, 0.05], [0.0, 0.0, 0.03]],
        inertias=[[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]] for _ in range(6)],
        friction_coeffs=[0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
        gravity=[0.0, 0.0, -9.81]
    )
    
    # 创建运动学动力学限制
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[3.14, 2.0, 2.5, 3.14, 2.0, 3.14],
        min_joint_positions=[-3.14, -2.0, -2.5, -3.14, -2.0, -3.14],
        max_joint_velocities=[2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        max_joint_accelerations=[10.0, 10.0, 10.0, 15.0, 15.0, 15.0],
        max_joint_jerks=[100.0, 100.0, 100.0, 150.0, 150.0, 150.0],
        max_joint_torques=[100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
    )
    
    return RobotModel(
        name="ER15-1400",
        n_joints=6,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def create_flexible_joint_parameters() -> FlexibleJointParameters:
    """创建柔性关节参数"""
    return FlexibleJointParameters(
        joint_stiffness=np.array([1e5, 1e5, 8e4, 5e4, 3e4, 2e4]),  # 不同关节不同刚度
        joint_damping=np.array([100.0, 100.0, 80.0, 50.0, 30.0, 20.0]),  # 不同阻尼
        motor_inertia=np.array([0.01, 0.01, 0.008, 0.005, 0.003, 0.002]),
        link_inertia=np.array([0.1, 0.1, 0.08, 0.05, 0.03, 0.02]),
        gear_ratio=np.array([100.0, 100.0, 80.0, 50.0, 30.0, 20.0]),
        transmission_compliance=np.array([1e-6, 1e-6, 1.5e-6, 2e-6, 3e-6, 4e-6])
    )


def create_payload_scenarios() -> List[PayloadInfo]:
    """创建不同的负载场景"""
    scenarios = []
    
    # 轻负载
    scenarios.append(PayloadInfo(
        mass=1.0,
        center_of_mass=[0.05, 0.02, 0.1],
        inertia=[[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]],
        identification_confidence=0.95
    ))
    
    # 中等负载
    scenarios.append(PayloadInfo(
        mass=5.0,
        center_of_mass=[0.1, 0.05, 0.15],
        inertia=[[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        identification_confidence=0.9
    ))
    
    # 重负载（长悬臂）
    scenarios.append(PayloadInfo(
        mass=10.0,
        center_of_mass=[0.2, 0.1, 0.3],  # 较大的重心偏移
        inertia=[[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
        identification_confidence=0.85
    ))
    
    return scenarios


def generate_test_trajectory(duration: float = 5.0, dt: float = 0.001) -> List[ControlCommand]:
    """生成测试轨迹"""
    t = np.arange(0, duration, dt)
    trajectory = []
    
    for i, time_step in enumerate(t):
        # 生成正弦波轨迹
        positions = np.array([
            0.5 * np.sin(2 * np.pi * 0.2 * time_step),
            0.3 * np.sin(2 * np.pi * 0.3 * time_step + np.pi/4),
            0.4 * np.sin(2 * np.pi * 0.25 * time_step + np.pi/2),
            0.6 * np.sin(2 * np.pi * 0.15 * time_step + 3*np.pi/4),
            0.2 * np.sin(2 * np.pi * 0.4 * time_step + np.pi),
            0.3 * np.sin(2 * np.pi * 0.35 * time_step + 5*np.pi/4)
        ])
        
        # 计算速度（数值微分）
        if i > 0:
            velocities = (positions - prev_positions) / dt
        else:
            velocities = np.zeros(6)
        
        # 计算加速度
        if i > 1:
            accelerations = (velocities - prev_velocities) / dt
        else:
            accelerations = np.zeros(6)
        
        # 简化的力矩计算
        torques = 10.0 * positions + 5.0 * velocities + 2.0 * accelerations
        
        command = ControlCommand(
            joint_positions=positions,
            joint_velocities=velocities,
            joint_accelerations=accelerations,
            joint_torques=torques,
            control_mode="torque",
            timestamp=time_step
        )
        
        trajectory.append(command)
        prev_positions = positions.copy()
        prev_velocities = velocities.copy()
    
    return trajectory


def simulate_end_effector_sensor(command: ControlCommand, noise_level: float = 0.01) -> EndEffectorSensorData:
    """模拟末端执行器传感器数据"""
    # 简化的正向运动学（假设末端位置）
    position = np.array([
        0.5 + 0.3 * command.joint_positions[0] + 0.2 * command.joint_positions[1],
        0.3 + 0.2 * command.joint_positions[2] + 0.1 * command.joint_positions[3],
        0.8 + 0.1 * command.joint_positions[4] + 0.05 * command.joint_positions[5]
    ])
    
    # 添加噪声
    position += np.random.normal(0, noise_level, 3)
    
    # 计算速度和加速度
    velocity = np.array([
        0.3 * command.joint_velocities[0] + 0.2 * command.joint_velocities[1],
        0.2 * command.joint_velocities[2] + 0.1 * command.joint_velocities[3],
        0.1 * command.joint_velocities[4] + 0.05 * command.joint_velocities[5]
    ])
    
    acceleration = np.array([
        0.3 * command.joint_accelerations[0] + 0.2 * command.joint_accelerations[1],
        0.2 * command.joint_accelerations[2] + 0.1 * command.joint_accelerations[3],
        0.1 * command.joint_accelerations[4] + 0.05 * command.joint_accelerations[5]
    ])
    
    # 模拟接触力
    force = np.random.normal(0, 1.0, 3)
    torque = np.random.normal(0, 0.5, 3)
    
    return EndEffectorSensorData(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        force=force,
        torque=torque,
        timestamp=command.timestamp
    )


def demonstrate_flexible_joint_compensation():
    """演示柔性关节补偿算法"""
    print("=" * 80)
    print("柔性关节补偿算法演示")
    print("=" * 80)
    
    # 1. 创建机器人模型和参数
    print("\n1. 初始化机器人模型和柔性参数...")
    robot_model = create_sample_robot_model()
    flexible_params = create_flexible_joint_parameters()
    payload_scenarios = create_payload_scenarios()
    
    print(f"✓ 机器人模型: {robot_model.name}, {robot_model.n_joints}关节")
    print(f"✓ 柔性参数: 刚度范围 {flexible_params.joint_stiffness.min():.0f}-{flexible_params.joint_stiffness.max():.0f} Nm/rad")
    print(f"✓ 负载场景: {len(payload_scenarios)}种不同负载")
    
    # 2. 创建振动抑制器
    print("\n2. 创建增强的振动抑制器...")
    suppressor = VibrationSuppressor(robot_model, flexible_params)
    
    print(f"✓ 柔性关节观测器: {len(suppressor.flexible_joint_observer)}个")
    print(f"✓ 虚拟传感器置信度: {suppressor.get_virtual_sensor_confidence():.3f}")
    print(f"✓ 末端执行器控制器已初始化")
    
    # 3. 生成测试轨迹
    print("\n3. 生成测试轨迹...")
    trajectory = generate_test_trajectory(duration=2.0, dt=0.01)
    print(f"✓ 轨迹长度: {len(trajectory)}点")
    print(f"✓ 轨迹持续时间: {trajectory[-1].timestamp:.1f}秒")
    
    # 4. 测试不同负载场景下的补偿效果
    print("\n4. 测试不同负载场景下的补偿效果...")
    
    results = {}
    
    for i, payload in enumerate(payload_scenarios):
        scenario_name = f"负载{i+1} ({payload.mass:.1f}kg)"
        print(f"\n   场景: {scenario_name}")
        
        # 记录补偿前后的数据
        original_torques = []
        compensated_torques = []
        compensation_amounts = []
        virtual_sensor_confidences = []
        flexible_joint_deflections = []
        
        # 模拟机器人状态
        current_state = RobotState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_pose=np.eye(4),
            timestamp=0.0
        )
        
        for j, command in enumerate(trajectory[:100]):  # 只取前100个点进行演示
            # 更新机器人状态
            current_state.joint_positions = command.joint_positions
            current_state.joint_velocities = command.joint_velocities
            current_state.joint_accelerations = command.joint_accelerations
            current_state.joint_torques = command.joint_torques
            current_state.timestamp = command.timestamp
            
            # 应用柔性关节补偿
            compensated_command = suppressor.compensate_flexible_joints(
                command, current_state, payload
            )
            
            # 模拟末端执行器传感器数据
            sensor_data = simulate_end_effector_sensor(command)
            
            # 应用末端反馈补偿
            final_command = suppressor.apply_end_effector_feedback(
                compensated_command, sensor_data
            )
            
            # 记录数据
            original_torques.append(command.joint_torques.copy())
            compensated_torques.append(final_command.joint_torques.copy())
            compensation_amounts.append(
                np.linalg.norm(final_command.joint_torques - command.joint_torques)
            )
            virtual_sensor_confidences.append(suppressor.get_virtual_sensor_confidence())
            
            # 记录柔性关节偏转
            joint_0_state = suppressor.get_flexible_joint_state(0)
            flexible_joint_deflections.append(abs(joint_0_state['deflection']))
        
        # 计算统计信息
        avg_compensation = np.mean(compensation_amounts)
        max_compensation = np.max(compensation_amounts)
        avg_confidence = np.mean(virtual_sensor_confidences)
        avg_deflection = np.mean(flexible_joint_deflections)
        
        results[scenario_name] = {
            'avg_compensation': avg_compensation,
            'max_compensation': max_compensation,
            'avg_confidence': avg_confidence,
            'avg_deflection': avg_deflection,
            'original_torques': original_torques,
            'compensated_torques': compensated_torques
        }
        
        print(f"     平均补偿量: {avg_compensation:.3f} Nm")
        print(f"     最大补偿量: {max_compensation:.3f} Nm")
        print(f"     平均虚拟传感器置信度: {avg_confidence:.3f}")
        print(f"     平均柔性关节偏转: {avg_deflection:.6f} rad")
    
    # 5. 测试参数调整功能
    print("\n5. 测试参数调整功能...")
    
    # 调整末端执行器增益
    new_position_gains = np.array([2000.0, 2000.0, 2000.0])
    new_velocity_gains = np.array([200.0, 200.0, 200.0])
    new_force_gains = np.array([0.2, 0.2, 0.2])
    new_integral_gains = np.array([20.0, 20.0, 20.0])
    
    suppressor.set_end_effector_gains(
        new_position_gains, new_velocity_gains, new_force_gains, new_integral_gains
    )
    print("✓ 末端执行器增益已更新")
    
    # 启用自适应补偿
    suppressor.enable_adaptive_compensation(True)
    print("✓ 自适应补偿已启用")
    
    # 重置积分误差
    suppressor.reset_integral_errors()
    print("✓ 积分误差已重置")
    
    # 6. 获取诊断信息
    print("\n6. 获取补偿算法诊断信息...")
    diagnostics = suppressor.get_compensation_diagnostics()
    
    print(f"✓ 虚拟传感器置信度: {diagnostics['virtual_sensor_confidence']:.3f}")
    print(f"✓ 输入缓冲区大小: {diagnostics['buffer_sizes']['input_buffer']}")
    print(f"✓ 状态历史缓冲区大小: {diagnostics['buffer_sizes']['state_history']}")
    print(f"✓ 积分误差范数: {np.linalg.norm(diagnostics['integral_errors']):.6f}")
    
    # 显示每个关节的柔性状态
    print("\n   各关节柔性状态:")
    for i in range(robot_model.n_joints):
        joint_state = diagnostics['flexible_joint_states'][f'joint_{i}']
        print(f"     关节{i}: 偏转={joint_state['deflection']:.6f} rad, "
              f"偏转率={joint_state['deflection_rate']:.6f} rad/s")
    
    # 7. 性能分析
    print("\n7. 性能分析...")
    
    # 测试计算性能
    start_time = time.time()
    test_command = trajectory[50]
    test_state = current_state
    test_payload = payload_scenarios[1]
    
    for _ in range(100):
        compensated = suppressor.compensate_flexible_joints(test_command, test_state, test_payload)
    
    computation_time = (time.time() - start_time) / 100 * 1000  # ms
    print(f"✓ 柔性补偿计算时间: {computation_time:.3f} ms/次")
    
    # 测试末端反馈性能
    start_time = time.time()
    test_sensor_data = simulate_end_effector_sensor(test_command)
    
    for _ in range(100):
        feedback_compensated = suppressor.apply_end_effector_feedback(test_command, test_sensor_data)
    
    feedback_time = (time.time() - start_time) / 100 * 1000  # ms
    print(f"✓ 末端反馈补偿计算时间: {feedback_time:.3f} ms/次")
    
    # 8. 结果总结
    print("\n8. 结果总结")
    print("=" * 50)
    
    for scenario_name, result in results.items():
        print(f"\n{scenario_name}:")
        print(f"  - 补偿效果: 平均{result['avg_compensation']:.3f} Nm, "
              f"最大{result['max_compensation']:.3f} Nm")
        print(f"  - 虚拟传感器性能: 平均置信度{result['avg_confidence']:.3f}")
        print(f"  - 柔性关节性能: 平均偏转{result['avg_deflection']:.6f} rad")
    
    print(f"\n算法性能:")
    print(f"  - 柔性补偿: {computation_time:.3f} ms/次")
    print(f"  - 末端反馈: {feedback_time:.3f} ms/次")
    print(f"  - 总计算时间: {computation_time + feedback_time:.3f} ms/次")
    
    print("\n✓ 柔性关节补偿算法演示完成!")
    print("  增强的算法成功实现了:")
    print("  1. 柔性关节动力学建模和状态估计")
    print("  2. 多层次补偿策略（柔性、传动、负载自适应）")
    print("  3. 末端反馈补偿和虚拟传感器融合")
    print("  4. 实时性能和参数可调性")
    
    return results


def plot_compensation_results(results: Dict[str, Any]):
    """绘制补偿结果"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('柔性关节补偿算法效果分析', fontsize=16)
        
        # 提取数据
        scenarios = list(results.keys())
        avg_compensations = [results[s]['avg_compensation'] for s in scenarios]
        max_compensations = [results[s]['max_compensation'] for s in scenarios]
        avg_confidences = [results[s]['avg_confidence'] for s in scenarios]
        avg_deflections = [results[s]['avg_deflection'] for s in scenarios]
        
        # 补偿量对比
        axes[0, 0].bar(scenarios, avg_compensations, alpha=0.7, label='平均补偿量')
        axes[0, 0].bar(scenarios, max_compensations, alpha=0.7, label='最大补偿量')
        axes[0, 0].set_title('补偿量对比')
        axes[0, 0].set_ylabel('补偿量 (Nm)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 虚拟传感器置信度
        axes[0, 1].bar(scenarios, avg_confidences, alpha=0.7, color='green')
        axes[0, 1].set_title('虚拟传感器置信度')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 柔性关节偏转
        axes[1, 0].bar(scenarios, avg_deflections, alpha=0.7, color='orange')
        axes[1, 0].set_title('柔性关节偏转')
        axes[1, 0].set_ylabel('偏转 (rad)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 力矩时间序列（第一个场景）
        first_scenario = scenarios[0]
        original = np.array(results[first_scenario]['original_torques'])
        compensated = np.array(results[first_scenario]['compensated_torques'])
        
        time_steps = np.arange(len(original))
        axes[1, 1].plot(time_steps, original[:, 0], label='原始力矩', alpha=0.7)
        axes[1, 1].plot(time_steps, compensated[:, 0], label='补偿后力矩', alpha=0.7)
        axes[1, 1].set_title(f'{first_scenario} - 关节1力矩对比')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('力矩 (Nm)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('flexible_joint_compensation_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ 结果图表已保存为 'flexible_joint_compensation_results.png'")
        
    except ImportError:
        print("\n注意: matplotlib未安装，跳过结果绘图")


if __name__ == "__main__":
    # 运行演示
    results = demonstrate_flexible_joint_compensation()
    
    # 绘制结果（如果matplotlib可用）
    plot_compensation_results(results)
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("本演示展示了增强的柔性关节补偿算法的以下特性：")
    print("1. 柔性关节双质量动力学模型和状态观测器")
    print("2. 多层次补偿策略（柔性、传动、负载自适应）")
    print("3. 末端执行器反馈补偿和虚拟传感器融合")
    print("4. 卡尔曼滤波器状态估计和置信度评估")
    print("5. 实时性能和参数可调性")
    print("6. 不同负载场景下的自适应补偿效果")
    print("=" * 80)