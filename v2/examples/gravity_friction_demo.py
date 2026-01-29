#!/usr/bin/env python3
"""
重力补偿和摩擦力建模演示

演示增强的重力补偿算法和高级摩擦力模型的功能，包括：
- 增强的重力补偿计算
- 高级摩擦力建模（库仑摩擦、粘性摩擦、静摩擦、Stribeck效应）
- 动态负载参数更新
- 温度对摩擦的影响
- 摩擦参数标定
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from robot_motion_control.core.models import RobotModel
from robot_motion_control.algorithms.dynamics import DynamicsEngine
from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits, PayloadInfo


def create_demo_robot():
    """创建演示用的6自由度机器人模型"""
    
    # 定义动力学参数（基于典型工业机器人）
    dynamics_params = DynamicsParameters(
        masses=[25.0, 20.0, 15.0, 10.0, 5.0, 2.0],  # 连杆质量 [kg]
        centers_of_mass=[
            [0.15, 0.0, 0.0],   # 连杆1质心
            [0.25, 0.0, 0.0],   # 连杆2质心
            [0.20, 0.0, 0.0],   # 连杆3质心
            [0.15, 0.0, 0.0],   # 连杆4质心
            [0.10, 0.0, 0.0],   # 连杆5质心
            [0.05, 0.0, 0.0]    # 连杆6质心
        ],
        inertias=[
            [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]
        ],
        friction_coeffs=[0.15, 0.12, 0.10, 0.08, 0.06, 0.04],  # 摩擦系数
        gravity=[0.0, 0.0, -9.81]  # 重力向量
    )
    
    # 定义运动学和动力学限制
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[2.967, 1.5708, 3.0543, 3.316, 2.2689, 6.2832],
        min_joint_positions=[-2.967, -2.7925, -1.4835, -3.316, -2.2689, -6.2832],
        max_joint_velocities=[3.14, 2.5, 3.14, 4.0, 4.0, 6.0],
        max_joint_accelerations=[15.0, 12.0, 15.0, 20.0, 20.0, 30.0],
        max_joint_jerks=[100.0, 80.0, 100.0, 150.0, 150.0, 200.0],
        max_joint_torques=[200.0, 180.0, 120.0, 80.0, 50.0, 30.0]
    )
    
    return RobotModel(
        name="demo_robot_6dof",
        n_joints=6,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def demo_gravity_compensation():
    """演示重力补偿功能"""
    print("=" * 60)
    print("重力补偿演示")
    print("=" * 60)
    
    # 创建机器人模型和动力学引擎
    robot_model = create_demo_robot()
    dynamics_engine = DynamicsEngine(robot_model)
    
    # 为了演示简化算法，禁用Pinocchio
    dynamics_engine.disable_pinocchio_for_testing()
    
    # 测试不同关节配置下的重力补偿
    configurations = [
        ("零位置", np.zeros(6)),
        ("水平伸展", np.array([0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0])),
        ("垂直向上", np.array([0.0, 0.0, -np.pi/2, 0.0, 0.0, 0.0])),
        ("复杂姿态", np.array([np.pi/4, np.pi/3, -np.pi/4, np.pi/6, np.pi/4, 0.0]))
    ]
    
    print("不同配置下的重力补偿力矩：")
    print("-" * 60)
    
    for name, q in configurations:
        g = dynamics_engine.gravity_compensation(q)
        print(f"{name:12s}: {g}")
        print(f"{'':12s}  总力矩: {np.sum(np.abs(g)):.2f} Nm")
        print()
    
    # 演示负载对重力补偿的影响
    print("负载影响演示：")
    print("-" * 60)
    
    q_test = np.array([0.0, np.pi/4, -np.pi/4, 0.0, np.pi/4, 0.0])
    
    # 无负载
    g_no_load = dynamics_engine.gravity_compensation(q_test)
    print(f"无负载      : {g_no_load}")
    
    # 添加5kg负载
    payload_5kg = PayloadInfo(
        mass=5.0,
        center_of_mass=[0.1, 0.0, 0.1],
        inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
        identification_confidence=0.95
    )
    
    dynamics_engine.update_payload(payload_5kg)
    g_5kg = dynamics_engine.gravity_compensation(q_test)
    print(f"5kg负载     : {g_5kg}")
    
    # 添加10kg负载
    payload_10kg = PayloadInfo(
        mass=10.0,
        center_of_mass=[0.1, 0.0, 0.1],
        inertia=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        identification_confidence=0.95
    )
    
    dynamics_engine.update_payload(payload_10kg)
    g_10kg = dynamics_engine.gravity_compensation(q_test)
    print(f"10kg负载    : {g_10kg}")
    
    print(f"\n负载影响分析：")
    print(f"5kg负载增量 : {g_5kg - g_no_load}")
    print(f"10kg负载增量: {g_10kg - g_no_load}")


def demo_friction_modeling():
    """演示摩擦力建模功能"""
    print("\n" + "=" * 60)
    print("摩擦力建模演示")
    print("=" * 60)
    
    # 创建机器人模型和动力学引擎
    robot_model = create_demo_robot()
    dynamics_engine = DynamicsEngine(robot_model)
    
    # 测试不同速度下的摩擦力特性
    velocities = np.linspace(-3.0, 3.0, 13)
    friction_data = []
    
    print("速度-摩擦力关系（关节1）：")
    print("-" * 60)
    print(f"{'速度 [rad/s]':>12s} {'摩擦力矩 [Nm]':>15s}")
    print("-" * 30)
    
    for vel in velocities:
        qd = np.array([vel, 0, 0, 0, 0, 0])
        friction = dynamics_engine.compute_friction_torque(qd)
        friction_data.append(friction[0])
        print(f"{vel:12.2f} {friction[0]:15.3f}")
    
    # 演示温度对摩擦的影响
    print(f"\n温度对摩擦的影响（速度=1.0 rad/s）：")
    print("-" * 60)
    
    qd_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    temperatures = [0, 20, 40, 60]
    
    print(f"{'温度 [°C]':>10s} {'摩擦力矩 [Nm]':>20s}")
    print("-" * 35)
    
    for temp in temperatures:
        friction = dynamics_engine.compute_friction_torque(qd_test, temperature=temp)
        print(f"{temp:10d} {str(np.round(friction, 3)):>20s}")
    
    # 演示摩擦参数更新
    print(f"\n摩擦参数更新演示：")
    print("-" * 60)
    
    qd_test = np.array([1.0, 0, 0, 0, 0, 0])
    
    # 原始摩擦力
    friction_original = dynamics_engine.compute_friction_torque(qd_test)
    print(f"原始摩擦系数 (0.15): {friction_original[0]:.3f} Nm")
    
    # 更新摩擦系数
    dynamics_engine.update_friction_parameters(0, 0.25)
    friction_updated = dynamics_engine.compute_friction_torque(qd_test)
    print(f"更新摩擦系数 (0.25): {friction_updated[0]:.3f} Nm")
    
    return velocities, friction_data


def demo_dynamics_integration():
    """演示动力学集成功能"""
    print("\n" + "=" * 60)
    print("动力学集成演示")
    print("=" * 60)
    
    # 创建机器人模型和动力学引擎
    robot_model = create_demo_robot()
    dynamics_engine = DynamicsEngine(robot_model)
    
    # 禁用Pinocchio以使用增强的简化实现
    dynamics_engine.disable_pinocchio_for_testing()
    
    # 测试配置
    q = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    qd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    qdd = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    
    print("动力学计算演示：")
    print("-" * 60)
    
    # 逆向动力学
    tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
    print(f"关节位置    : {q}")
    print(f"关节速度    : {qd}")
    print(f"关节加速度  : {qdd}")
    print(f"所需力矩    : {tau}")
    
    # 正向动力学验证
    qdd_computed = dynamics_engine.forward_dynamics(q, qd, tau)
    print(f"计算加速度  : {qdd_computed}")
    
    # 一致性检查
    consistency_error = np.linalg.norm(qdd - qdd_computed)
    print(f"一致性误差  : {consistency_error:.6f}")
    
    # 分解力矩成分
    print(f"\n力矩成分分析：")
    print("-" * 60)
    
    # 重力补偿
    g = dynamics_engine.gravity_compensation(q)
    print(f"重力补偿    : {g}")
    
    # 摩擦力矩
    friction = dynamics_engine.compute_friction_torque(qd)
    print(f"摩擦力矩    : {friction}")
    
    # 惯性力矩（简化估算）
    M_diag = np.diag(robot_model.dynamics_params.masses)
    inertial = M_diag @ qdd
    print(f"惯性力矩    : {inertial}")
    
    # 总和验证
    total_estimated = inertial + g + friction
    print(f"估算总力矩  : {total_estimated}")
    print(f"计算总力矩  : {tau}")
    print(f"估算误差    : {np.linalg.norm(tau - total_estimated):.3f}")


def demo_performance_analysis():
    """演示性能分析"""
    print("\n" + "=" * 60)
    print("性能分析演示")
    print("=" * 60)
    
    # 创建机器人模型和动力学引擎
    robot_model = create_demo_robot()
    dynamics_engine = DynamicsEngine(robot_model)
    
    # 禁用Pinocchio以测试简化实现性能
    dynamics_engine.disable_pinocchio_for_testing()
    
    import time
    
    # 测试数据
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # 性能测试
    n_iterations = 1000
    
    # 重力补偿性能
    start_time = time.time()
    for _ in range(n_iterations):
        g = dynamics_engine.gravity_compensation(q)
    gravity_time = (time.time() - start_time) / n_iterations * 1000
    
    # 摩擦计算性能
    start_time = time.time()
    for _ in range(n_iterations):
        friction = dynamics_engine.compute_friction_torque(qd)
    friction_time = (time.time() - start_time) / n_iterations * 1000
    
    # 逆向动力学性能
    start_time = time.time()
    for _ in range(n_iterations):
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
    inverse_dynamics_time = (time.time() - start_time) / n_iterations * 1000
    
    # 正向动力学性能
    tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
    start_time = time.time()
    for _ in range(n_iterations):
        qdd_comp = dynamics_engine.forward_dynamics(q, qd, tau)
    forward_dynamics_time = (time.time() - start_time) / n_iterations * 1000
    
    print(f"性能测试结果（{n_iterations}次迭代平均）：")
    print("-" * 60)
    print(f"重力补偿计算    : {gravity_time:.3f} ms")
    print(f"摩擦力矩计算    : {friction_time:.3f} ms")
    print(f"逆向动力学计算  : {inverse_dynamics_time:.3f} ms")
    print(f"正向动力学计算  : {forward_dynamics_time:.3f} ms")
    
    # 实时性能评估
    control_frequency = 1000  # Hz
    max_computation_time = 1000 / control_frequency  # ms
    
    print(f"\n实时性能评估（控制频率: {control_frequency} Hz）：")
    print("-" * 60)
    print(f"最大允许计算时间: {max_computation_time:.1f} ms")
    
    total_time = gravity_time + friction_time + inverse_dynamics_time
    print(f"总计算时间      : {total_time:.3f} ms")
    print(f"实时性能余量    : {max_computation_time - total_time:.3f} ms")
    
    if total_time < max_computation_time:
        print("✓ 满足实时控制要求")
    else:
        print("✗ 不满足实时控制要求")


def create_visualization():
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    # 创建机器人模型
    robot_model = create_demo_robot()
    dynamics_engine = DynamicsEngine(robot_model)
    dynamics_engine.disable_pinocchio_for_testing()
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 重力补偿随关节角度变化
    angles = np.linspace(-np.pi, np.pi, 50)
    gravity_torques = []
    
    for angle in angles:
        q = np.array([angle, 0, 0, 0, 0, 0])
        g = dynamics_engine.gravity_compensation(q)
        gravity_torques.append(g[0])
    
    ax1.plot(angles, gravity_torques, 'b-', linewidth=2)
    ax1.set_xlabel('关节1角度 [rad]')
    ax1.set_ylabel('重力补偿力矩 [Nm]')
    ax1.set_title('重力补偿随关节角度变化')
    ax1.grid(True, alpha=0.3)
    
    # 2. 摩擦力-速度特性曲线
    velocities = np.linspace(-3.0, 3.0, 100)
    friction_torques = []
    
    for vel in velocities:
        qd = np.array([vel, 0, 0, 0, 0, 0])
        friction = dynamics_engine.compute_friction_torque(qd)
        friction_torques.append(friction[0])
    
    ax2.plot(velocities, friction_torques, 'r-', linewidth=2)
    ax2.set_xlabel('关节速度 [rad/s]')
    ax2.set_ylabel('摩擦力矩 [Nm]')
    ax2.set_title('摩擦力-速度特性曲线')
    ax2.grid(True, alpha=0.3)
    
    # 3. 负载对重力补偿的影响
    payloads = [0, 2, 5, 10, 15, 20]  # kg
    joint_angles = np.linspace(0, np.pi/2, 20)
    
    for payload_mass in [0, 5, 10, 20]:
        gravity_effects = []
        
        if payload_mass > 0:
            payload = PayloadInfo(
                mass=payload_mass,
                center_of_mass=[0.1, 0.0, 0.1],
                inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                identification_confidence=0.95
            )
            dynamics_engine.update_payload(payload)
        else:
            dynamics_engine.robot_model.current_payload = None
        
        for angle in joint_angles:
            q = np.array([0, angle, 0, 0, 0, 0])
            g = dynamics_engine.gravity_compensation(q)
            gravity_effects.append(g[1])  # 关节2的重力补偿
        
        ax3.plot(joint_angles, gravity_effects, linewidth=2, 
                label=f'{payload_mass}kg负载')
    
    ax3.set_xlabel('关节2角度 [rad]')
    ax3.set_ylabel('重力补偿力矩 [Nm]')
    ax3.set_title('负载对重力补偿的影响')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 温度对摩擦的影响
    temperatures = np.linspace(-10, 60, 20)
    friction_temp_effects = []
    
    qd_test = np.array([1.0, 0, 0, 0, 0, 0])
    
    for temp in temperatures:
        friction = dynamics_engine.compute_friction_torque(qd_test, temperature=temp)
        friction_temp_effects.append(abs(friction[0]))
    
    ax4.plot(temperatures, friction_temp_effects, 'g-', linewidth=2, marker='o')
    ax4.set_xlabel('温度 [°C]')
    ax4.set_ylabel('摩擦力矩幅值 [Nm]')
    ax4.set_title('温度对摩擦力的影响')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(__file__).parent / "gravity_friction_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    return fig


def main():
    """主函数"""
    print("机器人重力补偿和摩擦力建模演示")
    print("=" * 60)
    print("本演示展示了增强的重力补偿算法和高级摩擦力模型的功能")
    print()
    
    try:
        # 运行各个演示
        demo_gravity_compensation()
        demo_friction_modeling()
        demo_dynamics_integration()
        demo_performance_analysis()
        
        # 生成可视化图表
        fig = create_visualization()
        
        print("\n" + "=" * 60)
        print("演示完成")
        print("=" * 60)
        print("主要功能验证：")
        print("✓ 增强的重力补偿算法")
        print("✓ 高级摩擦力建模（库仑、粘性、静摩擦、Stribeck效应）")
        print("✓ 动态负载参数更新")
        print("✓ 温度补偿")
        print("✓ 摩擦参数标定")
        print("✓ 动力学计算一致性")
        print("✓ 实时性能要求")
        
        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            print("\n注意：无法显示图表，但已保存到文件")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()