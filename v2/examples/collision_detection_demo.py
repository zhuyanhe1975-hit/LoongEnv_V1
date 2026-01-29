"""
碰撞检测算法演示

演示基于距离的碰撞检测和避让策略的使用方法和效果。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    RobotState, ControlCommand, KinodynamicLimits, DynamicsParameters
)
from src.robot_motion_control.algorithms.collision_detection import (
    CollisionMonitor, CollisionDetector, CollisionAvoidance,
    DistanceCalculator, CollisionType
)


def create_demo_robot_model() -> RobotModel:
    """创建演示用的机器人模型"""
    
    # 创建运动学限制
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[3.14, 2.0, 2.5, 3.14, 2.0, 3.14],
        min_joint_positions=[-3.14, -2.0, -2.5, -3.14, -2.0, -3.14],
        max_joint_velocities=[2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        max_joint_accelerations=[5.0, 5.0, 5.0, 10.0, 10.0, 10.0],
        max_joint_jerks=[50.0, 50.0, 50.0, 100.0, 100.0, 100.0],
        max_joint_torques=[100.0, 100.0, 50.0, 20.0, 20.0, 10.0]
    )
    
    # 创建动力学参数
    dynamics_params = DynamicsParameters(
        masses=[5.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        centers_of_mass=[
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.15],
            [0.0, 0.0, 0.12],
            [0.0, 0.0, 0.08],
            [0.0, 0.0, 0.06],
            [0.0, 0.0, 0.04]
        ],
        inertias=[
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.05]],
            [[0.08, 0.0, 0.0], [0.0, 0.08, 0.0], [0.0, 0.0, 0.04]],
            [[0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 0.03]],
            [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.02]],
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.01]],
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.005]]
        ],
        friction_coeffs=[0.1, 0.1, 0.08, 0.06, 0.04, 0.02],
        gravity=[0.0, 0.0, -9.81]
    )
    
    # 创建机器人模型
    robot_model = RobotModel(
        name="ER15-1400-Demo",
        n_joints=6,
        kinodynamic_limits=kinodynamic_limits,
        dynamics_params=dynamics_params
    )
    
    return robot_model


def demo_distance_calculations():
    """演示距离计算算法"""
    print("=== 距离计算算法演示 ===")
    
    # 球体间距离计算
    print("\n1. 球体间距离计算:")
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([1.5, 0.0, 0.0])
    radius1 = 0.3
    radius2 = 0.2
    
    distance, point1, point2 = DistanceCalculator.sphere_sphere_distance(
        center1, radius1, center2, radius2
    )
    
    print(f"  球心1: {center1}, 半径: {radius1}")
    print(f"  球心2: {center2}, 半径: {radius2}")
    print(f"  距离: {distance:.3f} m")
    print(f"  最近点1: {point1}")
    print(f"  最近点2: {point2}")
    
    # 球体与圆柱体距离计算
    print("\n2. 球体与圆柱体距离计算:")
    sphere_center = np.array([2.0, 0.0, 0.0])
    sphere_radius = 0.2
    cylinder_start = np.array([0.0, 0.0, -1.0])
    cylinder_end = np.array([0.0, 0.0, 1.0])
    cylinder_radius = 0.3
    
    distance, point1, point2 = DistanceCalculator.sphere_cylinder_distance(
        sphere_center, sphere_radius,
        cylinder_start, cylinder_end, cylinder_radius
    )
    
    print(f"  球心: {sphere_center}, 半径: {sphere_radius}")
    print(f"  圆柱体: {cylinder_start} -> {cylinder_end}, 半径: {cylinder_radius}")
    print(f"  距离: {distance:.3f} m")
    print(f"  球上最近点: {point1}")
    print(f"  圆柱体上最近点: {point2}")
    
    # 点到盒子距离计算
    print("\n3. 点到盒子距离计算:")
    point = np.array([2.0, 1.5, 0.5])
    box_center = np.array([0.0, 0.0, 0.0])
    box_dimensions = np.array([1.0, 1.0, 1.0])
    
    distance, closest_point = DistanceCalculator.point_box_distance(
        point, box_center, box_dimensions
    )
    
    print(f"  点: {point}")
    print(f"  盒子中心: {box_center}, 尺寸: {box_dimensions}")
    print(f"  距离: {distance:.3f} m")
    print(f"  盒子上最近点: {closest_point}")


def demo_collision_detection():
    """演示碰撞检测算法"""
    print("\n=== 碰撞检测算法演示 ===")
    
    # 创建机器人模型和碰撞检测器
    robot_model = create_demo_robot_model()
    collision_detector = CollisionDetector(robot_model)
    
    print(f"机器人关节数: {robot_model.n_joints}")
    print(f"碰撞几何体数量: {len(collision_detector.collision_geometries)}")
    print(f"碰撞对数量: {len(collision_detector.collision_pairs)}")
    
    # 测试不同的机器人配置
    configurations = [
        ("正常配置", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("轻微弯曲", np.array([0.5, 0.3, -0.2, 0.1, -0.1, 0.0])),
        ("中等弯曲", np.array([1.0, 0.8, -0.5, 0.3, -0.3, 0.2])),
        ("极端配置", np.array([2.5, 1.5, -1.8, 2.0, -1.5, 2.8])),
    ]
    
    for config_name, joint_positions in configurations:
        print(f"\n{config_name}:")
        print(f"  关节位置: {joint_positions}")
        
        # 创建机器人状态
        robot_state = RobotState(
            joint_positions=joint_positions,
            joint_velocities=np.zeros(6),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=time.time()
        )
        
        # 检测碰撞
        start_time = time.time()
        collisions = collision_detector.check_collisions(robot_state)
        detection_time = time.time() - start_time
        
        print(f"  检测时间: {detection_time*1000:.2f} ms")
        print(f"  检测到碰撞数量: {len(collisions)}")
        
        if collisions:
            for i, collision in enumerate(collisions[:3]):  # 只显示前3个
                print(f"    碰撞 {i+1}:")
                print(f"      类型: {collision.collision_type.value}")
                print(f"      距离: {collision.distance:.4f} m")
                print(f"      严重程度: {collision.severity:.2f}")
                print(f"      几何体: {collision.collision_pair.geometry1.name} <-> {collision.collision_pair.geometry2.name}")


def demo_collision_avoidance():
    """演示碰撞避让算法"""
    print("\n=== 碰撞避让算法演示 ===")
    
    # 创建机器人模型和避让控制器
    robot_model = create_demo_robot_model()
    collision_avoidance = CollisionAvoidance(robot_model)
    
    print(f"排斥力增益: {collision_avoidance.repulsive_gain}")
    print(f"吸引力增益: {collision_avoidance.attractive_gain}")
    print(f"影响距离: {collision_avoidance.influence_distance} m")
    print(f"最大避让速度: {collision_avoidance.max_avoidance_velocity} rad/s")
    
    # 创建机器人状态
    robot_state = RobotState(
        joint_positions=np.array([1.5, 1.0, -1.0, 1.0, -0.5, 1.5]),
        joint_velocities=np.array([0.1, 0.05, -0.08, 0.12, -0.06, 0.09]),
        joint_accelerations=np.zeros(6),
        joint_torques=np.zeros(6),
        end_effector_transform=np.eye(4),
        timestamp=time.time()
    )
    
    # 期望控制指令
    desired_command = ControlCommand(
        joint_velocities=np.array([0.2, 0.1, -0.15, 0.25, -0.12, 0.18]),
        control_mode="velocity",
        timestamp=time.time()
    )
    
    print(f"\n当前关节位置: {robot_state.joint_positions}")
    print(f"当前关节速度: {robot_state.joint_velocities}")
    print(f"期望关节速度: {desired_command.joint_velocities}")
    
    # 首先检测碰撞
    collision_detector = CollisionDetector(robot_model)
    collisions = collision_detector.check_collisions(robot_state)
    
    print(f"\n检测到 {len(collisions)} 个碰撞")
    
    if collisions:
        # 计算避让指令
        avoidance_command = collision_avoidance.compute_avoidance_command(
            collisions, robot_state, desired_command
        )
        
        print(f"\n避让指令:")
        print(f"  避让速度: {avoidance_command.joint_velocities}")
        print(f"  避让加速度: {avoidance_command.joint_accelerations}")
        print(f"  避让力: {avoidance_command.avoidance_force}")
        print(f"  优先级: {avoidance_command.priority:.2f}")
        
        # 混合原始指令和避让指令
        blend_factor = min(avoidance_command.priority, 1.0)
        final_velocities = (
            (1 - blend_factor) * desired_command.joint_velocities +
            blend_factor * avoidance_command.joint_velocities
        )
        
        print(f"\n最终混合速度 (混合因子: {blend_factor:.2f}):")
        print(f"  {final_velocities}")
    else:
        print("无碰撞，不需要避让")


def demo_collision_monitor():
    """演示碰撞监控器"""
    print("\n=== 碰撞监控器演示 ===")
    
    # 创建机器人模型和监控器
    robot_model = create_demo_robot_model()
    collision_monitor = CollisionMonitor(robot_model)
    
    print(f"监控器状态: {'启用' if collision_monitor.is_enabled else '禁用'}")
    
    # 模拟一系列运动
    print("\n模拟机器人运动序列:")
    
    trajectory_data = []
    collision_data = []
    avoidance_data = []
    
    # 生成轨迹：从初始位置到目标位置
    initial_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_pos = np.array([2.0, 1.5, -1.5, 1.8, -1.2, 2.5])
    
    num_steps = 50
    for i in range(num_steps):
        # 线性插值
        alpha = i / (num_steps - 1)
        current_pos = (1 - alpha) * initial_pos + alpha * target_pos
        
        # 添加一些速度
        current_vel = 0.1 * np.sin(2 * np.pi * alpha * 3)  # 正弦速度曲线
        
        # 创建机器人状态
        robot_state = RobotState(
            joint_positions=current_pos,
            joint_velocities=np.full(6, current_vel),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=time.time() + i * 0.02
        )
        
        # 期望控制指令
        desired_command = ControlCommand(
            joint_velocities=np.full(6, current_vel),
            control_mode="velocity",
            timestamp=robot_state.timestamp
        )
        
        # 更新监控器
        collisions, avoidance_command = collision_monitor.update(
            robot_state, desired_command
        )
        
        # 记录数据
        trajectory_data.append({
            'step': i,
            'positions': current_pos.copy(),
            'velocities': robot_state.joint_velocities.copy()
        })
        
        collision_data.append({
            'step': i,
            'num_collisions': len(collisions),
            'max_severity': max([c.severity for c in collisions]) if collisions else 0.0,
            'min_distance': min([c.distance for c in collisions]) if collisions else float('inf')
        })
        
        if avoidance_command:
            avoidance_data.append({
                'step': i,
                'priority': avoidance_command.priority,
                'avoidance_velocity_norm': np.linalg.norm(avoidance_command.joint_velocities)
            })
        
        # 打印进度
        if i % 10 == 0:
            print(f"  步骤 {i}: 位置 {current_pos[0]:.2f}, 碰撞数 {len(collisions)}")
    
    # 显示统计信息
    stats = collision_monitor.get_collision_statistics()
    print(f"\n最终统计:")
    print(f"  总检测碰撞数: {stats['total_collisions_detected']}")
    print(f"  总避让动作数: {stats['total_avoidance_actions']}")
    print(f"  最近碰撞数: {stats['recent_collisions_count']}")
    print(f"  平均碰撞严重程度: {stats['average_collision_severity']:.3f}")
    
    return trajectory_data, collision_data, avoidance_data


def plot_collision_analysis(trajectory_data: List, collision_data: List, avoidance_data: List):
    """绘制碰撞分析图表"""
    print("\n=== 生成碰撞分析图表 ===")
    
    try:
        # 提取数据
        steps = [d['step'] for d in trajectory_data]
        positions = np.array([d['positions'] for d in trajectory_data])
        num_collisions = [d['num_collisions'] for d in collision_data]
        max_severities = [d['max_severity'] for d in collision_data]
        min_distances = [d['min_distance'] if d['min_distance'] != float('inf') else 0 
                        for d in collision_data]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('碰撞检测与避让分析', fontsize=16)
        
        # 1. 关节位置轨迹
        ax1 = axes[0, 0]
        for i in range(6):
            ax1.plot(steps, positions[:, i], label=f'关节 {i+1}')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('关节位置 (rad)')
        ax1.set_title('关节位置轨迹')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 碰撞数量
        ax2 = axes[0, 1]
        ax2.plot(steps, num_collisions, 'r-', linewidth=2)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('碰撞数量')
        ax2.set_title('检测到的碰撞数量')
        ax2.grid(True)
        
        # 3. 碰撞严重程度
        ax3 = axes[1, 0]
        ax3.plot(steps, max_severities, 'orange', linewidth=2)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('最大严重程度')
        ax3.set_title('碰撞严重程度')
        ax3.set_ylim(0, 1)
        ax3.grid(True)
        
        # 4. 最小距离
        ax4 = axes[1, 1]
        ax4.plot(steps, min_distances, 'g-', linewidth=2)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('最小距离 (m)')
        ax4.set_title('最小碰撞距离')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('collision_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'collision_analysis.png'")
        
        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            print("无法显示图表（非交互环境）")
            
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")


def performance_benchmark():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    robot_model = create_demo_robot_model()
    collision_monitor = CollisionMonitor(robot_model)
    
    # 测试参数
    num_iterations = 1000
    
    print(f"运行 {num_iterations} 次碰撞检测...")
    
    # 性能测试
    times = []
    collision_counts = []
    
    start_total = time.time()
    
    for i in range(num_iterations):
        # 随机机器人状态
        robot_state = RobotState(
            joint_positions=np.random.uniform(-2, 2, 6),
            joint_velocities=np.random.uniform(-0.5, 0.5, 6),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=time.time()
        )
        
        desired_command = ControlCommand(
            joint_velocities=np.random.uniform(-0.2, 0.2, 6),
            control_mode="velocity"
        )
        
        # 计时
        start_iter = time.time()
        collisions, avoidance_command = collision_monitor.update(
            robot_state, desired_command
        )
        iter_time = time.time() - start_iter
        
        times.append(iter_time)
        collision_counts.append(len(collisions))
        
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  完成 {i + 1}/{num_iterations}")
    
    total_time = time.time() - start_total
    
    # 统计结果
    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    std_time = np.std(times)
    
    avg_collisions = np.mean(collision_counts)
    max_collisions = np.max(collision_counts)
    
    print(f"\n性能统计:")
    print(f"  总时间: {total_time:.2f} s")
    print(f"  平均每次: {avg_time*1000:.2f} ms")
    print(f"  最大时间: {max_time*1000:.2f} ms")
    print(f"  最小时间: {min_time*1000:.2f} ms")
    print(f"  标准差: {std_time*1000:.2f} ms")
    print(f"  频率: {1/avg_time:.1f} Hz")
    
    print(f"\n碰撞统计:")
    print(f"  平均碰撞数: {avg_collisions:.2f}")
    print(f"  最大碰撞数: {max_collisions}")
    
    # 最终监控器统计
    final_stats = collision_monitor.get_collision_statistics()
    print(f"\n监控器统计:")
    print(f"  总检测碰撞: {final_stats['total_collisions_detected']}")
    print(f"  总避让动作: {final_stats['total_avoidance_actions']}")


def main():
    """主演示函数"""
    print("机器人碰撞检测与避让算法演示")
    print("=" * 50)
    
    try:
        # 1. 距离计算演示
        demo_distance_calculations()
        
        # 2. 碰撞检测演示
        demo_collision_detection()
        
        # 3. 碰撞避让演示
        demo_collision_avoidance()
        
        # 4. 碰撞监控演示
        trajectory_data, collision_data, avoidance_data = demo_collision_monitor()
        
        # 5. 生成分析图表
        plot_collision_analysis(trajectory_data, collision_data, avoidance_data)
        
        # 6. 性能基准测试
        performance_benchmark()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()