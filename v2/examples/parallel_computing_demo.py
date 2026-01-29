#!/usr/bin/env python3
"""
并行计算优化演示

展示机器人运动控制系统中多线程并行计算优化的功能和性能提升。
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from robot_motion_control.core.controller import RobotMotionController, ControllerConfig
from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.parallel_computing import ParallelConfig, ParallelMode
from robot_motion_control.core.types import Waypoint, RobotState, TrajectoryPoint
from robot_motion_control.algorithms.parallel_dynamics import ParallelOptimizedDynamicsEngine
from robot_motion_control.algorithms.parallel_trajectory_planning import ParallelOptimizedTrajectoryPlanner
from robot_motion_control.algorithms.parallel_path_control import ParallelOptimizedPathController


def create_test_robot_model() -> RobotModel:
    """创建测试用机器人模型"""
    print("创建ER15-1400机器人模型...")
    return RobotModel.create_test_model(n_joints=6)


def generate_test_trajectory(robot_model: RobotModel, n_points: int = 50) -> List[Waypoint]:
    """生成测试轨迹"""
    print(f"生成包含{n_points}个点的测试轨迹...")
    
    waypoints = []
    n_joints = robot_model.n_joints
    
    for i in range(n_points):
        # 生成平滑的正弦轨迹
        t = i / (n_points - 1) * 2 * np.pi
        
        position = np.array([
            0.5 * np.sin(t),
            0.3 * np.cos(t),
            0.2 * np.sin(2 * t),
            0.4 * np.cos(1.5 * t),
            0.1 * np.sin(3 * t),
            0.2 * np.cos(2.5 * t)
        ])
        
        waypoint = Waypoint(
            position=position,
            velocity=np.zeros(n_joints),
            time_constraint=i * 0.01
        )
        waypoints.append(waypoint)
    
    return waypoints


def benchmark_dynamics_performance(robot_model: RobotModel) -> Dict[str, Any]:
    """基准测试动力学计算性能"""
    print("\n=== 动力学计算性能基准测试 ===")
    
    # 创建标准动力学引擎
    from src.robot_motion_control.algorithms.dynamics import DynamicsEngine
    standard_engine = DynamicsEngine(robot_model)
    
    # 创建并行优化动力学引擎
    parallel_config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=4)
    parallel_engine = ParallelOptimizedDynamicsEngine(robot_model, parallel_config)
    
    n_joints = robot_model.n_joints
    test_sizes = [5, 10, 20, 50, 100]
    results = {
        'test_sizes': test_sizes,
        'sequential_times': [],
        'parallel_times': [],
        'speedups': []
    }
    
    for size in test_sizes:
        print(f"\n测试批处理大小: {size}")
        
        # 生成测试数据
        q_list = [np.random.randn(n_joints) for _ in range(size)]
        qd_list = [np.random.randn(n_joints) for _ in range(size)]
        tau_list = [np.random.randn(n_joints) for _ in range(size)]
        
        # 顺序计算
        start_time = time.time()
        for q, qd, tau in zip(q_list, qd_list, tau_list):
            standard_engine.forward_dynamics(q, qd, tau)
        sequential_time = time.time() - start_time
        
        # 并行计算
        start_time = time.time()
        parallel_engine.batch_forward_dynamics(q_list, qd_list, tau_list)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"  顺序计算时间: {sequential_time:.4f}s")
        print(f"  并行计算时间: {parallel_time:.4f}s")
        print(f"  加速比: {speedup:.2f}x")
        
        results['sequential_times'].append(sequential_time)
        results['parallel_times'].append(parallel_time)
        results['speedups'].append(speedup)
    
    return results


def benchmark_trajectory_planning_performance(robot_model: RobotModel) -> Dict[str, Any]:
    """基准测试轨迹规划性能"""
    print("\n=== 轨迹规划性能基准测试 ===")
    
    # 创建标准轨迹规划器
    from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner
    standard_planner = TrajectoryPlanner(robot_model)
    
    # 创建并行优化轨迹规划器
    parallel_config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=4)
    parallel_planner = ParallelOptimizedTrajectoryPlanner(robot_model, parallel_config)
    
    trajectory_sizes = [10, 20, 50, 100]
    results = {
        'trajectory_sizes': trajectory_sizes,
        'sequential_times': [],
        'parallel_times': [],
        'speedups': []
    }
    
    for size in trajectory_sizes:
        print(f"\n测试轨迹大小: {size}个点")
        
        # 生成测试轨迹
        waypoints = generate_test_trajectory(robot_model, size)
        
        # 顺序S7插补
        start_time = time.time()
        standard_planner.interpolate_s7_trajectory(waypoints)
        sequential_time = time.time() - start_time
        
        # 并行S7插补
        start_time = time.time()
        parallel_planner.parallel_interpolate_s7_trajectory(waypoints)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"  顺序插补时间: {sequential_time:.4f}s")
        print(f"  并行插补时间: {parallel_time:.4f}s")
        print(f"  加速比: {speedup:.2f}x")
        
        results['sequential_times'].append(sequential_time)
        results['parallel_times'].append(parallel_time)
        results['speedups'].append(speedup)
    
    return results


def benchmark_path_control_performance(robot_model: RobotModel) -> Dict[str, Any]:
    """基准测试路径控制性能"""
    print("\n=== 路径控制性能基准测试 ===")
    
    # 创建标准路径控制器
    from src.robot_motion_control.algorithms.path_control import PathController
    standard_controller = PathController(robot_model)
    
    # 创建并行优化路径控制器
    parallel_config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=4)
    parallel_controller = ParallelOptimizedPathController(robot_model, parallel_config=parallel_config)
    
    batch_sizes = [5, 10, 20, 50]
    results = {
        'batch_sizes': batch_sizes,
        'sequential_times': [],
        'parallel_times': [],
        'speedups': []
    }
    
    n_joints = robot_model.n_joints
    
    for size in batch_sizes:
        print(f"\n测试批处理大小: {size}")
        
        # 生成测试数据
        reference_trajectory = []
        current_states = []
        
        for i in range(size):
            ref_point = TrajectoryPoint(
                position=np.random.randn(n_joints),
                velocity=np.random.randn(n_joints) * 0.1,
                acceleration=np.random.randn(n_joints) * 0.01,
                jerk=np.zeros(n_joints),
                time=i * 0.001,
                path_parameter=i / (size - 1)
            )
            reference_trajectory.append(ref_point)
            
            state = RobotState(
                joint_positions=np.random.randn(n_joints),
                joint_velocities=np.random.randn(n_joints) * 0.1,
                joint_accelerations=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
                timestamp=i * 0.001
            )
            current_states.append(state)
        
        # 顺序控制计算
        start_time = time.time()
        for ref, state in zip(reference_trajectory, current_states):
            standard_controller.compute_control(ref, state)
        sequential_time = time.time() - start_time
        
        # 并行控制计算
        start_time = time.time()
        parallel_controller.batch_compute_control(reference_trajectory, current_states)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"  顺序控制时间: {sequential_time:.4f}s")
        print(f"  并行控制时间: {parallel_time:.4f}s")
        print(f"  加速比: {speedup:.2f}x")
        
        results['sequential_times'].append(sequential_time)
        results['parallel_times'].append(parallel_time)
        results['speedups'].append(speedup)
    
    return results


def demonstrate_integrated_parallel_controller(robot_model: RobotModel):
    """演示集成并行控制器"""
    print("\n=== 集成并行控制器演示 ===")
    
    # 创建启用并行计算的控制器
    config = ControllerConfig(
        enable_parallel_computing=True,
        parallel_mode=ParallelMode.THREAD,
        max_parallel_workers=4,
        parallel_batch_threshold=8
    )
    
    controller = RobotMotionController(robot_model, config)
    
    # 生成测试轨迹
    waypoints = generate_test_trajectory(robot_model, 30)
    
    print("规划轨迹...")
    start_time = time.time()
    trajectory = controller.plan_trajectory(waypoints, optimize_time=False)
    planning_time = time.time() - start_time
    
    print(f"轨迹规划完成，耗时: {planning_time:.4f}s")
    print(f"生成轨迹点数: {len(trajectory)}")
    
    # 模拟控制执行
    print("\n模拟控制执行...")
    initial_state = RobotState(
        joint_positions=np.zeros(robot_model.n_joints),
        joint_velocities=np.zeros(robot_model.n_joints),
        joint_accelerations=np.zeros(robot_model.n_joints),
        joint_torques=np.zeros(robot_model.n_joints),
        timestamp=0.0
    )
    
    start_time = time.time()
    control_commands = []
    current_state = initial_state
    
    for i, trajectory_point in enumerate(trajectory[:20]):  # 限制执行点数
        command = controller.compute_control(current_state, trajectory_point.time)
        control_commands.append(command)
        
        # 简化的状态更新
        if hasattr(command, 'joint_positions') and command.joint_positions is not None:
            current_state.joint_positions = command.joint_positions.copy()
        current_state.timestamp = trajectory_point.time
    
    execution_time = time.time() - start_time
    
    print(f"控制执行完成，耗时: {execution_time:.4f}s")
    print(f"执行控制指令数: {len(control_commands)}")
    
    # 获取性能报告
    status = controller.get_controller_status()
    if "parallel_performance" in status:
        print("\n并行性能统计:")
        parallel_perf = status["parallel_performance"]
        
        if "dynamics_performance" in parallel_perf:
            dynamics_perf = parallel_perf["dynamics_performance"]
            print(f"  动力学并行调用: {dynamics_perf.get('parallel_calls', 0)}")
            print(f"  动力学平均加速比: {dynamics_perf.get('avg_speedup', 1.0):.2f}x")
        
        if "trajectory_planner_performance" in parallel_perf:
            traj_perf = parallel_perf["trajectory_planner_performance"]
            print(f"  轨迹规划并行调用: {traj_perf.get('parallel_topp_calls', 0)}")
        
        if "path_controller_performance" in parallel_perf:
            ctrl_perf = parallel_perf["path_controller_performance"]
            print(f"  路径控制并行调用: {ctrl_perf.get('batch_processing_calls', 0)}")
            print(f"  路径控制平均加速比: {ctrl_perf.get('average_speedup', 1.0):.2f}x")


def demonstrate_performance_optimization(robot_model: RobotModel):
    """演示性能优化功能"""
    print("\n=== 性能优化演示 ===")
    
    # 创建并行控制器
    config = ControllerConfig(
        enable_parallel_computing=True,
        parallel_mode=ParallelMode.THREAD,
        max_parallel_workers=4
    )
    
    controller = RobotMotionController(robot_model, config)
    
    # 生成样本数据
    sample_trajectories = []
    sample_states = []
    
    for _ in range(3):
        waypoints = generate_test_trajectory(robot_model, 15)
        trajectory = controller.plan_trajectory(waypoints, optimize_time=False)
        sample_trajectories.append(trajectory)
        
        state = RobotState(
            joint_positions=np.random.randn(robot_model.n_joints),
            joint_velocities=np.zeros(robot_model.n_joints),
            joint_accelerations=np.zeros(robot_model.n_joints),
            joint_torques=np.zeros(robot_model.n_joints),
            timestamp=0.0
        )
        sample_states.append(state)
    
    print("运行性能优化...")
    start_time = time.time()
    controller.optimize_parallel_performance(
        sample_trajectories, sample_states, optimization_iterations=3
    )
    optimization_time = time.time() - start_time
    
    print(f"性能优化完成，耗时: {optimization_time:.4f}s")
    
    # 显示优化后的配置
    status = controller.get_controller_status()
    if "parallel_performance" in status:
        print("\n优化后的配置:")
        parallel_perf = status["parallel_performance"]
        config_info = parallel_perf.get("parallel_config", {})
        print(f"  并行模式: {config_info.get('mode', 'unknown')}")
        print(f"  工作线程数: {config_info.get('max_workers', 'unknown')}")
        print(f"  内存优化: {config_info.get('memory_optimization', 'unknown')}")


def plot_performance_results(
    dynamics_results: Dict[str, Any],
    trajectory_results: Dict[str, Any],
    control_results: Dict[str, Any]
):
    """绘制性能结果图表"""
    print("\n生成性能对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('并行计算性能对比', fontsize=16)
    
    # 动力学计算性能
    ax1 = axes[0, 0]
    ax1.plot(dynamics_results['test_sizes'], dynamics_results['sequential_times'], 
             'b-o', label='顺序计算')
    ax1.plot(dynamics_results['test_sizes'], dynamics_results['parallel_times'], 
             'r-s', label='并行计算')
    ax1.set_xlabel('批处理大小')
    ax1.set_ylabel('计算时间 (s)')
    ax1.set_title('动力学计算性能')
    ax1.legend()
    ax1.grid(True)
    
    # 轨迹规划性能
    ax2 = axes[0, 1]
    ax2.plot(trajectory_results['trajectory_sizes'], trajectory_results['sequential_times'], 
             'b-o', label='顺序计算')
    ax2.plot(trajectory_results['trajectory_sizes'], trajectory_results['parallel_times'], 
             'r-s', label='并行计算')
    ax2.set_xlabel('轨迹点数')
    ax2.set_ylabel('计算时间 (s)')
    ax2.set_title('轨迹规划性能')
    ax2.legend()
    ax2.grid(True)
    
    # 路径控制性能
    ax3 = axes[1, 0]
    ax3.plot(control_results['batch_sizes'], control_results['sequential_times'], 
             'b-o', label='顺序计算')
    ax3.plot(control_results['batch_sizes'], control_results['parallel_times'], 
             'r-s', label='并行计算')
    ax3.set_xlabel('批处理大小')
    ax3.set_ylabel('计算时间 (s)')
    ax3.set_title('路径控制性能')
    ax3.legend()
    ax3.grid(True)
    
    # 加速比对比
    ax4 = axes[1, 1]
    ax4.plot(dynamics_results['test_sizes'], dynamics_results['speedups'], 
             'g-o', label='动力学计算')
    ax4.plot(trajectory_results['trajectory_sizes'], trajectory_results['speedups'], 
             'b-s', label='轨迹规划')
    ax4.plot(control_results['batch_sizes'], control_results['speedups'], 
             'r-^', label='路径控制')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='基准线')
    ax4.set_xlabel('任务大小')
    ax4.set_ylabel('加速比')
    ax4.set_title('并行计算加速比')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('parallel_computing_performance.png', dpi=300, bbox_inches='tight')
    print("性能图表已保存为 'parallel_computing_performance.png'")
    
    # 显示图表（如果在交互环境中）
    try:
        plt.show()
    except:
        pass


def main():
    """主演示函数"""
    print("机器人运动控制系统 - 并行计算优化演示")
    print("=" * 50)
    
    # 创建机器人模型
    robot_model = create_test_robot_model()
    
    # 性能基准测试
    dynamics_results = benchmark_dynamics_performance(robot_model)
    trajectory_results = benchmark_trajectory_planning_performance(robot_model)
    control_results = benchmark_path_control_performance(robot_model)
    
    # 集成控制器演示
    demonstrate_integrated_parallel_controller(robot_model)
    
    # 性能优化演示
    demonstrate_performance_optimization(robot_model)
    
    # 生成性能图表
    plot_performance_results(dynamics_results, trajectory_results, control_results)
    
    # 总结
    print("\n" + "=" * 50)
    print("并行计算优化演示完成")
    print("\n主要成果:")
    print("1. 实现了多线程并行计算优化")
    print("2. 支持动力学、轨迹规划、路径控制的并行化")
    print("3. 提供了自动性能调优功能")
    print("4. 实现了内存访问优化")
    print("5. 集成了性能监控和分析工具")
    
    # 显示平均性能提升
    avg_dynamics_speedup = np.mean(dynamics_results['speedups'])
    avg_trajectory_speedup = np.mean(trajectory_results['speedups'])
    avg_control_speedup = np.mean(control_results['speedups'])
    
    print(f"\n平均性能提升:")
    print(f"  动力学计算: {avg_dynamics_speedup:.2f}x")
    print(f"  轨迹规划: {avg_trajectory_speedup:.2f}x")
    print(f"  路径控制: {avg_control_speedup:.2f}x")
    
    overall_speedup = (avg_dynamics_speedup + avg_trajectory_speedup + avg_control_speedup) / 3
    print(f"  整体平均: {overall_speedup:.2f}x")


if __name__ == "__main__":
    main()