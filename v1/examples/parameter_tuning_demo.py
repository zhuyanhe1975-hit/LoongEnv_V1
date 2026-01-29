#!/usr/bin/env python3
"""
参数自动调优演示

演示机器人运动控制系统的参数自动调优功能，包括：
1. 控制器增益调优
2. 轨迹规划参数调优
3. 抑振参数调优
4. 综合参数调优
5. 调优报告生成
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from pathlib import Path

from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    RobotState, TrajectoryPoint, Waypoint, PayloadInfo
)
from src.robot_motion_control.algorithms.parameter_tuning import (
    ParameterTuner, TuningReportGenerator, OptimizationConfig, 
    PerformanceWeights, ParameterBounds, OptimizationMethod, ParameterType
)


def create_sample_robot_model():
    """创建示例机器人模型"""
    try:
        # 尝试使用真实的机器人模型
        mjcf_path = "/home/yhzhu/LoongEnv/ER15-1400-mjcf/ER15-1400.xml"
        if os.path.exists(mjcf_path):
            robot_model = RobotModel.from_mjcf(mjcf_path)
            print(f"使用真实机器人模型: {mjcf_path}")
            return robot_model
    except Exception as e:
        print(f"无法加载真实机器人模型: {e}")
    
    # 创建简化的6自由度机器人模型
    print("使用简化的6自由度机器人模型")
    
    from src.robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
    
    # 创建动力学参数
    dynamics_params = DynamicsParameters(
        masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
        centers_of_mass=[[0.0, 0.0, 0.1]] * 6,
        inertias=[[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]] * 6,
        friction_coeffs=[0.1] * 6,
        gravity=[0.0, 0.0, -9.81]
    )
    
    # 创建运动学限制
    kinodynamic_limits = KinodynamicLimits(
        max_joint_positions=[np.pi] * 6,
        min_joint_positions=[-np.pi] * 6,
        max_joint_velocities=[2.0] * 6,
        max_joint_accelerations=[5.0] * 6,
        max_joint_jerks=[20.0] * 6,
        max_joint_torques=[100.0] * 6
    )
    
    robot_model = RobotModel(
        name="ER15-1400",
        n_joints=6,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )
    
    return robot_model


def create_test_trajectory():
    """创建测试轨迹"""
    # 创建简单的关节空间轨迹
    waypoints = []
    
    # 起始点
    waypoints.append(Waypoint(
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ))
    
    # 中间点
    waypoints.append(Waypoint(
        position=np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
    ))
    
    # 终点
    waypoints.append(Waypoint(
        position=np.array([1.0, 0.6, -0.4, 0.2, 0.8, -0.2])
    ))
    
    # 生成轨迹点
    trajectory = []
    n_points = 100
    
    for i in range(n_points):
        t = i / (n_points - 1)
        
        # 简单的线性插值
        if t <= 0.5:
            # 第一段
            alpha = t * 2
            pos = (1 - alpha) * waypoints[0].position + alpha * waypoints[1].position
        else:
            # 第二段
            alpha = (t - 0.5) * 2
            pos = (1 - alpha) * waypoints[1].position + alpha * waypoints[2].position
        
        # 计算速度和加速度（简化）
        vel = np.random.normal(0, 0.1, 6)  # 随机速度
        acc = np.random.normal(0, 0.05, 6)  # 随机加速度
        jerk = np.random.normal(0, 0.02, 6)  # 随机加加速度
        
        trajectory.append(TrajectoryPoint(
            position=pos,
            velocity=vel,
            acceleration=acc,
            jerk=jerk,
            time=t * 2.0,  # 总时间2秒
            path_parameter=t
        ))
    
    return trajectory


def create_test_scenarios(robot_model):
    """创建测试场景"""
    scenarios = []
    
    # 场景1：标准负载
    initial_state1 = RobotState(
        joint_positions=np.zeros(robot_model.n_joints),
        joint_velocities=np.zeros(robot_model.n_joints),
        joint_accelerations=np.zeros(robot_model.n_joints),
        joint_torques=np.zeros(robot_model.n_joints),
        end_effector_transform=np.eye(4),
        timestamp=0.0
    )
    
    scenarios.append({
        'name': '标准负载',
        'initial_state': initial_state1,
        'payload': PayloadInfo(
            mass=1.0,
            center_of_mass=[0.0, 0.0, 0.1],
            inertia=[[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
            identification_confidence=1.0
        )
    })
    
    # 场景2：重负载
    initial_state2 = RobotState(
        joint_positions=np.array([0.1, -0.1, 0.05, -0.05, 0.1, -0.1]),
        joint_velocities=np.zeros(robot_model.n_joints),
        joint_accelerations=np.zeros(robot_model.n_joints),
        joint_torques=np.zeros(robot_model.n_joints),
        end_effector_transform=np.eye(4),
        timestamp=0.0
    )
    
    scenarios.append({
        'name': '重负载',
        'initial_state': initial_state2,
        'payload': PayloadInfo(
            mass=5.0,
            center_of_mass=[0.0, 0.0, 0.2],
            inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            identification_confidence=0.9
        )
    })
    
    # 场景3：轻负载
    initial_state3 = RobotState(
        joint_positions=np.array([-0.1, 0.1, -0.05, 0.05, -0.1, 0.1]),
        joint_velocities=np.zeros(robot_model.n_joints),
        joint_accelerations=np.zeros(robot_model.n_joints),
        joint_torques=np.zeros(robot_model.n_joints),
        end_effector_transform=np.eye(4),
        timestamp=0.0
    )
    
    scenarios.append({
        'name': '轻负载',
        'initial_state': initial_state3,
        'payload': PayloadInfo(
            mass=0.2,
            center_of_mass=[0.0, 0.0, 0.05],
            inertia=[[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]],
            identification_confidence=0.95
        )
    })
    
    return scenarios


def demonstrate_control_gains_tuning(tuner, trajectory, scenarios):
    """演示控制器增益调优"""
    print("\n=== 控制器增益调优演示 ===")
    
    # 设置自定义参数边界
    parameter_bounds = {
        'kp': ParameterBounds(
            lower=np.ones(tuner.robot_model.n_joints) * 50.0,
            upper=np.ones(tuner.robot_model.n_joints) * 500.0
        ),
        'ki': ParameterBounds(
            lower=np.ones(tuner.robot_model.n_joints) * 1.0,
            upper=np.ones(tuner.robot_model.n_joints) * 50.0
        ),
        'kd': ParameterBounds(
            lower=np.ones(tuner.robot_model.n_joints) * 1.0,
            upper=np.ones(tuner.robot_model.n_joints) * 20.0
        )
    }
    
    # 执行调优
    result = tuner.tune_control_gains(trajectory, scenarios, parameter_bounds)
    
    # 显示结果
    print(f"调优成功: {result.success}")
    print(f"最优性能: {result.best_performance:.6f}")
    print(f"计算时间: {result.computation_time:.2f}s")
    print(f"迭代次数: {len(result.optimization_history)}")
    
    if result.success and result.optimal_parameters:
        print("\n最优参数:")
        for param_name, param_value in result.optimal_parameters.items():
            if isinstance(param_value, np.ndarray):
                print(f"  {param_name}: {param_value}")
            else:
                print(f"  {param_name}: {param_value}")
    
    return result


def demonstrate_trajectory_tuning(tuner, trajectory):
    """演示轨迹规划参数调优"""
    print("\n=== 轨迹规划参数调优演示 ===")
    
    # 创建测试路径
    test_paths = [[point for point in trajectory]]
    
    # 设置自定义参数边界
    parameter_bounds = {
        'max_velocity_scale': ParameterBounds(lower=0.5, upper=1.5),
        'max_acceleration_scale': ParameterBounds(lower=0.5, upper=1.5),
        'jerk_limit_scale': ParameterBounds(lower=0.5, upper=1.5),
        'smoothing_factor': ParameterBounds(lower=0.1, upper=0.9)
    }
    
    # 执行调优
    result = tuner.tune_trajectory_parameters(test_paths, parameter_bounds)
    
    # 显示结果
    print(f"调优成功: {result.success}")
    print(f"最优性能: {result.best_performance:.6f}")
    print(f"计算时间: {result.computation_time:.2f}s")
    print(f"迭代次数: {len(result.optimization_history)}")
    
    if result.success and result.optimal_parameters:
        print("\n最优参数:")
        for param_name, param_value in result.optimal_parameters.items():
            print(f"  {param_name}: {param_value}")
    
    return result


def demonstrate_vibration_tuning(tuner, trajectory):
    """演示抑振参数调优"""
    print("\n=== 抑振参数调优演示 ===")
    
    # 创建测试轨迹
    test_trajectories = [trajectory]
    
    # 设置自定义参数边界
    parameter_bounds = {
        'damping_ratio': ParameterBounds(lower=0.5, upper=1.5),
        'natural_frequency': ParameterBounds(lower=10.0, upper=50.0),
        'filter_order': ParameterBounds(lower=2, upper=4),
        'shaper_amplitude': ParameterBounds(lower=0.5, upper=1.0)
    }
    
    # 执行调优
    result = tuner.tune_vibration_parameters(test_trajectories, parameter_bounds)
    
    # 显示结果
    print(f"调优成功: {result.success}")
    print(f"最优性能: {result.best_performance:.6f}")
    print(f"计算时间: {result.computation_time:.2f}s")
    print(f"迭代次数: {len(result.optimization_history)}")
    
    if result.success and result.optimal_parameters:
        print("\n最优参数:")
        for param_name, param_value in result.optimal_parameters.items():
            print(f"  {param_name}: {param_value}")
    
    return result


def demonstrate_comprehensive_tuning(tuner, trajectory, scenarios):
    """演示综合参数调优"""
    print("\n=== 综合参数调优演示 ===")
    
    # 选择要调优的参数类型
    parameter_types = [
        ParameterType.CONTROL_GAINS,
        ParameterType.TRAJECTORY_PARAMS,
        ParameterType.VIBRATION_PARAMS
    ]
    
    # 执行综合调优
    results = tuner.comprehensive_tuning(trajectory, scenarios, parameter_types)
    
    # 显示结果
    print(f"调优了 {len(results)} 类参数:")
    
    for param_type, result in results.items():
        print(f"\n{param_type.value}:")
        print(f"  成功: {result.success}")
        print(f"  性能: {result.best_performance:.6f}")
        print(f"  时间: {result.computation_time:.2f}s")
        
        if result.success:
            print(f"  评估指标: {result.evaluation_metrics}")
    
    return results


def demonstrate_report_generation(results, robot_model, config, performance_weights):
    """演示报告生成"""
    print("\n=== 调优报告生成演示 ===")
    
    # 创建报告生成器
    report_generator = TuningReportGenerator("tuning_reports")
    
    # 准备参数边界信息（简化）
    parameter_bounds = {
        'control_gains': {
            'kp': ParameterBounds(
                lower=np.ones(robot_model.n_joints) * 50.0,
                upper=np.ones(robot_model.n_joints) * 500.0
            )
        }
    }
    
    # 生成报告
    report = report_generator.generate_report(
        results, robot_model, config, performance_weights, parameter_bounds
    )
    
    print(f"报告生成时间: {report.timestamp}")
    print(f"总体性能提升: {report.overall_performance_improvement:.2f}%")
    print(f"生成了 {len(report.plots_paths)} 个图表")
    print(f"提供了 {len(report.recommendations)} 条建议")
    
    print("\n优化建议:")
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    return report


def main():
    """主函数"""
    print("机器人参数自动调优演示")
    print("=" * 50)
    
    try:
        # 创建机器人模型
        print("创建机器人模型...")
        robot_model = create_sample_robot_model()
        print(f"机器人关节数: {robot_model.n_joints}")
        
        # 创建测试数据
        print("创建测试轨迹和场景...")
        trajectory = create_test_trajectory()
        scenarios = create_test_scenarios(robot_model)
        print(f"轨迹点数: {len(trajectory)}")
        print(f"测试场景数: {len(scenarios)}")
        
        # 配置优化参数
        config = OptimizationConfig(
            method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
            max_iterations=20,  # 减少迭代次数以加快演示
            population_size=8,
            parallel_workers=2,
            verbose=True
        )
        
        # 配置性能权重
        performance_weights = PerformanceWeights(
            tracking_accuracy=0.5,
            settling_time=0.2,
            overshoot=0.15,
            energy_efficiency=0.1,
            vibration_suppression=0.05,
            safety_margin=0.0
        )
        
        # 创建参数调优器
        print("创建参数调优器...")
        tuner = ParameterTuner(robot_model, config, performance_weights)
        
        # 演示各种调优功能
        results = {}
        
        # 1. 控制器增益调优
        try:
            control_result = demonstrate_control_gains_tuning(tuner, trajectory, scenarios)
            results[ParameterType.CONTROL_GAINS] = control_result
        except Exception as e:
            print(f"控制器增益调优失败: {e}")
        
        # 2. 轨迹规划参数调优
        try:
            trajectory_result = demonstrate_trajectory_tuning(tuner, trajectory)
            results[ParameterType.TRAJECTORY_PARAMS] = trajectory_result
        except Exception as e:
            print(f"轨迹规划参数调优失败: {e}")
        
        # 3. 抑振参数调优
        try:
            vibration_result = demonstrate_vibration_tuning(tuner, trajectory)
            results[ParameterType.VIBRATION_PARAMS] = vibration_result
        except Exception as e:
            print(f"抑振参数调优失败: {e}")
        
        # 4. 综合调优（如果单独调优成功）
        if len(results) >= 2:
            try:
                comprehensive_results = demonstrate_comprehensive_tuning(tuner, trajectory, scenarios)
                # 更新结果
                results.update(comprehensive_results)
            except Exception as e:
                print(f"综合调优失败: {e}")
        
        # 5. 生成调优报告
        if results:
            try:
                report = demonstrate_report_generation(
                    results, robot_model, config, performance_weights
                )
                print(f"\n报告已保存到: tuning_reports/")
            except Exception as e:
                print(f"报告生成失败: {e}")
        
        print("\n=== 演示完成 ===")
        print("参数自动调优功能演示成功完成！")
        
        # 显示最终统计
        successful_tunings = sum(1 for result in results.values() if result.success)
        print(f"成功调优: {successful_tunings}/{len(results)} 类参数")
        
        if successful_tunings > 0:
            avg_improvement = np.mean([
                (1.0 - result.best_performance) * 100 
                for result in results.values() 
                if result.success and result.best_performance < 1.0
            ])
            print(f"平均性能提升: {avg_improvement:.2f}%")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()