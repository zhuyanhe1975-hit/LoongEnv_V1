#!/usr/bin/env python3
"""
诊断参数调优崩溃问题

运行简化的调优测试，捕获详细的错误信息
"""

import sys
import os
from pathlib import Path
import traceback
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tuning_locally():
    """本地测试调优功能"""
    
    print("=" * 60)
    print("诊断参数调优崩溃问题")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        print("\n[步骤1] 导入模块...")
        from src.robot_motion_control.core.models import RobotModel
        from src.robot_motion_control.core.types import (
            DynamicsParameters, KinodynamicLimits, RobotState, Waypoint
        )
        from src.robot_motion_control.algorithms.parameter_tuning import (
            ParameterTuner, OptimizationConfig, PerformanceWeights, ParameterType
        )
        from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner
        print("✓ 模块导入成功")
        
        # 创建机器人模型
        print("\n[步骤2] 创建机器人模型...")
        dynamics_params = DynamicsParameters(
            masses=[54.52, 11.11, 25.03, 10.81, 4.48, 0.28],
            centers_of_mass=[
                [0.09835, -0.02908, -0.0995],
                [0.25263, -0.00448, 0.15471],
                [0.03913, -0.02495, 0.03337],
                [-0.00132, -0.0012, -0.30035],
                [0.0004, -0.03052, 0.01328],
                [0, 0, 0]
            ],
            inertias=[
                [[1.16916852, 0.0865367, -0.47354118],
                 [0.0865367, 1.39934751, 0.11859959],
                 [-0.47354118, 0.11859959, 1.00920236]],
                [[0.04507715, -0.00764148, -0.01800527],
                 [-0.00764148, 0.58269106, 0.00057833],
                 [-0.01800527, 0.00057833, 0.60235638]],
                [[0.33717585, 0.06955124, 0.00142677],
                 [0.06955124, 0.38576036, -0.00313441],
                 [0.00142677, -0.00313441, 0.24095087]],
                [[0.28066314, -0.00003381, 0.00084678],
                 [-0.00003381, 0.27142738, 0.00437676],
                 [0.00084678, 0.00437676, 0.04425281]],
                [[0.01710138, -0.00002606, 0.00000867],
                 [-0.00002606, 0.01098115, -0.00175535],
                 [0.00000867, -0.00175535, 0.01408541]],
                [[0.0001346961, 0.0000076, -0.00000827],
                 [0.0000076, 0.0001645611, 0.000118982],
                 [-0.00000827, 0.000118982, 0.001539171]]
            ],
            friction_coeffs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[2.967, 1.5708, 3.0543, 3.316, 2.2689, 6.2832],
            min_joint_positions=[-2.967, -2.7925, -1.4835, -3.316, -2.2689, -6.2832],
            max_joint_velocities=[3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
            max_joint_accelerations=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            max_joint_jerks=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            max_joint_torques=[100.0, 100.0, 100.0, 50.0, 50.0, 25.0]
        )
        
        robot_model = RobotModel(
            name="ER15-1400",
            n_joints=6,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits,
            urdf_path=str(project_root / "models" / "ER15-1400.urdf")
        )
        print("✓ 机器人模型创建成功")
        
        # 创建参数调优器
        print("\n[步骤3] 创建参数调优器...")
        config = OptimizationConfig(
            max_iterations=5,  # 减少迭代次数
            population_size=3,  # 减少种群大小
            verbose=True
        )
        weights = PerformanceWeights()
        parameter_tuner = ParameterTuner(robot_model, config, weights)
        print("✓ 参数调优器创建成功")
        
        # 创建测试轨迹
        print("\n[步骤4] 创建测试轨迹...")
        test_waypoints = [
            Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            Waypoint(position=np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])),
            Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        ]
        
        trajectory_planner = TrajectoryPlanner(robot_model)
        reference_trajectory = trajectory_planner.interpolate_s7_trajectory(test_waypoints)
        print(f"✓ 测试轨迹创建成功，共 {len(reference_trajectory)} 个点")
        
        # 创建测试场景
        print("\n[步骤5] 创建测试场景...")
        test_scenarios = [
            {"initial_state": RobotState(
                joint_positions=np.zeros(6),
                joint_velocities=np.zeros(6),
                joint_accelerations=np.zeros(6),
                joint_torques=np.zeros(6),
                end_effector_transform=np.eye(4),
                timestamp=0.0
            )}
        ]
        print("✓ 测试场景创建成功")
        
        # 第一次调优
        print("\n[步骤6] 执行第一次参数调优...")
        print("=" * 60)
        try:
            results1 = parameter_tuner.comprehensive_tuning(
                reference_trajectory, 
                test_scenarios, 
                [ParameterType.CONTROL_GAINS]
            )
            print("=" * 60)
            print("✓ 第一次调优完成")
            for param_type, result in results1.items():
                print(f"  {param_type.value}: success={result.success}, score={result.best_performance:.6f}")
        except Exception as e:
            print("=" * 60)
            print(f"✗ 第一次调优失败: {e}")
            traceback.print_exc()
            return False
        
        # 重置调优器状态
        print("\n[步骤7] 重置调优器状态...")
        parameter_tuner.optimization_history = {}
        parameter_tuner.evaluation_count = 0
        parameter_tuner.baseline_performance = None
        parameter_tuner._path_controller = None
        parameter_tuner._trajectory_planner = None
        print("✓ 调优器状态已重置")
        
        # 第二次调优
        print("\n[步骤8] 执行第二次参数调优...")
        print("=" * 60)
        try:
            results2 = parameter_tuner.comprehensive_tuning(
                reference_trajectory, 
                test_scenarios, 
                [ParameterType.CONTROL_GAINS]
            )
            print("=" * 60)
            print("✓ 第二次调优完成")
            for param_type, result in results2.items():
                print(f"  {param_type.value}: success={result.success}, score={result.best_performance:.6f}")
        except Exception as e:
            print("=" * 60)
            print(f"✗ 第二次调优失败: {e}")
            traceback.print_exc()
            return False
        
        print("\n" + "=" * 60)
        print("✓ 诊断完成：两次调优都成功")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 诊断过程出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tuning_locally()
    exit(0 if success else 1)
