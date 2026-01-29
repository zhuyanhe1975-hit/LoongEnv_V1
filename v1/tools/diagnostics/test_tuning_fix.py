#!/usr/bin/env python3
"""
测试参数调优修复效果

验证修复后的调优功能是否能产生不同的性能评估结果。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, RobotState, Waypoint
)
from src.robot_motion_control.algorithms.parameter_tuning import (
    ParameterTuner, OptimizationConfig, PerformanceWeights,
    ParameterType, OptimizationMethod
)
from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner


def create_robot_model():
    """创建机器人模型"""
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
    
    return RobotModel(
        name="ER15-1400",
        n_joints=6,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


def test_performance_variation():
    """测试不同参数是否产生不同的性能评估"""
    print("=" * 60)
    print("测试参数调优修复效果")
    print("=" * 60)
    
    # 创建机器人模型
    print("\n[步骤1] 创建机器人模型...")
    robot_model = create_robot_model()
    print("✓ 机器人模型创建成功")
    
    # 创建参数调优器
    print("\n[步骤2] 创建参数调优器...")
    tuner = ParameterTuner(robot_model)
    print("✓ 参数调优器创建成功")
    
    # 创建测试轨迹
    print("\n[步骤3] 创建测试轨迹...")
    test_waypoints = [
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([1.0, -0.6, 0.4, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])),
        Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ]
    
    trajectory_planner = TrajectoryPlanner(robot_model)
    reference_trajectory = trajectory_planner.interpolate_s7_trajectory(test_waypoints)
    print(f"✓ 参考轨迹生成完成，共 {len(reference_trajectory)} 个点")
    
    # 创建测试场景
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
    
    # 测试不同的PID参数
    print("\n[步骤4] 测试不同PID参数的性能...")
    
    test_params = [
        {
            'name': '低增益',
            'kp': np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
            'ki': np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            'kd': np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        },
        {
            'name': '中等增益',
            'kp': np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0]),
            'ki': np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
            'kd': np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
        },
        {
            'name': '高增益',
            'kp': np.array([500.0, 500.0, 500.0, 500.0, 500.0, 500.0]),
            'ki': np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
            'kd': np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
        },
        {
            'name': '不平衡增益',
            'kp': np.array([100.0, 300.0, 200.0, 400.0, 150.0, 250.0]),
            'ki': np.array([10.0, 30.0, 20.0, 40.0, 15.0, 25.0]),
            'kd': np.array([10.0, 20.0, 15.0, 25.0, 12.0, 18.0])
        }
    ]
    
    performances = []
    
    for params in test_params:
        # 评估性能
        score = tuner._evaluate_control_performance(
            params,
            reference_trajectory,
            test_scenarios
        )
        performances.append(score)
        print(f"  {params['name']:12s}: 性能分数 = {score:.6f}")
    
    # 分析结果
    print("\n[步骤5] 分析结果...")
    unique_scores = len(set(performances))
    score_range = max(performances) - min(performances)
    score_std = np.std(performances)
    
    print(f"  唯一分数数量: {unique_scores}")
    print(f"  分数范围: {score_range:.6f}")
    print(f"  标准差: {score_std:.6f}")
    
    # 判断修复是否成功
    print("\n" + "=" * 60)
    if unique_scores >= 3 and score_range > 0.01:
        print("✅ 修复成功！")
        print("   - 不同参数产生不同的性能评估")
        print("   - 优化器现在可以区分参数的好坏")
        print("   - 调优功能应该能正常工作")
    elif unique_scores >= 2:
        print("⚠️  部分成功")
        print("   - 有一些性能差异，但可能不够明显")
        print("   - 建议增加参数变化范围或调整动力学模型")
    else:
        print("❌ 修复失败")
        print("   - 所有参数的性能仍然相同")
        print("   - 需要进一步调查问题")
    print("=" * 60)
    
    return unique_scores >= 3 and score_range > 0.01


if __name__ == '__main__':
    try:
        success = test_performance_variation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
