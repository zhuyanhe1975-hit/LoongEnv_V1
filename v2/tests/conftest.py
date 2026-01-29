"""
pytest配置文件

定义测试夹具和共享配置，为单元测试和属性测试提供支持。
"""

import pytest
import numpy as np
from hypothesis import settings, Verbosity

from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, RobotState,
    TrajectoryPoint, PayloadInfo
)


# Hypothesis配置
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("fast", max_examples=10, verbosity=Verbosity.quiet)
settings.register_profile("thorough", max_examples=1000, verbosity=Verbosity.verbose)

# 根据环境变量选择配置
import os
profile = os.getenv("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)


@pytest.fixture
def sample_robot_model():
    """创建示例机器人模型"""
    n_joints = 6
    
    dynamics_params = DynamicsParameters(
        masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
        centers_of_mass=[[0.0, 0.0, 0.1]] * n_joints,
        inertias=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * n_joints,
        friction_coeffs=[0.1] * n_joints,
        gravity=[0.0, 0.0, -9.81]
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
        name="test_robot",
        n_joints=n_joints,
        dynamics_params=dynamics_params,
        kinodynamic_limits=kinodynamic_limits
    )


@pytest.fixture
def sample_robot_state():
    """创建示例机器人状态"""
    n_joints = 6
    
    return RobotState(
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        joint_velocities=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        joint_accelerations=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        joint_torques=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        end_effector_transform=np.eye(4),
        timestamp=1.0
    )


@pytest.fixture
def sample_trajectory_point():
    """创建示例轨迹点"""
    n_joints = 6
    
    return TrajectoryPoint(
        position=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        velocity=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        acceleration=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        jerk=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        time=1.0,
        path_parameter=0.5
    )


@pytest.fixture
def sample_payload_info():
    """创建示例负载信息"""
    return PayloadInfo(
        mass=2.0,
        center_of_mass=[0.0, 0.0, 0.1],
        inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.05]],
        identification_confidence=0.9
    )