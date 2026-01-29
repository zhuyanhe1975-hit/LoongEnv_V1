"""
核心数据类型测试

测试核心数据结构的创建、验证和操作功能。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from robot_motion_control.core.types import (
    RobotState, TrajectoryPoint, DynamicsParameters,
    KinodynamicLimits, PayloadInfo, ControlCommand,
    Waypoint, PerformanceMetrics
)


class TestRobotState:
    """机器人状态测试"""
    
    def test_robot_state_creation(self, sample_robot_state):
        """测试机器人状态创建"""
        state = sample_robot_state
        
        assert len(state.joint_positions) == 6
        assert len(state.joint_velocities) == 6
        assert len(state.joint_accelerations) == 6
        assert len(state.joint_torques) == 6
        assert state.end_effector_transform.shape == (4, 4)
        assert state.timestamp == 1.0
    
    def test_robot_state_dimension_validation(self):
        """测试机器人状态维度验证"""
        with pytest.raises(AssertionError):
            RobotState(
                joint_positions=np.array([1, 2, 3]),
                joint_velocities=np.array([1, 2]),  # 维度不匹配
                joint_accelerations=np.array([1, 2, 3]),
                joint_torques=np.array([1, 2, 3]),
                end_effector_transform=np.eye(4),
                timestamp=0.0
            )


class TestTrajectoryPoint:
    """轨迹点测试"""
    
    def test_trajectory_point_creation(self, sample_trajectory_point):
        """测试轨迹点创建"""
        point = sample_trajectory_point
        
        assert len(point.position) == 6
        assert len(point.velocity) == 6
        assert len(point.acceleration) == 6
        assert len(point.jerk) == 6
        assert point.time == 1.0
        assert point.path_parameter == 0.5
    
    def test_path_parameter_validation(self):
        """测试路径参数验证"""
        with pytest.raises(AssertionError):
            TrajectoryPoint(
                position=np.array([1, 2, 3]),
                velocity=np.array([1, 2, 3]),
                acceleration=np.array([1, 2, 3]),
                jerk=np.array([1, 2, 3]),
                time=1.0,
                path_parameter=1.5  # 超出范围
            )


class TestDynamicsParameters:
    """动力学参数测试"""
    
    def test_dynamics_parameters_creation(self):
        """测试动力学参数创建"""
        params = DynamicsParameters(
            masses=[1.0, 2.0, 3.0],
            centers_of_mass=[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
            inertias=[
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
            ],
            friction_coeffs=[0.1, 0.2, 0.3]
        )
        
        assert len(params.masses) == 3
        assert len(params.centers_of_mass) == 3
        assert len(params.inertias) == 3
        assert len(params.friction_coeffs) == 3
        assert params.gravity == [0.0, 0.0, -9.81]
    
    def test_com_dimension_validation(self):
        """测试质心维度验证"""
        with pytest.raises(ValueError):
            DynamicsParameters(
                masses=[1.0],
                centers_of_mass=[[0, 0]],  # 不是3D向量
                inertias=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                friction_coeffs=[0.1]
            )


class TestKinodynamicLimits:
    """运动学动力学限制测试"""
    
    def test_kinodynamic_limits_creation(self):
        """测试运动学动力学限制创建"""
        limits = KinodynamicLimits(
            max_joint_positions=[1.0, 2.0, 3.0],
            min_joint_positions=[-1.0, -2.0, -3.0],
            max_joint_velocities=[1.0, 2.0, 3.0],
            max_joint_accelerations=[10.0, 20.0, 30.0],
            max_joint_jerks=[100.0, 200.0, 300.0],
            max_joint_torques=[50.0, 100.0, 150.0]
        )
        
        assert len(limits.max_joint_positions) == 3
        assert len(limits.min_joint_positions) == 3
        assert len(limits.max_joint_velocities) == 3
    
    def test_positive_limits_validation(self):
        """测试正数限制验证"""
        with pytest.raises(ValueError):
            KinodynamicLimits(
                max_joint_positions=[1.0],
                min_joint_positions=[-1.0],
                max_joint_velocities=[-1.0],  # 必须为正数
                max_joint_accelerations=[10.0],
                max_joint_jerks=[100.0],
                max_joint_torques=[50.0]
            )


class TestPayloadInfo:
    """负载信息测试"""
    
    def test_payload_info_creation(self, sample_payload_info):
        """测试负载信息创建"""
        payload = sample_payload_info
        
        assert payload.mass == 2.0
        assert len(payload.center_of_mass) == 3
        assert len(payload.inertia) == 3
        assert payload.identification_confidence == 0.9
    
    def test_mass_validation(self):
        """测试质量验证"""
        with pytest.raises(ValueError):
            PayloadInfo(
                mass=-1.0,  # 质量不能为负
                center_of_mass=[0, 0, 0],
                inertia=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )


class TestControlCommand:
    """控制指令测试"""
    
    def test_control_command_creation(self):
        """测试控制指令创建"""
        command = ControlCommand(
            joint_positions=np.array([1, 2, 3]),
            control_mode="position",
            timestamp=1.0
        )
        
        assert len(command.joint_positions) == 3
        assert command.control_mode == "position"
        assert command.timestamp == 1.0
    
    def test_control_mode_validation(self):
        """测试控制模式验证"""
        with pytest.raises(ValueError):
            ControlCommand(
                joint_positions=np.array([1, 2, 3]),
                control_mode="invalid_mode"  # 无效模式
            )
    
    def test_position_mode_validation(self):
        """测试位置模式验证"""
        with pytest.raises(ValueError):
            ControlCommand(
                control_mode="position"  # 位置模式但没有提供位置
            )