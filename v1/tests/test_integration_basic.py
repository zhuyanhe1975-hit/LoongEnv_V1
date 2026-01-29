"""
基础集成测试

测试核心组件的基本集成功能，验证系统的基本工作流程。
"""

import pytest
import numpy as np

from robot_motion_control import (
    RobotMotionController, RobotModel, DynamicsEngine,
    TrajectoryPlanner, PathController, VibrationSuppressor
)
from robot_motion_control.algorithms.path_control import ControlMode
from robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, RobotState,
    TrajectoryPoint, Waypoint, ControlCommand
)


class TestBasicIntegration:
    """基础集成测试"""
    
    def test_robot_model_creation(self):
        """测试机器人模型创建"""
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
        
        robot_model = RobotModel(
            name="test_robot",
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
        
        assert robot_model.name == "test_robot"
        assert robot_model.n_joints == n_joints
        assert len(robot_model.dynamics_params.masses) == n_joints
    
    def test_dynamics_engine_basic(self, sample_robot_model):
        """测试动力学引擎基本功能"""
        dynamics_engine = DynamicsEngine(sample_robot_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qd = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        qdd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # 测试逆动力学
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        assert len(tau) == sample_robot_model.n_joints
        assert not np.any(np.isnan(tau))
        
        # 测试正向动力学
        qdd_computed = dynamics_engine.forward_dynamics(q, qd, tau)
        assert len(qdd_computed) == sample_robot_model.n_joints
        assert not np.any(np.isnan(qdd_computed))
        
        # 测试雅可比矩阵
        jacobian = dynamics_engine.jacobian(q)
        assert jacobian.shape == (6, sample_robot_model.n_joints)
        assert not np.any(np.isnan(jacobian))
        
        # 测试重力补偿
        g = dynamics_engine.gravity_compensation(q)
        assert len(g) == sample_robot_model.n_joints
        assert not np.any(np.isnan(g))
    
    def test_trajectory_planner_basic(self, sample_robot_model):
        """测试轨迹规划器基本功能"""
        planner = TrajectoryPlanner(sample_robot_model)
        
        # 创建简单路径
        waypoints = [
            Waypoint(position=np.zeros(6)),
            Waypoint(position=np.ones(6) * 0.5),
            Waypoint(position=np.ones(6))
        ]
        
        # 测试S型插补
        trajectory = planner.interpolate_s7_trajectory(waypoints)
        
        assert len(trajectory) > 0
        assert all(isinstance(point, TrajectoryPoint) for point in trajectory)
        assert trajectory[0].path_parameter == 0.0
        assert trajectory[-1].path_parameter == 1.0
        
        # 测试TOPP轨迹规划
        topp_trajectory = planner.generate_topp_trajectory(
            waypoints, sample_robot_model.kinodynamic_limits
        )
        
        assert len(topp_trajectory) > 0
        assert all(isinstance(point, TrajectoryPoint) for point in topp_trajectory)
    
    def test_path_controller_basic(self, sample_robot_model, sample_robot_state, sample_trajectory_point):
        """测试路径控制器基本功能"""
        # 使用PID模式以保持向后兼容性
        controller = PathController(sample_robot_model, control_mode=ControlMode.PID)
        
        # 测试控制计算
        command = controller.compute_control(sample_trajectory_point, sample_robot_state)
        
        assert isinstance(command, ControlCommand)
        assert command.joint_positions is not None
        assert len(command.joint_positions) == sample_robot_model.n_joints
        assert not np.any(np.isnan(command.joint_positions))
    
    def test_vibration_suppressor_basic(self, sample_robot_model):
        """测试振动抑制器基本功能"""
        suppressor = VibrationSuppressor(sample_robot_model)
        
        # 创建测试控制指令
        command = ControlCommand(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            control_mode="position"
        )
        
        # 测试输入整形
        shaped_command = suppressor.apply_input_shaping(command)
        
        assert isinstance(shaped_command, ControlCommand)
        assert shaped_command.joint_positions is not None
        assert len(shaped_command.joint_positions) == sample_robot_model.n_joints
        assert not np.any(np.isnan(shaped_command.joint_positions))
    
    def test_robot_motion_controller_basic(self, sample_robot_model):
        """测试机器人运动控制器基本功能"""
        controller = RobotMotionController(sample_robot_model)
        
        # 测试控制器状态
        status = controller.get_controller_status()
        assert isinstance(status, dict)
        assert "is_active" in status
        assert "emergency_stop" in status
        
        # 测试性能指标
        metrics = controller.get_performance_metrics()
        assert hasattr(metrics, 'computation_time')
        assert hasattr(metrics, 'tracking_error')
        assert hasattr(metrics, 'vibration_amplitude')
        
        # 测试轨迹规划
        waypoints = [
            Waypoint(position=np.zeros(6)),
            Waypoint(position=np.ones(6))
        ]
        
        trajectory = controller.plan_trajectory(waypoints)
        assert len(trajectory) > 0
        assert controller.current_trajectory is not None
    
    @pytest.mark.skip(reason="端到端工作流测试需要调整")
    def test_end_to_end_workflow(self, sample_robot_model):
        """测试端到端工作流程"""
        # 创建控制器
        controller = RobotMotionController(sample_robot_model)
        
        # 规划轨迹
        waypoints = [
            Waypoint(position=np.zeros(6)),
            Waypoint(position=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])),
            Waypoint(position=np.ones(6))
        ]
        
        trajectory = controller.plan_trajectory(waypoints, optimize_time=True)
        assert len(trajectory) > 0
        
        # 启动控制
        controller.start_control()
        assert controller.is_active
        
        # 模拟控制循环
        current_state = RobotState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=0.0
        )
        
        # 计算几步控制指令
        for i in range(5):
            current_state.timestamp = i * 0.01
            command = controller.compute_control(current_state)
            
            assert isinstance(command, ControlCommand)
            assert command.joint_positions is not None
            assert not np.any(np.isnan(command.joint_positions))
            
            # 简单更新状态（模拟机器人响应）
            if command.joint_positions is not None:
                current_state.joint_positions = command.joint_positions * 0.1
        
        # 停止控制
        controller.stop_control()
        assert not controller.is_active
        
        # 检查性能指标
        metrics = controller.get_performance_metrics()
        assert metrics.computation_time >= 0
        assert metrics.tracking_error >= 0