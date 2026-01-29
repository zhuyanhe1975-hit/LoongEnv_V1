"""
路径控制器测试

测试高精度路径跟踪控制器的各种功能和控制模式。
验证前馈控制、反馈控制和组合控制的性能。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from robot_motion_control.algorithms.path_control import PathController, ControlMode
from robot_motion_control.core.types import (
    RobotState, TrajectoryPoint, ControlCommand, DynamicsParameters, 
    KinodynamicLimits, AlgorithmError
)
from robot_motion_control.core.models import RobotModel


class TestPathController:
    """路径控制器测试类"""
    
    @pytest.fixture
    def sample_robot_model(self):
        """创建测试用机器人模型"""
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
        
        return RobotModel(
            name="test_robot",
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
    
    @pytest.fixture
    def sample_trajectory_point(self):
        """创建测试用轨迹点"""
        return TrajectoryPoint(
            position=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            velocity=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            acceleration=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            jerk=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            time=1.0,
            path_parameter=0.5
        )
    
    @pytest.fixture
    def sample_robot_state(self):
        """创建测试用机器人状态"""
        return RobotState(
            joint_positions=np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
            joint_velocities=np.array([0.005, 0.015, 0.025, 0.035, 0.045, 0.055]),
            joint_accelerations=np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
            joint_torques=np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
            end_effector_transform=np.eye(4),
            timestamp=1.0
        )
    
    def test_path_controller_initialization(self, sample_robot_model):
        """测试路径控制器初始化"""
        # 默认初始化
        controller = PathController(sample_robot_model)
        
        assert controller.robot_model == sample_robot_model
        assert controller.n_joints == 6
        assert controller.control_mode == ControlMode.COMPUTED_TORQUE
        assert controller.enable_feedforward == True
        assert controller.enable_adaptation == True
        
        # 自定义初始化
        controller_custom = PathController(
            sample_robot_model,
            control_mode=ControlMode.PID,
            enable_feedforward=False,
            enable_adaptation=False
        )
        
        assert controller_custom.control_mode == ControlMode.PID
        assert controller_custom.enable_feedforward == False
        assert controller_custom.enable_adaptation == False
    
    def test_pid_control_mode(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试PID控制模式"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.PID,
            enable_feedforward=False
        )
        
        # 计算控制指令
        command = controller.compute_control(sample_trajectory_point, sample_robot_state)
        
        # 验证输出
        assert isinstance(command, ControlCommand)
        assert command.control_mode == "position"
        assert command.joint_positions is not None
        assert len(command.joint_positions) == 6
        assert not np.any(np.isnan(command.joint_positions))
        
        # 验证PID控制逻辑
        position_error = sample_trajectory_point.position - sample_robot_state.joint_positions
        assert np.allclose(position_error, np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
    
    def test_computed_torque_control_mode(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试计算力矩控制模式"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.COMPUTED_TORQUE,
            enable_feedforward=True
        )
        
        # Mock动力学引擎以避免复杂的动力学计算
        with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
            mock_dynamics = Mock()
            mock_dynamics.inverse_dynamics.return_value = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            mock_dynamics.gravity_compensation.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            mock_dynamics.compute_friction_torque.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            mock_dynamics_class.return_value = mock_dynamics
            
            # 计算控制指令
            command = controller.compute_control(sample_trajectory_point, sample_robot_state)
            
            # 验证输出
            assert isinstance(command, ControlCommand)
            assert command.control_mode == "torque"
            assert command.joint_torques is not None
            assert len(command.joint_torques) == 6
            assert not np.any(np.isnan(command.joint_torques))
    
    def test_sliding_mode_control(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试滑模控制"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.SLIDING_MODE,
            enable_feedforward=False
        )
        
        # Mock动力学引擎
        with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
            mock_dynamics = Mock()
            mock_dynamics.inverse_dynamics.return_value = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0])
            mock_dynamics_class.return_value = mock_dynamics
            
            # 计算控制指令
            command = controller.compute_control(sample_trajectory_point, sample_robot_state)
            
            # 验证输出
            assert isinstance(command, ControlCommand)
            assert command.control_mode == "torque"
            assert command.joint_torques is not None
            assert len(command.joint_torques) == 6
            assert not np.any(np.isnan(command.joint_torques))
    
    def test_adaptive_control(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试自适应控制"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.ADAPTIVE,
            enable_feedforward=False
        )
        
        # Mock动力学引擎
        with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
            mock_dynamics = Mock()
            mock_dynamics.inverse_dynamics.return_value = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])
            mock_dynamics_class.return_value = mock_dynamics
            
            # 计算控制指令
            command = controller.compute_control(sample_trajectory_point, sample_robot_state)
            
            # 验证输出
            assert isinstance(command, ControlCommand)
            assert command.control_mode == "torque"
            assert command.joint_torques is not None
            assert len(command.joint_torques) == 6
            assert not np.any(np.isnan(command.joint_torques))
            
            # 验证参数自适应
            initial_estimates = controller.param_estimates.copy()
            
            # 再次计算控制指令
            command2 = controller.compute_control(sample_trajectory_point, sample_robot_state, dt=0.001)
            
            # 参数估计应该有所变化
            assert not np.allclose(controller.param_estimates, initial_estimates)
    
    def test_feedforward_control(self, sample_robot_model, sample_trajectory_point):
        """测试前馈控制"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.COMPUTED_TORQUE,
            enable_feedforward=True
        )
        
        # Mock动力学引擎
        with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
            mock_dynamics = Mock()
            mock_dynamics.inverse_dynamics.return_value = np.array([8.0, 16.0, 24.0, 32.0, 40.0, 48.0])
            mock_dynamics.gravity_compensation.return_value = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
            mock_dynamics.compute_friction_torque.return_value = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
            mock_dynamics_class.return_value = mock_dynamics
            
            # 测试前馈控制
            feedforward_output = controller.feedforward_control(sample_trajectory_point)
            
            # 验证输出
            assert isinstance(feedforward_output, np.ndarray)
            assert len(feedforward_output) == 6
            assert not np.any(np.isnan(feedforward_output))
            
            # 验证动力学引擎被调用
            mock_dynamics.inverse_dynamics.assert_called_once()
            mock_dynamics.gravity_compensation.assert_called_once()
            mock_dynamics.compute_friction_torque.assert_called_once()
    
    def test_control_gain_setting(self, sample_robot_model):
        """测试控制增益设置"""
        controller = PathController(sample_robot_model)
        
        # 设置新的增益
        new_kp = np.array([150.0, 160.0, 170.0, 180.0, 190.0, 200.0])
        new_ki = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        new_kd = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        
        controller.set_control_gains(kp=new_kp, ki=new_ki, kd=new_kd)
        
        # 验证增益设置
        assert np.allclose(controller.kp, new_kp)
        assert np.allclose(controller.ki, new_ki)
        assert np.allclose(controller.kd, new_kd)
        
        # 验证计算力矩控制增益也相应更新
        assert np.allclose(controller.kp_ct, new_kp * 2.5)
        assert np.allclose(controller.kd_ct, new_kd * 3.3)
    
    def test_control_mode_switching(self, sample_robot_model):
        """测试控制模式切换"""
        controller = PathController(sample_robot_model, control_mode=ControlMode.PID)
        
        assert controller.control_mode == ControlMode.PID
        
        # 切换到计算力矩控制
        controller.set_control_gains(control_mode=ControlMode.COMPUTED_TORQUE)
        assert controller.control_mode == ControlMode.COMPUTED_TORQUE
        
        # 切换到滑模控制
        controller.set_control_gains(control_mode=ControlMode.SLIDING_MODE)
        assert controller.control_mode == ControlMode.SLIDING_MODE
    
    def test_performance_monitoring(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试性能监控"""
        controller = PathController(sample_robot_model, enable_feedforward=False)
        
        # 执行几次控制计算
        for i in range(5):
            sample_robot_state.timestamp = i * 0.001
            command = controller.compute_control(sample_trajectory_point, sample_robot_state)
        
        # 获取性能指标
        performance = controller.get_tracking_performance()
        
        # 验证性能指标
        assert isinstance(performance, dict)
        assert 'mean_tracking_error' in performance
        assert 'max_tracking_error' in performance
        assert 'rms_tracking_error' in performance
        assert 'mean_computation_time' in performance
        
        assert performance['mean_tracking_error'] >= 0
        assert performance['max_tracking_error'] >= 0
        assert performance['rms_tracking_error'] >= 0
        assert performance['mean_computation_time'] >= 0
        
        # 验证记录了正确数量的数据点
        assert len(controller.tracking_errors) == 5
        assert len(controller.computation_times) == 5
    
    def test_controller_state_reset(self, sample_robot_model):
        """测试控制器状态重置"""
        controller = PathController(sample_robot_model)
        
        # 设置一些状态
        controller.integral_error = np.ones(6) * 0.5
        controller.last_error = np.ones(6) * 0.3
        controller.param_estimates = np.ones(6) * 1.5
        controller.tracking_errors = [0.1, 0.2, 0.3]
        
        # 重置状态
        controller.reset_controller_state()
        
        # 验证状态被重置
        assert np.allclose(controller.integral_error, np.zeros(6))
        assert np.allclose(controller.last_error, np.zeros(6))
        assert np.allclose(controller.param_estimates, np.ones(6))
        assert len(controller.tracking_errors) == 0
        assert len(controller.control_efforts) == 0
        assert len(controller.computation_times) == 0
    
    def test_feedforward_gain_setting(self, sample_robot_model):
        """测试前馈增益设置"""
        controller = PathController(sample_robot_model)
        
        # 设置前馈增益
        controller.set_feedforward_gains(
            feedforward_gain=0.8,
            gravity_gain=1.2,
            friction_gain=0.6
        )
        
        # 验证增益设置
        assert controller.feedforward_gain == 0.8
        assert controller.gravity_compensation_gain == 1.2
        assert controller.friction_compensation_gain == 0.6
    
    def test_feedforward_enable_disable(self, sample_robot_model):
        """测试前馈控制启用/禁用"""
        controller = PathController(sample_robot_model, enable_feedforward=True)
        
        assert controller.enable_feedforward == True
        
        # 禁用前馈控制
        controller.enable_feedforward_control(False)
        assert controller.enable_feedforward == False
        
        # 重新启用前馈控制
        controller.enable_feedforward_control(True)
        assert controller.enable_feedforward == True
    
    def test_integral_windup_protection(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试积分饱和保护"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.PID,
            enable_feedforward=False
        )
        
        # 设置大的积分增益
        controller.set_control_gains(ki=np.ones(6) * 1000.0)
        
        # 创建大的跟踪误差
        large_error_state = sample_robot_state
        large_error_state.joint_positions = np.zeros(6)  # 大误差
        
        # 执行多次控制计算
        for i in range(100):
            large_error_state.timestamp = i * 0.001
            command = controller.compute_control(sample_trajectory_point, large_error_state, dt=0.001)
        
        # 验证积分项被限制
        integral_limit = 0.1
        assert np.all(np.abs(controller.integral_error) <= integral_limit)
    
    def test_error_handling(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试错误处理"""
        controller = PathController(sample_robot_model)
        
        # Mock动力学引擎抛出异常
        with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
            mock_dynamics = Mock()
            mock_dynamics.inverse_dynamics.side_effect = Exception("动力学计算失败")
            mock_dynamics_class.return_value = mock_dynamics
            
            # 应该抛出AlgorithmError
            with pytest.raises(AlgorithmError):
                controller.compute_control(sample_trajectory_point, sample_robot_state)
    
    def test_time_step_calculation(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试时间步长计算"""
        controller = PathController(sample_robot_model, enable_feedforward=False)
        
        # 第一次调用，应该使用默认时间步长
        sample_robot_state.timestamp = 1.0
        command1 = controller.compute_control(sample_trajectory_point, sample_robot_state)
        
        # 第二次调用，应该计算实际时间步长
        sample_robot_state.timestamp = 1.005  # 5ms后
        command2 = controller.compute_control(sample_trajectory_point, sample_robot_state)
        
        # 验证控制器记录了时间
        assert controller.last_time == 1.005
    
    def test_auto_tune_gains_basic(self, sample_robot_model):
        """测试自动调参基本功能"""
        controller = PathController(sample_robot_model)
        
        # 创建简单的测试数据
        reference_trajectory = [
            Mock(position=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
            Mock(position=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        ]
        
        simulation_states = [
            Mock(joint_positions=np.array([0.09, 0.19, 0.29, 0.39, 0.49, 0.59])),
            Mock(joint_positions=np.array([0.18, 0.28, 0.38, 0.48, 0.58, 0.68]))
        ]
        
        # 记录原始增益
        original_kp = controller.kp.copy()
        
        # 执行自动调参
        best_gains = controller.auto_tune_gains(reference_trajectory, simulation_states)
        
        # 验证返回了增益字典
        assert isinstance(best_gains, dict)
        assert 'kp' in best_gains
        assert 'ki' in best_gains
        assert 'kd' in best_gains
        
        # 验证增益可能发生了变化（取决于优化结果）
        assert isinstance(best_gains['kp'], np.ndarray)
        assert len(best_gains['kp']) == 6


class TestPathControllerIntegration:
    """路径控制器集成测试"""
    
    def test_control_loop_simulation(self, sample_robot_model):
        """测试控制循环仿真"""
        controller = PathController(
            sample_robot_model,
            control_mode=ControlMode.PID,
            enable_feedforward=False
        )
        
        # 设置更合理的PID增益
        controller.set_control_gains(
            kp=np.ones(6) * 50.0,
            ki=np.ones(6) * 5.0,
            kd=np.ones(6) * 10.0
        )
        
        # 创建简单轨迹
        trajectory_points = []
        for i in range(10):
            t = i * 0.01
            pos = np.sin(t) * np.ones(6) * 0.1
            vel = np.cos(t) * np.ones(6) * 0.1
            acc = -np.sin(t) * np.ones(6) * 0.1
            
            point = TrajectoryPoint(
                position=pos,
                velocity=vel,
                acceleration=acc,
                jerk=np.zeros(6),
                time=t,
                path_parameter=t / 0.09
            )
            trajectory_points.append(point)
        
        # 模拟控制循环
        current_state = RobotState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=0.0
        )
        
        tracking_errors = []
        
        for i, ref_point in enumerate(trajectory_points):
            current_state.timestamp = ref_point.time
            
            # 计算控制指令
            command = controller.compute_control(ref_point, current_state)
            
            # 简单的机器人响应模拟（更稳定的模型）
            if command.joint_positions is not None:
                # 限制位置变化以避免数值不稳定
                position_error = command.joint_positions - current_state.joint_positions
                position_change = np.clip(position_error * 0.1, -0.01, 0.01)  # 限制变化幅度
                current_state.joint_positions += position_change
                current_state.joint_velocities = position_change / 0.01
            
            # 计算跟踪误差
            error = np.linalg.norm(ref_point.position - current_state.joint_positions)
            tracking_errors.append(error)
        
        # 验证跟踪性能
        final_error = tracking_errors[-1]
        mean_error = np.mean(tracking_errors)
        
        # 验证误差在合理范围内
        assert final_error < 1.0  # 最终误差应该小于1.0
        assert mean_error < 0.5   # 平均误差应该小于0.5
        
        # 验证性能监控数据
        performance = controller.get_tracking_performance()
        assert performance['mean_tracking_error'] > 0
        assert len(controller.tracking_errors) == len(trajectory_points)
    
    def test_different_control_modes_comparison(self, sample_robot_model, sample_trajectory_point, sample_robot_state):
        """测试不同控制模式的比较"""
        control_modes = [ControlMode.PID, ControlMode.COMPUTED_TORQUE, ControlMode.SLIDING_MODE]
        
        results = {}
        
        for mode in control_modes:
            controller = PathController(
                sample_robot_model,
                control_mode=mode,
                enable_feedforward=False
            )
            
            # Mock动力学引擎（对于需要的模式）
            if mode != ControlMode.PID:
                with patch('robot_motion_control.algorithms.dynamics.DynamicsEngine') as mock_dynamics_class:
                    mock_dynamics = Mock()
                    mock_dynamics.inverse_dynamics.return_value = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                    mock_dynamics_class.return_value = mock_dynamics
                    
                    command = controller.compute_control(sample_trajectory_point, sample_robot_state)
            else:
                command = controller.compute_control(sample_trajectory_point, sample_robot_state)
            
            results[mode] = command
        
        # 验证所有模式都产生了有效的控制指令
        for mode, command in results.items():
            assert isinstance(command, ControlCommand)
            if mode == ControlMode.PID:
                assert command.control_mode == "position"
                assert command.joint_positions is not None
            else:
                assert command.control_mode == "torque"
                assert command.joint_torques is not None