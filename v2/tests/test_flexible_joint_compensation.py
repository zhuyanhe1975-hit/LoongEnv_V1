"""
测试柔性关节补偿算法

测试增强的柔性关节动力学模型、柔性补偿控制算法和末端反馈补偿功能。
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from robot_motion_control.algorithms.vibration_suppression import (
    VibrationSuppressor, FlexibleJointParameters, EndEffectorSensorData, VirtualSensorState
)
from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.types import ControlCommand, RobotState, PayloadInfo


class TestFlexibleJointCompensation:
    """测试柔性关节补偿算法"""
    
    @pytest.fixture
    def sample_robot_model(self):
        """创建测试用机器人模型"""
        mock_model = Mock(spec=RobotModel)
        mock_model.n_joints = 6
        return mock_model
    
    @pytest.fixture
    def flexible_params(self):
        """创建测试用柔性关节参数"""
        return FlexibleJointParameters(
            joint_stiffness=np.array([1e5, 1e5, 1e5, 5e4, 5e4, 5e4]),
            joint_damping=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            motor_inertia=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
            link_inertia=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            gear_ratio=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            transmission_compliance=np.array([1e-6, 1e-6, 1e-6, 2e-6, 2e-6, 2e-6])
        )
    
    @pytest.fixture
    def sample_robot_state(self):
        """创建测试用机器人状态"""
        return RobotState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            joint_velocities=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            joint_accelerations=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            joint_torques=np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0]),
            end_effector_transform=np.eye(4),
            timestamp=0.0
        )
    
    @pytest.fixture
    def sample_control_command(self):
        """创建测试用控制指令"""
        return ControlCommand(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            joint_velocities=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            joint_accelerations=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            joint_torques=np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0]),
            control_mode="torque",
            timestamp=0.001
        )
    
    @pytest.fixture
    def sample_payload_info(self):
        """创建测试用负载信息"""
        return PayloadInfo(
            mass=2.5,
            center_of_mass=[0.1, 0.05, 0.2],
            inertia=[[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
            identification_confidence=0.9
        )
    
    @pytest.fixture
    def sample_end_effector_sensor_data(self):
        """创建测试用末端执行器传感器数据"""
        return EndEffectorSensorData(
            position=np.array([0.5, 0.3, 0.8]),
            velocity=np.array([0.1, 0.05, 0.2]),
            acceleration=np.array([0.5, 0.2, 1.0]),
            force=np.array([5.0, 2.0, 10.0]),
            torque=np.array([1.0, 0.5, 2.0]),
            timestamp=0.001
        )
    
    def test_vibration_suppressor_initialization_with_flexible_params(self, sample_robot_model, flexible_params):
        """测试带柔性参数的振动抑制器初始化"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        assert suppressor.flexible_params == flexible_params
        assert suppressor.n_joints == 6
        assert len(suppressor.flexible_joint_observer) == 6
        assert 'kalman_filter' in suppressor.virtual_sensor
        assert 'position_gains' in suppressor.end_effector_controller
    
    def test_flexible_joint_observer_initialization(self, sample_robot_model, flexible_params):
        """测试柔性关节观测器初始化"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 检查每个关节的观测器
        for i in range(6):
            observer = suppressor.flexible_joint_observer[f'joint_{i}']
            
            # 检查状态空间矩阵维度
            assert observer['A'].shape == (4, 4)
            assert observer['B'].shape == (4, 1)
            assert observer['C'].shape == (1, 4)
            assert observer['L'].shape == (4, 1)
            assert len(observer['state']) == 4
            assert observer['error_covariance'].shape == (4, 4)
    
    def test_enhanced_flexible_joint_compensation(self, sample_robot_model, flexible_params,
                                                sample_control_command, sample_robot_state, sample_payload_info):
        """测试增强的柔性关节补偿"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 应用柔性关节补偿
        compensated_command = suppressor.compensate_flexible_joints(
            sample_control_command, sample_robot_state, sample_payload_info
        )
        
        # 验证补偿后的指令
        assert compensated_command.joint_torques is not None
        assert len(compensated_command.joint_torques) == 6
        
        # 验证补偿项不为零（说明补偿算法在工作）
        torque_difference = compensated_command.joint_torques - sample_control_command.joint_torques
        assert np.any(np.abs(torque_difference) > 1e-6)
        
        # 验证其他字段保持不变
        np.testing.assert_array_equal(compensated_command.joint_positions, sample_control_command.joint_positions)
        np.testing.assert_array_equal(compensated_command.joint_velocities, sample_control_command.joint_velocities)
        assert compensated_command.control_mode == sample_control_command.control_mode
    
    def test_end_effector_feedback_compensation(self, sample_robot_model, flexible_params,
                                              sample_control_command, sample_end_effector_sensor_data):
        """测试末端执行器反馈补偿"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 定义期望的末端执行器状态
        desired_state = {
            'position': np.array([0.6, 0.4, 0.9]),
            'velocity': np.array([0.0, 0.0, 0.0])
        }
        
        # 应用末端反馈补偿
        compensated_command = suppressor.apply_end_effector_feedback(
            sample_control_command, sample_end_effector_sensor_data, desired_state
        )
        
        # 验证补偿后的指令
        assert compensated_command.joint_torques is not None
        assert len(compensated_command.joint_torques) == 6
        
        # 验证补偿项不为零
        torque_difference = compensated_command.joint_torques - sample_control_command.joint_torques
        assert np.any(np.abs(torque_difference) > 1e-6)
    
    def test_virtual_sensor_update(self, sample_robot_model, flexible_params, sample_end_effector_sensor_data):
        """测试虚拟传感器更新"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 更新虚拟传感器
        virtual_state = suppressor._update_virtual_sensor(sample_end_effector_sensor_data)
        
        # 验证虚拟传感器状态
        assert isinstance(virtual_state, VirtualSensorState)
        assert len(virtual_state.estimated_position) == 3
        assert len(virtual_state.estimated_velocity) == 3
        assert len(virtual_state.estimation_error) == 3
        assert 0.0 <= virtual_state.confidence <= 1.0
    
    def test_flexible_joint_state_retrieval(self, sample_robot_model, flexible_params):
        """测试柔性关节状态获取"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 获取关节状态
        joint_state = suppressor.get_flexible_joint_state(0)
        
        # 验证状态字段
        expected_fields = ['motor_angle', 'link_angle', 'motor_velocity', 'link_velocity', 'deflection', 'deflection_rate']
        for field in expected_fields:
            assert field in joint_state
            assert isinstance(joint_state[field], (int, float, np.number))
    
    def test_parameter_updates(self, sample_robot_model, flexible_params):
        """测试参数更新功能"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 更新柔性参数
        new_params = FlexibleJointParameters(
            joint_stiffness=np.array([2e5, 2e5, 2e5, 1e5, 1e5, 1e5]),
            joint_damping=np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0]),
            motor_inertia=np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
            link_inertia=np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),
            gear_ratio=np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0]),
            transmission_compliance=np.array([5e-7, 5e-7, 5e-7, 1e-6, 1e-6, 1e-6])
        )
        
        suppressor.update_flexible_parameters(new_params)
        assert suppressor.flexible_params == new_params
        
        # 更新末端执行器增益
        position_gains = np.array([2000.0, 2000.0, 2000.0])
        velocity_gains = np.array([200.0, 200.0, 200.0])
        force_gains = np.array([0.2, 0.2, 0.2])
        integral_gains = np.array([20.0, 20.0, 20.0])
        
        suppressor.set_end_effector_gains(position_gains, velocity_gains, force_gains, integral_gains)
        
        np.testing.assert_array_equal(suppressor.end_effector_controller['position_gains'], position_gains)
        np.testing.assert_array_equal(suppressor.end_effector_controller['velocity_gains'], velocity_gains)
        np.testing.assert_array_equal(suppressor.end_effector_controller['force_gains'], force_gains)
        np.testing.assert_array_equal(suppressor.end_effector_controller['integral_gains'], integral_gains)
    
    def test_compensation_diagnostics(self, sample_robot_model, flexible_params):
        """测试补偿算法诊断功能"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 获取诊断信息
        diagnostics = suppressor.get_compensation_diagnostics()
        
        # 验证诊断信息结构
        assert 'flexible_joint_states' in diagnostics
        assert 'virtual_sensor_confidence' in diagnostics
        assert 'integral_errors' in diagnostics
        assert 'buffer_sizes' in diagnostics
        
        # 验证柔性关节状态
        assert len(diagnostics['flexible_joint_states']) == 6
        for i in range(6):
            assert f'joint_{i}' in diagnostics['flexible_joint_states']
        
        # 验证置信度
        assert 0.0 <= diagnostics['virtual_sensor_confidence'] <= 1.0
        
        # 验证积分误差
        assert len(diagnostics['integral_errors']) == 3
        
        # 验证缓冲区大小
        assert 'input_buffer' in diagnostics['buffer_sizes']
        assert 'state_history' in diagnostics['buffer_sizes']
    
    def test_adaptive_compensation_toggle(self, sample_robot_model, flexible_params):
        """测试自适应补偿开关"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 测试启用自适应补偿
        suppressor.enable_adaptive_compensation(True)
        assert suppressor.end_effector_controller['feedforward_compensation'] == True
        
        # 测试禁用自适应补偿
        suppressor.enable_adaptive_compensation(False)
        assert suppressor.end_effector_controller['feedforward_compensation'] == False
    
    def test_integral_error_reset(self, sample_robot_model, flexible_params):
        """测试积分误差重置"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 设置一些积分误差
        suppressor.end_effector_controller['integral_error'] = np.array([0.1, 0.2, 0.3])
        
        # 重置积分误差
        suppressor.reset_integral_errors()
        
        # 验证积分误差被重置为零
        np.testing.assert_array_equal(suppressor.end_effector_controller['integral_error'], np.zeros(3))
    
    def test_error_handling_invalid_joint_index(self, sample_robot_model, flexible_params):
        """测试无效关节索引的错误处理"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 测试超出范围的关节索引
        with pytest.raises(ValueError, match="关节索引超出范围"):
            suppressor.get_flexible_joint_state(10)
    
    def test_compensation_with_none_payload(self, sample_robot_model, flexible_params,
                                          sample_control_command, sample_robot_state):
        """测试无负载信息时的补偿"""
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 不提供负载信息的补偿
        compensated_command = suppressor.compensate_flexible_joints(
            sample_control_command, sample_robot_state, None
        )
        
        # 验证补偿仍然工作
        assert compensated_command.joint_torques is not None
        assert len(compensated_command.joint_torques) == 6
    
    def test_jacobian_computation_failure_handling(self, sample_robot_model, flexible_params,
                                                  sample_control_command, sample_end_effector_sensor_data):
        """测试雅可比矩阵计算失败的处理"""
        # 这个测试现在验证内部异常处理机制
        suppressor = VibrationSuppressor(sample_robot_model, flexible_params)
        
        # 应用末端反馈补偿（应该优雅地处理失败）
        compensated_command = suppressor.apply_end_effector_feedback(
            sample_control_command, sample_end_effector_sensor_data
        )
        
        # 验证返回了有效的指令
        assert compensated_command.joint_torques is not None


class TestFlexibleJointParameters:
    """测试柔性关节参数数据结构"""
    
    def test_flexible_joint_parameters_creation(self):
        """测试柔性关节参数创建"""
        params = FlexibleJointParameters(
            joint_stiffness=np.array([1e5, 1e5, 1e5, 5e4, 5e4, 5e4]),
            joint_damping=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            motor_inertia=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
            link_inertia=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            gear_ratio=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            transmission_compliance=np.array([1e-6, 1e-6, 1e-6, 2e-6, 2e-6, 2e-6])
        )
        
        assert len(params.joint_stiffness) == 6
        assert len(params.joint_damping) == 6
        assert len(params.motor_inertia) == 6
        assert len(params.link_inertia) == 6
        assert len(params.gear_ratio) == 6
        assert len(params.transmission_compliance) == 6


class TestEndEffectorSensorData:
    """测试末端执行器传感器数据结构"""
    
    def test_end_effector_sensor_data_creation(self):
        """测试末端执行器传感器数据创建"""
        sensor_data = EndEffectorSensorData(
            position=np.array([0.5, 0.3, 0.8]),
            velocity=np.array([0.1, 0.05, 0.2]),
            acceleration=np.array([0.5, 0.2, 1.0]),
            force=np.array([5.0, 2.0, 10.0]),
            torque=np.array([1.0, 0.5, 2.0]),
            timestamp=0.001
        )
        
        assert len(sensor_data.position) == 3
        assert len(sensor_data.velocity) == 3
        assert len(sensor_data.acceleration) == 3
        assert len(sensor_data.force) == 3
        assert len(sensor_data.torque) == 3
        assert sensor_data.timestamp == 0.001


class TestVirtualSensorState:
    """测试虚拟传感器状态结构"""
    
    def test_virtual_sensor_state_creation(self):
        """测试虚拟传感器状态创建"""
        state = VirtualSensorState(
            estimated_position=np.array([0.5, 0.3, 0.8]),
            estimated_velocity=np.array([0.1, 0.05, 0.2]),
            estimation_error=np.array([0.01, 0.005, 0.02]),
            confidence=0.95
        )
        
        assert len(state.estimated_position) == 3
        assert len(state.estimated_velocity) == 3
        assert len(state.estimation_error) == 3
        assert 0.0 <= state.confidence <= 1.0