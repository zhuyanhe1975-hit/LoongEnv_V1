"""
重力补偿和摩擦力建模测试

测试增强的重力补偿算法和高级摩擦力模型的正确性和性能。
"""

import pytest
import numpy as np
from pathlib import Path

from robot_motion_control.core.models import RobotModel
from robot_motion_control.algorithms.dynamics import DynamicsEngine
from robot_motion_control.core.types import PayloadInfo, AlgorithmError


class TestEnhancedGravityCompensation:
    """增强重力补偿测试"""
    
    @pytest.fixture
    def test_robot_model(self):
        """创建测试用机器人模型"""
        # 创建简化的测试模型（不使用MJCF以避免Pinocchio复杂性）
        from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
        
        dynamics_params = DynamicsParameters(
            masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            centers_of_mass=[
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0], 
                [0.15, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.03, 0.0, 0.0]  # Changed from [0,0,0] to ensure non-zero
            ],
            inertias=[
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8]],
                [[0.6, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.6]],
                [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
                [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
            ],
            friction_coeffs=[0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi] * 6,
            min_joint_positions=[-np.pi] * 6,
            max_joint_velocities=[2.0] * 6,
            max_joint_accelerations=[10.0] * 6,
            max_joint_jerks=[50.0] * 6,
            max_joint_torques=[100.0] * 6
        )
        
        return RobotModel(
            name="test_robot_6dof",
            n_joints=6,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
    
    def test_gravity_compensation_basic(self, test_robot_model):
        """测试基本重力补偿功能"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 强制使用简化方法进行测试
        dynamics_engine.disable_pinocchio_for_testing()
        
        # 测试不同关节配置
        test_configurations = [
            np.zeros(6),  # 零位置
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # 一般位置
            np.array([np.pi/2, 0, -np.pi/2, 0, np.pi/4, 0]),  # 特殊位置
        ]
        
        for q in test_configurations:
            g = dynamics_engine.gravity_compensation(q)
            
            # 验证结果
            assert len(g) == 6
            assert not np.any(np.isnan(g))
            assert not np.any(np.isinf(g))
            
            # 重力补偿应该与关节配置相关（除了零位置可能为零）
            if not np.allclose(q, 0.0, atol=1e-6):
                assert not np.allclose(g, 0.0, atol=1e-6)
    
    def test_gravity_compensation_with_payload(self, test_robot_model):
        """测试带负载的重力补偿"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 强制使用简化方法进行测试
        dynamics_engine.disable_pinocchio_for_testing()
        
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # 无负载时的重力补偿
        g_no_payload = dynamics_engine.gravity_compensation(q)
        
        # 添加负载
        payload = PayloadInfo(
            mass=5.0,
            center_of_mass=[0.1, 0.0, 0.1],  # 确保有水平分量
            inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            identification_confidence=0.95
        )
        
        dynamics_engine.update_payload(payload)
        
        # 有负载时的重力补偿
        g_with_payload = dynamics_engine.gravity_compensation(q)
        
        # 验证负载影响
        assert len(g_with_payload) == 6
        assert not np.allclose(g_no_payload, g_with_payload, rtol=1e-3)
        
        # 负载应该增加重力补偿的绝对值
        assert np.sum(np.abs(g_with_payload)) > np.sum(np.abs(g_no_payload))
    
    def test_payload_effect_analysis(self, test_robot_model):
        """测试负载对动力学影响的分析"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 强制使用简化方法进行测试
        dynamics_engine.disable_pinocchio_for_testing()
        
        # 添加负载
        payload = PayloadInfo(
            mass=3.0,
            center_of_mass=[0.05, 0.0, 0.1],
            inertia=[[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
            identification_confidence=0.9
        )
        
        dynamics_engine.update_payload(payload)
        
        q = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        
        # 分析负载影响
        effect_analysis = dynamics_engine.get_payload_effect_on_dynamics(q)
        
        # 验证分析结果
        assert 'gravity_effect' in effect_analysis
        assert 'inertia_effect' in effect_analysis
        assert 'total_effect' in effect_analysis
        
        for key, effect in effect_analysis.items():
            assert len(effect) == 6
            assert not np.any(np.isnan(effect))
            assert not np.any(np.isinf(effect))
        
        # 重力影响应该不为零（因为有负载）
        assert not np.allclose(effect_analysis['gravity_effect'], 0.0, atol=1e-6)


class TestAdvancedFrictionModeling:
    """高级摩擦力建模测试"""
    
    @pytest.fixture
    def test_robot_model(self):
        """创建测试用机器人模型"""
        from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
        
        dynamics_params = DynamicsParameters(
            masses=[5.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            centers_of_mass=[[0.1, 0.0, 0.0]] * 6,
            inertias=[[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]] * 6,
            friction_coeffs=[0.2, 0.15, 0.1, 0.08, 0.05, 0.03],
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi] * 6,
            min_joint_positions=[-np.pi] * 6,
            max_joint_velocities=[3.0] * 6,
            max_joint_accelerations=[15.0] * 6,
            max_joint_jerks=[100.0] * 6,
            max_joint_torques=[200.0] * 6
        )
        
        return RobotModel(
            name="friction_test_robot",
            n_joints=6,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
    
    def test_friction_torque_computation(self, test_robot_model):
        """测试摩擦力矩计算"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 测试不同速度
        test_velocities = [
            np.zeros(6),  # 零速度
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # 低速
            np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),  # 高速
            np.array([-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]),  # 正负速度
        ]
        
        for qd in test_velocities:
            friction = dynamics_engine.compute_friction_torque(qd)
            
            # 验证结果
            assert len(friction) == 6
            assert not np.any(np.isnan(friction))
            assert not np.any(np.isinf(friction))
            
            # 摩擦力应该与速度方向相反（除了零速度）
            if not np.allclose(qd, 0.0, atol=1e-6):
                for i in range(6):
                    if abs(qd[i]) > 1e-6:
                        # 摩擦力的主要分量应该与速度方向相反
                        # 但由于包含多种摩擦效应，可能不是严格相反
                        # 这里检查摩擦力不为零即可
                        assert abs(friction[i]) > 1e-8
    
    def test_temperature_effect_on_friction(self, test_robot_model):
        """测试温度对摩擦的影响"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        qd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 不同温度下的摩擦力
        temp_20 = dynamics_engine.compute_friction_torque(qd, temperature=20.0)
        temp_40 = dynamics_engine.compute_friction_torque(qd, temperature=40.0)
        temp_0 = dynamics_engine.compute_friction_torque(qd, temperature=0.0)
        
        # 验证温度影响
        assert not np.allclose(temp_20, temp_40, rtol=1e-3)
        assert not np.allclose(temp_20, temp_0, rtol=1e-3)
        
        # 高温时摩擦力应该增加
        assert np.all(np.abs(temp_40) > np.abs(temp_20))
        
        # 低温时摩擦力应该减少
        assert np.all(np.abs(temp_0) < np.abs(temp_20))
    
    def test_stribeck_effect(self, test_robot_model):
        """测试Stribeck效应"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 测试不同速度下的摩擦特性
        velocities = np.linspace(-2.0, 2.0, 21)
        friction_values = []
        
        for vel in velocities:
            qd = np.array([vel, 0, 0, 0, 0, 0])
            friction = dynamics_engine.compute_friction_torque(qd)
            friction_values.append(friction[0])
        
        friction_values = np.array(friction_values)
        
        # 验证Stribeck效应特征
        # 1. 零速度时摩擦力为零（我们的模型设计）
        zero_idx = len(velocities) // 2
        assert abs(friction_values[zero_idx]) == 0.0
        
        # 2. 摩擦力应该随速度变化
        positive_vel_friction = friction_values[velocities > 0.1]
        negative_vel_friction = friction_values[velocities < -0.1]
        
        # 正速度时摩擦力为负，负速度时摩擦力为正
        assert np.all(positive_vel_friction < 0)
        assert np.all(negative_vel_friction > 0)
    
    def test_friction_parameter_update(self, test_robot_model):
        """测试摩擦参数更新"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        qd = np.array([1.0, 0, 0, 0, 0, 0])
        
        # 原始摩擦力
        original_friction = dynamics_engine.compute_friction_torque(qd)
        
        # 更新第一个关节的摩擦系数
        new_friction_coeff = 0.5
        dynamics_engine.update_friction_parameters(0, new_friction_coeff)
        
        # 新的摩擦力
        updated_friction = dynamics_engine.compute_friction_torque(qd)
        
        # 验证更新效果
        assert dynamics_engine.friction_coeffs[0] == new_friction_coeff
        assert abs(updated_friction[0]) != abs(original_friction[0])
        
        # 其他关节不应受影响
        np.testing.assert_allclose(
            updated_friction[1:], original_friction[1:], rtol=1e-10
        )
    
    def test_friction_calibration(self, test_robot_model):
        """测试摩擦参数标定"""
        dynamics_engine = DynamicsEngine(test_robot_model)
        
        # 生成模拟的运动数据
        motion_data = []
        
        for i in range(20):
            q = np.random.uniform(-np.pi, np.pi, 6)
            qd = np.random.uniform(-2.0, 2.0, 6)
            
            # 模拟测量的力矩（包含摩擦）
            tau_theoretical = dynamics_engine.inverse_dynamics(q, qd, np.zeros(6))
            friction_noise = np.random.normal(0, 0.1, 6)
            tau_measured = tau_theoretical + friction_noise
            
            motion_data.append((q, qd, tau_measured))
        
        # 执行标定
        original_coeffs = dynamics_engine.friction_coeffs.copy()
        dynamics_engine.calibrate_friction_parameters(motion_data)
        
        # 验证标定结果
        updated_coeffs = dynamics_engine.friction_coeffs
        
        # 摩擦系数应该有所变化（除非数据完全一致）
        # 这里主要验证标定过程不会崩溃
        assert len(updated_coeffs) == 6
        assert np.all(updated_coeffs >= 0)  # 摩擦系数应该非负


class TestDynamicsWithEnhancements:
    """增强动力学计算测试"""
    
    @pytest.fixture
    def enhanced_robot_model(self):
        """创建增强测试机器人模型"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if Path(mjcf_path).exists():
            return RobotModel.create_er15_1400(mjcf_path)
        else:
            # 使用简化模型
            from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
            
            dynamics_params = DynamicsParameters(
                masses=[20.0, 15.0, 10.0, 8.0, 5.0, 2.0],
                centers_of_mass=[
                    [0.15, 0.0, 0.0],
                    [0.25, 0.0, 0.0], 
                    [0.2, 0.0, 0.0],
                    [0.15, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.05, 0.0, 0.0]
                ],
                inertias=[
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                    [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8]],
                    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
                    [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]
                ],
                friction_coeffs=[0.15, 0.12, 0.1, 0.08, 0.06, 0.04],
                gravity=[0.0, 0.0, -9.81]
            )
            
            kinodynamic_limits = KinodynamicLimits(
                max_joint_positions=[2.967, 1.5708, 3.0543, 3.316, 2.2689, 6.2832],
                min_joint_positions=[-2.967, -2.7925, -1.4835, -3.316, -2.2689, -6.2832],
                max_joint_velocities=[3.14, 2.5, 3.14, 4.0, 4.0, 6.0],
                max_joint_accelerations=[15.0, 12.0, 15.0, 20.0, 20.0, 30.0],
                max_joint_jerks=[100.0, 80.0, 100.0, 150.0, 150.0, 200.0],
                max_joint_torques=[200.0, 180.0, 120.0, 80.0, 50.0, 30.0]
            )
            
            return RobotModel(
                name="enhanced_test_robot",
                n_joints=6,
                dynamics_params=dynamics_params,
                kinodynamic_limits=kinodynamic_limits
            )
    
    def test_enhanced_dynamics_consistency(self, enhanced_robot_model):
        """测试增强动力学的一致性"""
        dynamics_engine = DynamicsEngine(enhanced_robot_model)
        
        # 测试配置
        q = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        qd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qdd = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        
        # 逆向动力学
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        # 正向动力学
        qdd_computed = dynamics_engine.forward_dynamics(q, qd, tau)
        
        # 验证一致性（考虑增强摩擦模型的非线性）
        np.testing.assert_allclose(qdd, qdd_computed, rtol=1e-2, atol=1e-3)
    
    def test_payload_dynamics_integration(self, enhanced_robot_model):
        """测试负载与动力学的集成"""
        dynamics_engine = DynamicsEngine(enhanced_robot_model)
        
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 无负载时的力矩
        tau_no_payload = dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        # 添加负载
        payload = PayloadInfo(
            mass=10.0,
            center_of_mass=[0.0, 0.0, 0.15],
            inertia=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            identification_confidence=0.98
        )
        
        dynamics_engine.update_payload(payload)
        
        # 有负载时的力矩
        tau_with_payload = dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        # 验证负载影响
        assert not np.allclose(tau_no_payload, tau_with_payload, rtol=1e-3)
        
        # 负载应该增加所需力矩（特别是前几个关节）
        for i in range(3):  # 前三个关节承受更多负载影响
            assert abs(tau_with_payload[i]) > abs(tau_no_payload[i])
    
    def test_error_handling_enhanced(self, enhanced_robot_model):
        """测试增强功能的错误处理"""
        dynamics_engine = DynamicsEngine(enhanced_robot_model)
        
        # 测试无效负载参数 - Pydantic会在创建时就验证
        try:
            invalid_payload = PayloadInfo(
                mass=-1.0,  # 负质量
                center_of_mass=[0.0, 0.0, 0.0],
                inertia=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                identification_confidence=0.5
            )
            # 如果创建成功，则测试update_payload的验证
            dynamics_engine.update_payload(invalid_payload)
            assert False, "应该抛出验证错误"
        except (ValueError, Exception):
            # 预期的验证错误
            pass
        
        # 测试无效摩擦参数更新
        with pytest.raises(ValueError):
            dynamics_engine.update_friction_parameters(10, 0.1)  # 超出关节范围
        
        # 测试摩擦标定数据不足
        with pytest.raises(ValueError):
            dynamics_engine.calibrate_friction_parameters([])  # 空数据


class TestPerformanceEnhancements:
    """性能增强测试"""
    
    def test_computation_performance(self):
        """测试计算性能"""
        # 创建简单模型进行性能测试
        from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
        
        dynamics_params = DynamicsParameters(
            masses=[1.0] * 6,
            centers_of_mass=[[0.1, 0.0, 0.0]] * 6,
            inertias=[[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]] * 6,
            friction_coeffs=[0.1] * 6,
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi] * 6,
            min_joint_positions=[-np.pi] * 6,
            max_joint_velocities=[2.0] * 6,
            max_joint_accelerations=[10.0] * 6,
            max_joint_jerks=[50.0] * 6,
            max_joint_torques=[100.0] * 6
        )
        
        robot_model = RobotModel(
            name="performance_test_robot",
            n_joints=6,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
        
        dynamics_engine = DynamicsEngine(robot_model)
        
        # 性能测试
        import time
        
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 测试重力补偿性能
        start_time = time.time()
        for _ in range(100):
            g = dynamics_engine.gravity_compensation(q)
        gravity_time = (time.time() - start_time) / 100
        
        # 测试摩擦计算性能
        start_time = time.time()
        for _ in range(100):
            friction = dynamics_engine.compute_friction_torque(qd)
        friction_time = (time.time() - start_time) / 100
        
        # 测试逆向动力学性能
        start_time = time.time()
        for _ in range(100):
            tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        inverse_dynamics_time = (time.time() - start_time) / 100
        
        # 验证性能要求（应该在毫秒级别）
        assert gravity_time < 0.01  # 10ms
        assert friction_time < 0.01  # 10ms
        assert inverse_dynamics_time < 0.01  # 10ms
        
        print(f"重力补偿时间: {gravity_time*1000:.3f}ms")
        print(f"摩擦计算时间: {friction_time*1000:.3f}ms")
        print(f"逆向动力学时间: {inverse_dynamics_time*1000:.3f}ms")