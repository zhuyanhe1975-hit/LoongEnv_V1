"""
Pinocchio动力学库集成测试

测试Pinocchio动力学库的集成，包括ER15-1400机械臂模型的加载和动力学计算。
"""

import pytest
import numpy as np
from pathlib import Path

from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.mjcf_parser import MJCFParser
from robot_motion_control.algorithms.dynamics import DynamicsEngine
from robot_motion_control.core.types import PayloadInfo, AlgorithmError


class TestMJCFParser:
    """MJCF解析器测试"""
    
    def test_mjcf_parser_creation(self):
        """测试MJCF解析器创建"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        # 检查文件是否存在
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        parser = MJCFParser(mjcf_path)
        
        assert parser.model_name == "ER15-1400"
        assert parser.get_joint_count() == 6
        
        joint_names = parser.get_joint_names()
        expected_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        assert joint_names == expected_names
    
    def test_mjcf_dynamics_extraction(self):
        """测试从MJCF提取动力学参数"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        parser = MJCFParser(mjcf_path)
        dynamics_params = parser.extract_dynamics_parameters()
        
        # 验证参数数量
        assert len(dynamics_params.masses) == 6
        assert len(dynamics_params.centers_of_mass) == 6
        assert len(dynamics_params.inertias) == 6
        assert len(dynamics_params.friction_coeffs) == 6
        
        # 验证质量为正数
        for mass in dynamics_params.masses:
            assert mass > 0
        
        # 验证重力向量
        assert len(dynamics_params.gravity) == 3
        assert dynamics_params.gravity[2] < 0  # 重力向下
    
    def test_mjcf_kinodynamic_limits(self):
        """测试从MJCF提取运动学限制"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        parser = MJCFParser(mjcf_path)
        limits = parser.extract_kinodynamic_limits()
        
        # 验证限制数量
        assert len(limits.max_joint_positions) == 6
        assert len(limits.min_joint_positions) == 6
        assert len(limits.max_joint_velocities) == 6
        assert len(limits.max_joint_accelerations) == 6
        
        # 验证限制合理性
        for i in range(6):
            assert limits.min_joint_positions[i] < limits.max_joint_positions[i]
            assert limits.max_joint_velocities[i] > 0
            assert limits.max_joint_accelerations[i] > 0


class TestER15_1400Model:
    """ER15-1400机器人模型测试"""
    
    def test_er15_1400_creation(self):
        """测试ER15-1400模型创建"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        robot_model = RobotModel.create_er15_1400(mjcf_path)
        
        assert robot_model.name == "ER15-1400"
        assert robot_model.n_joints == 6
        assert robot_model.mjcf_path == mjcf_path
        
        # 验证动力学参数
        assert len(robot_model.dynamics_params.masses) == 6
        
        # 验证运动学限制
        min_pos, max_pos = robot_model.get_joint_limits()
        assert len(min_pos) == 6
        assert len(max_pos) == 6
    
    def test_er15_1400_from_mjcf(self):
        """测试从MJCF文件加载ER15-1400"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        robot_model = RobotModel.from_mjcf(mjcf_path)
        
        assert robot_model.name == "ER15-1400"
        assert robot_model.n_joints == 6
        
        # 验证元数据
        assert "joint_names" in robot_model.metadata
        assert len(robot_model.metadata["joint_names"]) == 6


class TestPinocchioIntegration:
    """Pinocchio集成测试"""
    
    @pytest.fixture
    def er15_1400_model(self):
        """ER15-1400模型fixture"""
        mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        if not Path(mjcf_path).exists():
            pytest.skip(f"MJCF文件不存在: {mjcf_path}")
        
        return RobotModel.create_er15_1400(mjcf_path)
    
    def test_dynamics_engine_creation(self, er15_1400_model):
        """测试动力学引擎创建"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        assert dynamics_engine.robot_model == er15_1400_model
        assert dynamics_engine.n_joints == 6
        assert len(dynamics_engine.masses) == 6
    
    def test_pinocchio_model_initialization(self, er15_1400_model):
        """测试Pinocchio模型初始化"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 访问Pinocchio模型（触发延迟初始化）
        pinocchio_model = dynamics_engine.pinocchio_model
        
        if pinocchio_model is not None:
            # 如果Pinocchio可用，验证模型
            assert pinocchio_model.nq == 6  # 6个关节位置
            assert pinocchio_model.nv == 6  # 6个关节速度
        else:
            # 如果Pinocchio不可用，跳过测试
            pytest.skip("Pinocchio库不可用")
    
    def test_forward_dynamics_computation(self, er15_1400_model):
        """测试正向动力学计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 关节位置
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 关节速度
        tau = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])  # 关节力矩
        
        # 计算正向动力学
        qdd = dynamics_engine.forward_dynamics(q, qd, tau)
        
        # 验证结果
        assert len(qdd) == 6
        assert not np.any(np.isnan(qdd))
        assert not np.any(np.isinf(qdd))
    
    def test_inverse_dynamics_computation(self, er15_1400_model):
        """测试逆向动力学计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 关节位置
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 关节速度
        qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 关节加速度
        
        # 计算逆向动力学
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        # 验证结果
        assert len(tau) == 6
        assert not np.any(np.isnan(tau))
        assert not np.any(np.isinf(tau))
    
    def test_jacobian_computation(self, er15_1400_model):
        """测试雅可比矩阵计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 关节位置
        
        # 计算雅可比矩阵
        jacobian = dynamics_engine.jacobian(q)
        
        # 验证结果
        assert jacobian.shape == (6, 6)  # 6x6矩阵（3位置+3姿态 x 6关节）
        assert not np.any(np.isnan(jacobian))
        assert not np.any(np.isinf(jacobian))
    
    def test_gravity_compensation(self, er15_1400_model):
        """测试重力补偿计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 关节位置
        
        # 计算重力补偿
        g = dynamics_engine.gravity_compensation(q)
        
        # 验证结果
        assert len(g) == 6
        assert not np.any(np.isnan(g))
        assert not np.any(np.isinf(g))
    
    def test_dynamics_consistency(self, er15_1400_model):
        """测试动力学计算一致性"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 逆向动力学：计算所需力矩
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        # 正向动力学：用计算出的力矩计算加速度
        qdd_computed = dynamics_engine.forward_dynamics(q, qd, tau)
        
        # 验证一致性（允许数值误差）
        np.testing.assert_allclose(qdd, qdd_computed, rtol=1e-3, atol=1e-6)
    
    def test_payload_update(self, er15_1400_model):
        """测试负载更新"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 创建负载信息
        payload = PayloadInfo(
            mass=5.0,
            center_of_mass=[0.0, 0.0, 0.1],
            inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            identification_confidence=0.95
        )
        
        # 更新负载
        dynamics_engine.update_payload(payload)
        
        # 验证负载已更新
        assert er15_1400_model.current_payload == payload
    
    def test_mass_matrix_computation(self, er15_1400_model):
        """测试质量矩阵计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # 计算质量矩阵
        M = dynamics_engine.compute_mass_matrix(q)
        
        # 验证结果
        assert M.shape == (6, 6)
        assert not np.any(np.isnan(M))
        assert not np.any(np.isinf(M))
        
        # 质量矩阵应该是对称正定的
        np.testing.assert_allclose(M, M.T, rtol=1e-10)  # 对称性
        eigenvals = np.linalg.eigvals(M)
        assert np.all(eigenvals > 0)  # 正定性
    
    def test_coriolis_matrix_computation(self, er15_1400_model):
        """测试科里奥利矩阵计算"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 测试输入
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # 计算科里奥利矩阵
        C = dynamics_engine.compute_coriolis_matrix(q, qd)
        
        # 验证结果
        assert C.shape == (6, 6)
        assert not np.any(np.isnan(C))
        assert not np.any(np.isinf(C))
    
    def test_input_validation(self, er15_1400_model):
        """测试输入验证"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 错误的输入维度
        q_wrong = np.array([0.1, 0.2, 0.3])  # 只有3个关节
        qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        tau = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
        
        # 应该抛出ValueError
        with pytest.raises(ValueError):
            dynamics_engine.forward_dynamics(q_wrong, qd, tau)
    
    def test_cache_functionality(self, er15_1400_model):
        """测试缓存功能"""
        dynamics_engine = DynamicsEngine(er15_1400_model)
        
        # 启用缓存
        dynamics_engine.enable_cache(True)
        
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # 第一次计算
        J1 = dynamics_engine.jacobian(q)
        
        # 第二次计算（应该使用缓存）
        J2 = dynamics_engine.jacobian(q)
        
        # 结果应该相同
        np.testing.assert_array_equal(J1, J2)
        
        # 禁用缓存
        dynamics_engine.enable_cache(False)
        
        # 缓存应该被清除
        assert dynamics_engine._last_q is None
        assert dynamics_engine._last_jacobian is None


class TestErrorHandling:
    """错误处理测试"""
    
    def test_nonexistent_mjcf_file(self):
        """测试不存在的MJCF文件"""
        with pytest.raises(FileNotFoundError):
            MJCFParser("nonexistent_file.xml")
    
    def test_algorithm_error_handling(self):
        """测试算法错误处理"""
        # 创建一个简单的测试模型
        from robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
        
        dynamics_params = DynamicsParameters(
            masses=[1.0],
            centers_of_mass=[[0.0, 0.0, 0.0]],
            inertias=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            friction_coeffs=[0.1],
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi],
            min_joint_positions=[-np.pi],
            max_joint_velocities=[2.0],
            max_joint_accelerations=[10.0],
            max_joint_jerks=[50.0],
            max_joint_torques=[100.0]
        )
        
        robot_model = RobotModel(
            name="test_robot",
            n_joints=1,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
        
        dynamics_engine = DynamicsEngine(robot_model)
        
        # 测试无效输入（NaN值）
        q_invalid = np.array([np.nan])
        qd = np.array([0.1])
        tau = np.array([1.0])
        
        # 应该处理NaN输入而不崩溃
        try:
            result = dynamics_engine.forward_dynamics(q_invalid, qd, tau)
            # 如果没有抛出异常，结果应该是有效的或者包含NaN
            assert len(result) == 1
        except AlgorithmError:
            # 抛出AlgorithmError是可接受的
            pass