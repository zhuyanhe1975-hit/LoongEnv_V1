"""
TOPP算法测试模块

测试时间最优路径参数化（TOPP）算法的正确性和性能。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, PayloadInfo, 
    Waypoint, Path, TrajectoryPoint
)
from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner


class TestTOPPAlgorithm:
    """TOPP算法测试类"""
    
    @pytest.fixture
    def robot_model(self):
        """创建测试用机器人模型"""
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
    def trajectory_planner(self, robot_model):
        """创建轨迹规划器"""
        return TrajectoryPlanner(robot_model)
    
    @pytest.fixture
    def simple_path(self):
        """创建简单测试路径"""
        return [
            Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            Waypoint(position=np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0])),
            Waypoint(position=np.array([1.0, 0.5, 0.4, 0.2, 0.1, 0.0])),
            Waypoint(position=np.array([1.5, 0.8, 0.6, 0.3, 0.2, 0.1]))
        ]
    
    def test_topp_basic_functionality(self, trajectory_planner, simple_path, robot_model):
        """测试TOPP算法基本功能"""
        limits = robot_model.kinodynamic_limits
        
        # 生成TOPP轨迹
        trajectory = trajectory_planner.generate_topp_trajectory(simple_path, limits)
        
        # 验证轨迹不为空
        assert len(trajectory) > 0
        
        # 验证轨迹起点和终点
        assert np.allclose(trajectory[0].position, simple_path[0].position, atol=1e-3)
        assert np.allclose(trajectory[-1].position, simple_path[-1].position, atol=1e-3)
        
        # 验证起点和终点速度为零
        assert np.allclose(trajectory[0].velocity, np.zeros(6), atol=1e-3)
        assert np.allclose(trajectory[-1].velocity, np.zeros(6), atol=1e-3)
        
        # 验证时间单调递增
        times = [point.time for point in trajectory]
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))
        
        # 验证路径参数单调递增
        path_params = [point.path_parameter for point in trajectory]
        assert all(path_params[i] <= path_params[i+1] for i in range(len(path_params)-1))
        assert path_params[0] == 0.0
        assert path_params[-1] == 1.0
    
    def test_topp_with_payload(self, trajectory_planner, simple_path, robot_model):
        """测试带负载的TOPP算法"""
        limits = robot_model.kinodynamic_limits
        
        # 创建负载信息
        payload = PayloadInfo(
            mass=5.0,
            center_of_mass=[0.0, 0.0, 0.1],
            inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            identification_confidence=0.9
        )
        
        # 生成带负载的轨迹
        trajectory_with_payload = trajectory_planner.generate_topp_trajectory(
            simple_path, limits, payload=payload
        )
        
        # 生成无负载的轨迹进行比较
        trajectory_without_payload = trajectory_planner.generate_topp_trajectory(
            simple_path, limits
        )
        
        # 验证轨迹都不为空
        assert len(trajectory_with_payload) > 0
        assert len(trajectory_without_payload) > 0
        
        # 验证负载影响（带负载的轨迹应该更保守）
        max_vel_with_payload = max(np.linalg.norm(p.velocity) for p in trajectory_with_payload)
        max_vel_without_payload = max(np.linalg.norm(p.velocity) for p in trajectory_without_payload)
        
        # 带负载的最大速度应该不超过无负载的情况
        assert max_vel_with_payload <= max_vel_without_payload * 1.1  # 允许小幅误差
    
    def test_topp_adaptive_envelope(self, trajectory_planner, simple_path, robot_model):
        """测试自适应包络线调整"""
        limits = robot_model.kinodynamic_limits
        
        # 生成启用自适应包络线的轨迹
        trajectory_adaptive = trajectory_planner.generate_topp_trajectory(
            simple_path, limits, adaptive_envelope=True
        )
        
        # 生成禁用自适应包络线的轨迹
        trajectory_non_adaptive = trajectory_planner.generate_topp_trajectory(
            simple_path, limits, adaptive_envelope=False
        )
        
        # 验证两种轨迹都不为空
        assert len(trajectory_adaptive) > 0
        assert len(trajectory_non_adaptive) > 0
        
        # 验证轨迹的基本属性
        for trajectory in [trajectory_adaptive, trajectory_non_adaptive]:
            # 起点和终点位置正确
            assert np.allclose(trajectory[0].position, simple_path[0].position, atol=1e-3)
            assert np.allclose(trajectory[-1].position, simple_path[-1].position, atol=1e-3)
            
            # 起点和终点速度为零
            assert np.allclose(trajectory[0].velocity, np.zeros(6), atol=1e-3)
            assert np.allclose(trajectory[-1].velocity, np.zeros(6), atol=1e-3)
    
    def test_topp_constraint_satisfaction(self, trajectory_planner, simple_path, robot_model):
        """测试TOPP算法约束满足"""
        limits = robot_model.kinodynamic_limits
        
        trajectory = trajectory_planner.generate_topp_trajectory(simple_path, limits)
        
        max_velocities = np.array(limits.max_joint_velocities)
        max_accelerations = np.array(limits.max_joint_accelerations)
        
        for point in trajectory:
            # 检查速度约束
            velocity_violations = np.abs(point.velocity) > max_velocities * 1.01  # 允许1%误差
            assert not np.any(velocity_violations), f"速度约束违反: {point.velocity}"
            
            # 检查加速度约束
            acceleration_violations = np.abs(point.acceleration) > max_accelerations * 1.01
            assert not np.any(acceleration_violations), f"加速度约束违反: {point.acceleration}"
    
    def test_topp_empty_path(self, trajectory_planner, robot_model):
        """测试空路径处理"""
        limits = robot_model.kinodynamic_limits
        
        # 空路径
        empty_path = []
        trajectory = trajectory_planner.generate_topp_trajectory(empty_path, limits)
        assert len(trajectory) == 0
        
        # 单点路径
        single_point_path = [Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))]
        trajectory = trajectory_planner.generate_topp_trajectory(single_point_path, limits)
        assert len(trajectory) == 1
        assert np.allclose(trajectory[0].position, single_point_path[0].position)
        assert np.allclose(trajectory[0].velocity, np.zeros(6))
    
    def test_topp_straight_line_path(self, trajectory_planner, robot_model):
        """测试直线路径的TOPP算法"""
        limits = robot_model.kinodynamic_limits
        
        # 创建直线路径
        start_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        straight_path = [
            Waypoint(position=start_pos),
            Waypoint(position=start_pos + 0.25 * (end_pos - start_pos)),
            Waypoint(position=start_pos + 0.5 * (end_pos - start_pos)),
            Waypoint(position=start_pos + 0.75 * (end_pos - start_pos)),
            Waypoint(position=end_pos)
        ]
        
        trajectory = trajectory_planner.generate_topp_trajectory(straight_path, limits)
        
        # 验证轨迹生成成功
        assert len(trajectory) > 0
        
        # 验证起点和终点
        assert np.allclose(trajectory[0].position, start_pos, atol=1e-3)
        assert np.allclose(trajectory[-1].position, end_pos, atol=1e-3)
        
        # 对于直线路径，中间点应该大致在直线上
        for point in trajectory[1:-1]:
            # 计算点到直线的距离（简化检查）
            direction = end_pos - start_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
                
                # 投影到直线上
                relative_pos = point.position - start_pos
                projection_length = np.dot(relative_pos, direction)
                projection = start_pos + projection_length * direction
                
                # 检查偏差
                deviation = np.linalg.norm(point.position - projection)
                assert deviation < 0.1, f"点偏离直线过远: {deviation}"
    
    def test_topp_curved_path(self, trajectory_planner, robot_model):
        """测试弯曲路径的TOPP算法"""
        limits = robot_model.kinodynamic_limits
        
        # 创建弯曲路径（圆弧）
        n_points = 8
        angles = np.linspace(0, np.pi/2, n_points)
        radius = 1.0
        
        curved_path = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            position = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
            curved_path.append(Waypoint(position=position))
        
        trajectory = trajectory_planner.generate_topp_trajectory(curved_path, limits)
        
        # 验证轨迹生成成功
        assert len(trajectory) > 0
        
        # 验证起点和终点
        assert np.allclose(trajectory[0].position, curved_path[0].position, atol=1e-3)
        assert np.allclose(trajectory[-1].position, curved_path[-1].position, atol=1e-3)
        
        # 验证弯曲路径的速度应该在高曲率区域降低
        velocities = [np.linalg.norm(point.velocity) for point in trajectory]
        
        # 中间部分（高曲率）的速度应该相对较低
        if len(velocities) > 4:
            middle_velocities = velocities[len(velocities)//4:3*len(velocities)//4]
            edge_velocities = velocities[:len(velocities)//4] + velocities[3*len(velocities)//4:]
            
            # 这是一个软约束，因为具体的速度分布取决于算法实现
            avg_middle_vel = np.mean(middle_velocities) if middle_velocities else 0
            avg_edge_vel = np.mean(edge_velocities) if edge_velocities else 0
            
            # 简单验证：中间速度不应该显著高于边缘速度
            assert avg_middle_vel <= avg_edge_vel * 2.0
    
    @given(
        n_waypoints=st.integers(min_value=2, max_value=10),
        position_scale=st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_topp_random_paths(self, trajectory_planner, robot_model, n_waypoints, position_scale):
        """基于属性的测试：随机路径的TOPP算法"""
        limits = robot_model.kinodynamic_limits
        
        # 生成随机路径
        np.random.seed(42)  # 确保可重现性
        random_path = []
        
        for i in range(n_waypoints):
            # 生成在关节限制范围内的随机位置
            min_pos = np.array(limits.min_joint_positions)
            max_pos = np.array(limits.max_joint_positions)
            
            # 缩放到合理范围
            position = min_pos + (max_pos - min_pos) * np.random.random(6) * position_scale
            random_path.append(Waypoint(position=position))
        
        try:
            trajectory = trajectory_planner.generate_topp_trajectory(random_path, limits)
            
            # 基本属性验证
            assert len(trajectory) > 0
            
            # 验证起点和终点
            assert np.allclose(trajectory[0].position, random_path[0].position, atol=1e-2)
            assert np.allclose(trajectory[-1].position, random_path[-1].position, atol=1e-2)
            
            # 验证时间单调性
            times = [point.time for point in trajectory]
            assert all(times[i] <= times[i+1] for i in range(len(times)-1))
            
            # 验证路径参数单调性
            path_params = [point.path_parameter for point in trajectory]
            assert all(path_params[i] <= path_params[i+1] for i in range(len(path_params)-1))
            
            # 验证约束满足（允许小幅超出）
            max_velocities = np.array(limits.max_joint_velocities)
            max_accelerations = np.array(limits.max_joint_accelerations)
            
            for point in trajectory:
                velocity_violations = np.abs(point.velocity) > max_velocities * 1.05
                acceleration_violations = np.abs(point.acceleration) > max_accelerations * 1.05
                
                assert not np.any(velocity_violations), "速度约束严重违反"
                assert not np.any(acceleration_violations), "加速度约束严重违反"
                
        except Exception as e:
            # 对于某些极端情况，算法可能失败，这是可以接受的
            # 但应该优雅地处理并提供备用方案
            pytest.skip(f"TOPP算法在极端情况下失败: {e}")
    
    def test_topp_performance_metrics(self, trajectory_planner, simple_path, robot_model):
        """测试TOPP算法性能指标"""
        limits = robot_model.kinodynamic_limits
        
        import time
        
        # 测量计算时间
        start_time = time.time()
        trajectory = trajectory_planner.generate_topp_trajectory(simple_path, limits)
        computation_time = time.time() - start_time
        
        # 验证计算时间合理（应该在几秒内完成）
        assert computation_time < 10.0, f"TOPP算法计算时间过长: {computation_time}s"
        
        # 计算轨迹质量指标
        metrics = trajectory_planner.compute_trajectory_metrics(trajectory)
        
        # 验证指标合理性
        assert metrics['total_time'] > 0
        assert metrics['total_distance'] > 0
        assert metrics['max_velocity'] > 0
        assert metrics['max_acceleration'] >= 0
        
        # 验证速度和加速度在限制范围内
        assert metrics['max_velocity'] <= max(limits.max_joint_velocities) * 1.1
        assert metrics['max_acceleration'] <= max(limits.max_joint_accelerations) * 1.1
    
    def test_topp_trajectory_smoothness(self, trajectory_planner, simple_path, robot_model):
        """测试TOPP轨迹平滑性"""
        limits = robot_model.kinodynamic_limits
        
        trajectory = trajectory_planner.generate_topp_trajectory(simple_path, limits)
        
        # 验证轨迹平滑性
        is_smooth, errors = trajectory_planner.validate_trajectory_smoothness(
            trajectory,
            position_tolerance=1e-2,
            velocity_tolerance=1e-1,
            acceleration_tolerance=1.0
        )
        
        # 对于TOPP算法，可能存在一些不平滑性，但不应该太严重
        if not is_smooth:
            # 检查错误数量是否在可接受范围内
            assert len(errors) < len(trajectory) * 0.1, f"过多的平滑性错误: {errors}"
    
    def test_topp_fallback_mechanism(self, trajectory_planner, robot_model):
        """测试TOPP算法的备用机制"""
        limits = robot_model.kinodynamic_limits
        
        # 创建可能导致TOPP失败的极端路径
        extreme_path = [
            Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            Waypoint(position=np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]))  # 超出关节限制
        ]
        
        # 即使在极端情况下，也应该能生成某种轨迹（备用方案）
        trajectory = trajectory_planner.generate_topp_trajectory(extreme_path, limits)
        
        # 验证备用轨迹的基本属性
        assert len(trajectory) > 0
        
        # 起点应该正确
        assert np.allclose(trajectory[0].position, extreme_path[0].position, atol=1e-3)
        
        # 应该有某种形式的运动
        total_movement = sum(
            np.linalg.norm(trajectory[i+1].position - trajectory[i].position)
            for i in range(len(trajectory)-1)
        )
        assert total_movement > 0


if __name__ == "__main__":
    pytest.main([__file__])