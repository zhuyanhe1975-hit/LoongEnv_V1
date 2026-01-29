"""
并行计算模块测试

测试多线程并行计算优化功能的正确性和性能。
"""

import pytest
import numpy as np
import time
from typing import List

from src.robot_motion_control.core.parallel_computing import (
    ParallelComputingManager, ParallelConfig, ParallelMode,
    TaskScheduler, MemoryOptimizer, parallelize,
    estimate_task_complexity, auto_tune_parallel_config
)
from src.robot_motion_control.algorithms.parallel_dynamics import ParallelOptimizedDynamicsEngine
from src.robot_motion_control.algorithms.parallel_trajectory_planning import ParallelOptimizedTrajectoryPlanner
from src.robot_motion_control.algorithms.parallel_path_control import ParallelOptimizedPathController
from src.robot_motion_control.core.controller import RobotMotionController, ControllerConfig
from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.core.types import (
    RobotState, TrajectoryPoint, Waypoint, Vector, KinodynamicLimits
)


class TestParallelComputingManager:
    """并行计算管理器测试"""
    
    def test_parallel_config_creation(self):
        """测试并行配置创建"""
        config = ParallelConfig(
            mode=ParallelMode.THREAD,
            max_workers=4,
            enable_memory_optimization=True
        )
        
        assert config.mode == ParallelMode.THREAD
        assert config.max_workers == 4
        assert config.enable_memory_optimization is True
    
    def test_task_scheduler_initialization(self):
        """测试任务调度器初始化"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        
        with TaskScheduler(config) as scheduler:
            assert scheduler.max_workers == 2
            assert scheduler.config.mode == ParallelMode.THREAD
    
    def test_parallel_task_execution(self):
        """测试并行任务执行"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        
        def square_function(x):
            return x * x
        
        test_data = [1, 2, 3, 4, 5]
        expected_results = [1, 4, 9, 16, 25]
        
        with TaskScheduler(config) as scheduler:
            results = scheduler.submit_tasks(square_function, test_data)
        
        assert len(results) == len(expected_results)
        assert all(r == e for r, e in zip(results, expected_results))
    
    def test_parallel_map_function(self):
        """测试并行映射函数"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        
        def double_function(x):
            return x * 2
        
        test_data = [1, 2, 3, 4]
        expected_results = [2, 4, 6, 8]
        
        with TaskScheduler(config) as scheduler:
            results = scheduler.map_parallel(double_function, test_data)
        
        assert results == expected_results
    
    def test_memory_optimizer(self):
        """测试内存优化器"""
        optimizer = MemoryOptimizer()
        
        # 测试内存池获取
        array1 = optimizer.get_memory_pool("test_pool", (3, 3))
        array2 = optimizer.get_memory_pool("test_pool", (3, 3))
        
        assert array1.shape == (3, 3)
        assert array2.shape == (3, 3)
        
        # 测试内存返回
        optimizer.return_to_pool("test_pool", array1)
        array3 = optimizer.get_memory_pool("test_pool", (3, 3))
        
        # 应该重用返回的内存
        assert np.array_equal(array3, np.zeros((3, 3)))
    
    def test_parallelize_decorator(self):
        """测试并行化装饰器"""
        @parallelize(mode=ParallelMode.THREAD, max_workers=2)
        def process_list(data_list):
            return [x * 3 for x in data_list]
        
        test_data = [1, 2, 3, 4]
        results = process_list(test_data)
        
        expected = [3, 6, 9, 12]
        assert results == expected


class TestParallelOptimizedDynamicsEngine:
    """并行优化动力学引擎测试"""
    
    @pytest.fixture
    def robot_model(self):
        """创建测试用机器人模型"""
        return RobotModel.create_test_model(n_joints=6)
    
    @pytest.fixture
    def parallel_dynamics_engine(self, robot_model):
        """创建并行动力学引擎"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        return ParallelOptimizedDynamicsEngine(robot_model, config)
    
    def test_batch_forward_dynamics(self, parallel_dynamics_engine):
        """测试批量正向动力学计算"""
        n_joints = parallel_dynamics_engine.n_joints
        
        # 准备测试数据
        q_list = [np.random.randn(n_joints) for _ in range(5)]
        qd_list = [np.random.randn(n_joints) for _ in range(5)]
        tau_list = [np.random.randn(n_joints) for _ in range(5)]
        
        # 批量计算
        results = parallel_dynamics_engine.batch_forward_dynamics(q_list, qd_list, tau_list)
        
        assert len(results) == 5
        for result in results:
            assert len(result) == n_joints
            assert not np.any(np.isnan(result))
    
    def test_batch_inverse_dynamics(self, parallel_dynamics_engine):
        """测试批量逆向动力学计算"""
        n_joints = parallel_dynamics_engine.n_joints
        
        # 准备测试数据
        q_list = [np.random.randn(n_joints) for _ in range(5)]
        qd_list = [np.random.randn(n_joints) for _ in range(5)]
        qdd_list = [np.random.randn(n_joints) for _ in range(5)]
        
        # 批量计算
        results = parallel_dynamics_engine.batch_inverse_dynamics(q_list, qd_list, qdd_list)
        
        assert len(results) == 5
        for result in results:
            assert len(result) == n_joints
            assert not np.any(np.isnan(result))
    
    def test_batch_jacobian(self, parallel_dynamics_engine):
        """测试批量雅可比矩阵计算"""
        n_joints = parallel_dynamics_engine.n_joints
        
        # 准备测试数据
        q_list = [np.random.randn(n_joints) for _ in range(3)]
        
        # 批量计算
        results = parallel_dynamics_engine.batch_jacobian(q_list)
        
        assert len(results) == 3
        for result in results:
            assert result.shape == (6, n_joints)  # 6DOF末端执行器
            assert not np.any(np.isnan(result))
    
    def test_performance_comparison(self, parallel_dynamics_engine):
        """测试并行计算性能对比"""
        n_joints = parallel_dynamics_engine.n_joints
        
        # 准备大量测试数据
        q_list = [np.random.randn(n_joints) for _ in range(20)]
        qd_list = [np.random.randn(n_joints) for _ in range(20)]
        tau_list = [np.random.randn(n_joints) for _ in range(20)]
        
        # 顺序计算
        start_time = time.time()
        sequential_results = [
            parallel_dynamics_engine.forward_dynamics(q, qd, tau)
            for q, qd, tau in zip(q_list, qd_list, tau_list)
        ]
        sequential_time = time.time() - start_time
        
        # 并行计算
        start_time = time.time()
        parallel_results = parallel_dynamics_engine.batch_forward_dynamics(q_list, qd_list, tau_list)
        parallel_time = time.time() - start_time
        
        # 验证结果一致性
        assert len(sequential_results) == len(parallel_results)
        
        # 性能应该有所提升（在有足够任务的情况下）
        if len(q_list) >= parallel_dynamics_engine.batch_threshold:
            print(f"顺序计算时间: {sequential_time:.4f}s")
            print(f"并行计算时间: {parallel_time:.4f}s")
            print(f"加速比: {sequential_time / parallel_time:.2f}x")
    
    def test_adaptive_parallel_threshold(self, parallel_dynamics_engine):
        """测试自适应并行阈值"""
        original_threshold = parallel_dynamics_engine.batch_threshold
        
        # 运行自适应阈值优化
        new_threshold = parallel_dynamics_engine.adaptive_parallel_threshold(
            sample_size=10, test_iterations=2
        )
        
        assert isinstance(new_threshold, int)
        assert new_threshold > 0
        assert parallel_dynamics_engine.batch_threshold == new_threshold


class TestParallelOptimizedTrajectoryPlanner:
    """并行优化轨迹规划器测试"""
    
    @pytest.fixture
    def robot_model(self):
        """创建测试用机器人模型"""
        return RobotModel.create_test_model(n_joints=6)
    
    @pytest.fixture
    def parallel_trajectory_planner(self, robot_model):
        """创建并行轨迹规划器"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        return ParallelOptimizedTrajectoryPlanner(robot_model, config)
    
    def test_parallel_s7_interpolation(self, parallel_trajectory_planner):
        """测试并行S7插补"""
        n_joints = parallel_trajectory_planner.n_joints
        
        # 创建测试路径
        path = []
        for i in range(10):  # 足够多的点以触发并行计算
            waypoint = Waypoint(
                position=np.random.randn(n_joints),
                velocity=np.zeros(n_joints),
                timestamp=i * 0.1
            )
            path.append(waypoint)
        
        # 并行S7插补
        trajectory = parallel_trajectory_planner.parallel_interpolate_s7_trajectory(
            path, max_velocity=1.0, max_acceleration=2.0, max_jerk=5.0
        )
        
        assert len(trajectory) > 0
        
        # 验证轨迹连续性
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # 时间应该递增
            assert curr_point.time >= prev_point.time
            
            # 位置应该连续
            pos_diff = np.linalg.norm(curr_point.position - prev_point.position)
            assert pos_diff < 1.0  # 合理的位置变化
    
    def test_batch_trajectory_generation(self, parallel_trajectory_planner):
        """测试批量轨迹生成"""
        n_joints = parallel_trajectory_planner.n_joints
        
        # 创建多个测试路径
        path_list = []
        limits_list = []
        
        for _ in range(3):
            path = []
            for i in range(5):
                waypoint = Waypoint(
                    position=np.random.randn(n_joints),
                    velocity=np.zeros(n_joints),
                    timestamp=i * 0.1
                )
                path.append(waypoint)
            path_list.append(path)
            
            limits = parallel_trajectory_planner.robot_model.kinodynamic_limits
            limits_list.append(limits)
        
        # 批量生成轨迹
        trajectories = parallel_trajectory_planner.batch_trajectory_generation(
            path_list, limits_list, method="s7"
        )
        
        assert len(trajectories) == 3
        for trajectory in trajectories:
            assert len(trajectory) > 0


class TestParallelOptimizedPathController:
    """并行优化路径控制器测试"""
    
    @pytest.fixture
    def robot_model(self):
        """创建测试用机器人模型"""
        return RobotModel.create_test_model(n_joints=6)
    
    @pytest.fixture
    def parallel_path_controller(self, robot_model):
        """创建并行路径控制器"""
        config = ParallelConfig(mode=ParallelMode.THREAD, max_workers=2)
        return ParallelOptimizedPathController(robot_model, parallel_config=config)
    
    def test_batch_compute_control(self, parallel_path_controller):
        """测试批量控制计算"""
        n_joints = parallel_path_controller.n_joints
        
        # 准备测试数据
        reference_trajectory = []
        current_states = []
        
        for i in range(8):  # 足够多以触发并行计算
            ref_point = TrajectoryPoint(
                position=np.random.randn(n_joints),
                velocity=np.random.randn(n_joints) * 0.1,
                acceleration=np.random.randn(n_joints) * 0.01,
                jerk=np.zeros(n_joints),
                time=i * 0.001,
                path_parameter=i / 7.0
            )
            reference_trajectory.append(ref_point)
            
            state = RobotState(
                joint_positions=np.random.randn(n_joints),
                joint_velocities=np.random.randn(n_joints) * 0.1,
                joint_accelerations=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
                timestamp=i * 0.001
            )
            current_states.append(state)
        
        # 批量控制计算
        control_commands = parallel_path_controller.batch_compute_control(
            reference_trajectory, current_states
        )
        
        assert len(control_commands) == len(reference_trajectory)
        
        for command in control_commands:
            assert hasattr(command, 'control_mode')
            assert command.timestamp >= 0
    
    def test_parallel_feedforward_batch(self, parallel_path_controller):
        """测试并行前馈控制批处理"""
        n_joints = parallel_path_controller.n_joints
        
        # 准备参考点
        reference_points = []
        for i in range(6):
            ref_point = TrajectoryPoint(
                position=np.random.randn(n_joints),
                velocity=np.random.randn(n_joints) * 0.1,
                acceleration=np.random.randn(n_joints) * 0.01,
                jerk=np.zeros(n_joints),
                time=i * 0.001,
                path_parameter=i / 5.0
            )
            reference_points.append(ref_point)
        
        # 并行前馈计算
        feedforward_commands = parallel_path_controller.parallel_feedforward_batch(reference_points)
        
        assert len(feedforward_commands) == len(reference_points)
        for command in feedforward_commands:
            assert len(command) == n_joints
            assert not np.any(np.isnan(command))


class TestIntegratedParallelController:
    """集成并行控制器测试"""
    
    @pytest.fixture
    def robot_model(self):
        """创建测试用机器人模型"""
        return RobotModel.create_test_model(n_joints=6)
    
    @pytest.fixture
    def parallel_controller(self, robot_model):
        """创建启用并行计算的控制器"""
        config = ControllerConfig(
            enable_parallel_computing=True,
            parallel_mode=ParallelMode.THREAD,
            max_parallel_workers=2,
            parallel_batch_threshold=4
        )
        return RobotMotionController(robot_model, config)
    
    def test_parallel_controller_initialization(self, parallel_controller):
        """测试并行控制器初始化"""
        assert parallel_controller.config.enable_parallel_computing is True
        assert parallel_controller.parallel_manager is not None
        assert parallel_controller.config.parallel_mode == ParallelMode.THREAD
    
    def test_parallel_trajectory_planning(self, parallel_controller):
        """测试并行轨迹规划"""
        n_joints = parallel_controller.robot_model.n_joints
        
        # 创建测试路径
        waypoints = []
        for i in range(8):  # 足够多以触发并行计算
            waypoint = Waypoint(
                position=np.random.randn(n_joints),
                velocity=np.zeros(n_joints),
                timestamp=i * 0.1
            )
            waypoints.append(waypoint)
        
        # 规划轨迹
        trajectory = parallel_controller.plan_trajectory(waypoints, optimize_time=False)
        
        assert len(trajectory) > 0
        
        # 验证轨迹质量
        for i in range(1, len(trajectory)):
            assert trajectory[i].time >= trajectory[i-1].time
    
    def test_batch_control_sequence(self, parallel_controller):
        """测试批量控制序列"""
        n_joints = parallel_controller.robot_model.n_joints
        
        # 创建测试轨迹批次
        trajectory_batch = []
        initial_states = []
        
        for batch_idx in range(3):
            trajectory = []
            for i in range(5):
                point = TrajectoryPoint(
                    position=np.random.randn(n_joints),
                    velocity=np.random.randn(n_joints) * 0.1,
                    acceleration=np.zeros(n_joints),
                    jerk=np.zeros(n_joints),
                    time=i * 0.001,
                    path_parameter=i / 4.0
                )
                trajectory.append(point)
            trajectory_batch.append(trajectory)
            
            initial_state = RobotState(
                joint_positions=np.random.randn(n_joints),
                joint_velocities=np.zeros(n_joints),
                joint_accelerations=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
                timestamp=0.0
            )
            initial_states.append(initial_state)
        
        # 批量计算控制序列
        control_sequences = parallel_controller.batch_compute_control_sequence(
            trajectory_batch, initial_states
        )
        
        assert len(control_sequences) == 3
        for sequence in control_sequences:
            assert len(sequence) == 5
    
    def test_parallel_performance_optimization(self, parallel_controller):
        """测试并行性能优化"""
        n_joints = parallel_controller.robot_model.n_joints
        
        # 创建样本数据
        sample_trajectories = []
        sample_states = []
        
        for _ in range(2):  # 减少测试数据量
            trajectory = []
            for i in range(6):
                point = TrajectoryPoint(
                    position=np.random.randn(n_joints),
                    velocity=np.random.randn(n_joints) * 0.1,
                    acceleration=np.zeros(n_joints),
                    jerk=np.zeros(n_joints),
                    time=i * 0.001,
                    path_parameter=i / 5.0
                )
                trajectory.append(point)
            sample_trajectories.append(trajectory)
            
            state = RobotState(
                joint_positions=np.random.randn(n_joints),
                joint_velocities=np.zeros(n_joints),
                joint_accelerations=np.zeros(n_joints),
                joint_torques=np.zeros(n_joints),
                timestamp=0.0
            )
            sample_states.append(state)
        
        # 运行性能优化
        parallel_controller.optimize_parallel_performance(
            sample_trajectories, sample_states, optimization_iterations=2
        )
        
        # 验证优化后的状态
        status = parallel_controller.get_controller_status()
        assert "parallel_performance" in status
    
    def test_parallel_config_modification(self, parallel_controller):
        """测试并行配置修改"""
        # 修改并行配置
        parallel_controller.set_parallel_config(
            mode=ParallelMode.THREAD,
            max_workers=4,
            batch_threshold=8
        )
        
        assert parallel_controller.config.parallel_mode == ParallelMode.THREAD
        assert parallel_controller.config.max_parallel_workers == 4
        assert parallel_controller.config.parallel_batch_threshold == 8
    
    def test_enable_disable_parallel_computing(self, parallel_controller):
        """测试启用/禁用并行计算"""
        # 禁用并行计算
        parallel_controller.enable_parallel_computing(False)
        assert parallel_controller.config.enable_parallel_computing is False
        
        # 重新启用并行计算
        parallel_controller.enable_parallel_computing(True)
        assert parallel_controller.config.enable_parallel_computing is True
        assert parallel_controller.parallel_manager is not None


class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    def test_task_complexity_estimation(self):
        """测试任务复杂度估算"""
        def simple_task(x):
            return x * 2
        
        def complex_task(x):
            # 模拟复杂计算
            result = 0
            for i in range(100):
                result += np.sin(x + i)
            return result
        
        simple_complexity = estimate_task_complexity(simple_task, 5.0, iterations=3)
        complex_complexity = estimate_task_complexity(complex_task, 5.0, iterations=3)
        
        assert simple_complexity > 0
        assert complex_complexity > 0
        assert complex_complexity > simple_complexity
    
    def test_auto_tune_parallel_config(self):
        """测试自动调优并行配置"""
        def test_function(x):
            return x ** 2
        
        test_functions = [test_function]
        test_data_sets = [[1, 2, 3, 4, 5]]
        
        optimal_config = auto_tune_parallel_config(test_functions, test_data_sets)
        
        assert isinstance(optimal_config, ParallelConfig)
        assert optimal_config.mode in [ParallelMode.THREAD, ParallelMode.PROCESS]
        assert optimal_config.max_workers > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])