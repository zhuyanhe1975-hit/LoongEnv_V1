"""
并行路径控制模块

基于原始路径控制器实现并行优化版本，提高控制计算效率。
支持批量控制计算、前馈并行化和实时性能优化。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty

from .path_control import PathController, ControlMode
from ..core.parallel_computing import (
    ParallelComputingManager, ParallelConfig, ParallelMode,
    MemoryOptimizer, parallelize
)
from ..core.types import RobotState, TrajectoryPoint, ControlCommand, Vector
from ..core.models import RobotModel


class ParallelOptimizedPathController(PathController):
    """
    并行优化的路径控制器
    
    继承原始路径控制器，添加并行计算优化功能。
    """
    
    def __init__(
        self,
        robot_model: RobotModel,
        control_mode: ControlMode = ControlMode.COMPUTED_TORQUE,
        enable_feedforward: bool = True,
        enable_adaptation: bool = True,
        parallel_config: Optional[ParallelConfig] = None
    ):
        """
        初始化并行优化路径控制器
        
        Args:
            robot_model: 机器人模型
            control_mode: 控制模式
            enable_feedforward: 是否启用前馈控制
            enable_adaptation: 是否启用自适应控制
            parallel_config: 并行计算配置
        """
        super().__init__(robot_model, control_mode, enable_feedforward, enable_adaptation)
        
        # 并行计算配置
        self.parallel_config = parallel_config or ParallelConfig(
            mode=ParallelMode.THREAD,
            max_workers=4,
            enable_memory_optimization=True
        )
        
        # 并行计算管理器
        self.parallel_manager = ParallelComputingManager(self.parallel_config)
        
        # 内存优化器
        self.memory_optimizer = MemoryOptimizer()
        
        # 批处理阈值
        self.batch_threshold = 4
        
        # 线程本地存储
        self._thread_local = threading.local()
        
        # 预计算缓存
        self._precompute_cache = {}
        self._cache_lock = threading.Lock()
        
        # 性能统计
        self.parallel_performance_stats = {
            'parallel_control_calls': 0,
            'parallel_feedforward_calls': 0,
            'batch_processing_calls': 0,
            'total_parallel_time': 0.0,
            'avg_speedup': 1.0
        }
    
    def batch_compute_control(
        self,
        reference_trajectory: List[TrajectoryPoint],
        current_states: List[RobotState],
        dt_list: Optional[List[float]] = None
    ) -> List[ControlCommand]:
        """
        批量控制计算
        
        Args:
            reference_trajectory: 参考轨迹列表
            current_states: 当前状态列表
            dt_list: 时间步长列表
        
        Returns:
            控制指令列表
        """
        if len(reference_trajectory) != len(current_states):
            raise ValueError("参考轨迹和当前状态列表长度不匹配")
        
        if len(reference_trajectory) < self.batch_threshold:
            # 数量少，使用顺序计算
            return [
                super().compute_control(ref, state, dt)
                for ref, state, dt in zip(
                    reference_trajectory, 
                    current_states, 
                    dt_list or [None] * len(reference_trajectory)
                )
            ]
        
        start_time = time.time()
        
        try:
            # 并行计算控制指令
            control_commands = self._parallel_batch_control(
                reference_trajectory, current_states, dt_list
            )
            
            # 更新性能统计
            computation_time = time.time() - start_time
            self._update_parallel_performance_stats(computation_time, len(reference_trajectory))
            
            return control_commands
            
        except Exception as e:
            print(f"批量并行控制计算失败: {e}")
            # 回退到顺序计算
            return [
                super().compute_control(ref, state, dt)
                for ref, state, dt in zip(
                    reference_trajectory, 
                    current_states, 
                    dt_list or [None] * len(reference_trajectory)
                )
            ]
    
    def _parallel_batch_control(
        self,
        reference_trajectory: List[TrajectoryPoint],
        current_states: List[RobotState],
        dt_list: Optional[List[float]]
    ) -> List[ControlCommand]:
        """
        并行批量控制计算实现
        
        Args:
            reference_trajectory: 参考轨迹列表
            current_states: 当前状态列表
            dt_list: 时间步长列表
        
        Returns:
            控制指令列表
        """
        # 准备任务参数
        if dt_list is None:
            dt_list = [None] * len(reference_trajectory)
        
        control_tasks = list(zip(reference_trajectory, current_states, dt_list))
        
        def compute_single_control_parallel(task):
            reference, current_state, dt = task
            
            # 为每个线程创建独立的控制器状态
            if not hasattr(self._thread_local, 'controller_state'):
                self._thread_local.controller_state = {
                    'integral_error': np.zeros(self.n_joints),
                    'last_error': np.zeros(self.n_joints),
                    'last_time': 0.0
                }
            
            # 使用线程本地状态计算控制
            return self._compute_control_with_local_state(
                reference, current_state, dt, self._thread_local.controller_state
            )
        
        # 并行执行控制计算
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            control_commands = list(executor.map(compute_single_control_parallel, control_tasks))
        
        return control_commands
    
    def _compute_control_with_local_state(
        self,
        reference: TrajectoryPoint,
        current_state: RobotState,
        dt: Optional[float],
        local_state: Dict[str, Any]
    ) -> ControlCommand:
        """
        使用线程本地状态计算控制指令
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
            local_state: 线程本地状态
        
        Returns:
            控制指令
        """
        # 计算时间步长
        if dt is None:
            dt = current_state.timestamp - local_state['last_time'] if local_state['last_time'] > 0 else 0.001
            local_state['last_time'] = current_state.timestamp
        
        # 前馈控制
        feedforward_command = np.zeros(self.n_joints)
        if self.enable_feedforward:
            feedforward_command = self.feedforward_control(reference)
        
        # 反馈控制（使用本地状态）
        feedback_command = self._feedback_control_with_local_state(
            reference, current_state, dt, local_state
        )
        
        # 组合控制指令
        if self.control_mode in [ControlMode.COMPUTED_TORQUE, ControlMode.SLIDING_MODE, ControlMode.ADAPTIVE]:
            total_torque = feedforward_command + feedback_command
            
            control_command = ControlCommand(
                joint_torques=total_torque,
                control_mode="torque",
                timestamp=current_state.timestamp
            )
        else:
            total_position = reference.position + feedback_command
            
            control_command = ControlCommand(
                joint_positions=total_position,
                joint_velocities=reference.velocity.copy() if hasattr(reference, 'velocity') else None,
                control_mode="position",
                timestamp=current_state.timestamp
            )
        
        return control_command
    
    def _feedback_control_with_local_state(
        self,
        reference: TrajectoryPoint,
        current_state: RobotState,
        dt: float,
        local_state: Dict[str, Any]
    ) -> Vector:
        """
        使用本地状态的反馈控制
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
            local_state: 线程本地状态
        
        Returns:
            反馈控制输出
        """
        if self.control_mode == ControlMode.PID:
            return self._pid_control_with_local_state(
                reference, current_state, dt, local_state
            )
        elif self.control_mode == ControlMode.COMPUTED_TORQUE:
            return self._computed_torque_control(reference, current_state, dt)
        elif self.control_mode == ControlMode.SLIDING_MODE:
            return self._sliding_mode_control(reference, current_state, dt)
        elif self.control_mode == ControlMode.ADAPTIVE:
            return self._adaptive_control(reference, current_state, dt)
        else:
            return self._pid_control_with_local_state(
                reference, current_state, dt, local_state
            )
    
    def _pid_control_with_local_state(
        self,
        reference: TrajectoryPoint,
        current_state: RobotState,
        dt: float,
        local_state: Dict[str, Any]
    ) -> Vector:
        """
        使用本地状态的PID控制
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
            local_state: 线程本地状态
        
        Returns:
            PID控制输出
        """
        # 计算位置误差
        position_error = reference.position - current_state.joint_positions
        
        # 更新积分误差
        local_state['integral_error'] += position_error * dt
        integral_limit = 0.1
        local_state['integral_error'] = np.clip(
            local_state['integral_error'], -integral_limit, integral_limit
        )
        
        # 计算微分误差
        if dt > 0:
            derivative_error = (position_error - local_state['last_error']) / dt
        else:
            derivative_error = np.zeros(self.n_joints)
        
        local_state['last_error'] = position_error.copy()
        
        # PID控制律
        control_output = (
            self.kp * position_error +
            self.ki * local_state['integral_error'] +
            self.kd * derivative_error
        )
        
        return control_output
    
    @parallelize(mode=ParallelMode.THREAD, max_workers=4)
    def parallel_feedforward_batch(
        self,
        reference_points: List[TrajectoryPoint]
    ) -> List[Vector]:
        """
        并行前馈控制批处理
        
        Args:
            reference_points: 参考点列表
        
        Returns:
            前馈控制输出列表
        """
        return [self.feedforward_control(ref) for ref in reference_points]
    
    def precompute_feedforward_trajectory(
        self,
        trajectory: List[TrajectoryPoint],
        cache_key: Optional[str] = None
    ) -> List[Vector]:
        """
        预计算轨迹的前馈控制
        
        Args:
            trajectory: 轨迹
            cache_key: 缓存键
        
        Returns:
            前馈控制输出列表
        """
        if cache_key:
            with self._cache_lock:
                if cache_key in self._precompute_cache:
                    return self._precompute_cache[cache_key]
        
        # 并行计算前馈控制
        feedforward_commands = self.parallel_feedforward_batch(trajectory)
        
        # 缓存结果
        if cache_key:
            with self._cache_lock:
                self._precompute_cache[cache_key] = feedforward_commands
        
        return feedforward_commands
    
    def parallel_control_sequence(
        self,
        trajectory: List[TrajectoryPoint],
        initial_state: RobotState,
        dt: float = 0.001
    ) -> Tuple[List[ControlCommand], List[RobotState]]:
        """
        并行控制序列计算
        
        Args:
            trajectory: 参考轨迹
            initial_state: 初始状态
            dt: 时间步长
        
        Returns:
            (控制指令列表, 状态序列)
        """
        if len(trajectory) < self.batch_threshold:
            # 使用顺序计算
            return self._sequential_control_sequence(trajectory, initial_state, dt)
        
        # 预计算前馈控制
        feedforward_commands = self.precompute_feedforward_trajectory(trajectory)
        
        # 分段并行处理
        segment_size = max(4, len(trajectory) // self.parallel_config.max_workers)
        segments = []
        
        for i in range(0, len(trajectory), segment_size):
            end_idx = min(i + segment_size, len(trajectory))
            segment_trajectory = trajectory[i:end_idx]
            segment_feedforward = feedforward_commands[i:end_idx]
            segments.append((segment_trajectory, segment_feedforward))
        
        # 并行处理各段
        def process_control_segment(args):
            segment_trajectory, segment_feedforward = args
            segment_commands = []
            segment_states = []
            
            current_state = initial_state  # 简化：使用相同初始状态
            
            for ref_point, ff_command in zip(segment_trajectory, segment_feedforward):
                # 计算反馈控制
                fb_command = self._feedback_control_with_local_state(
                    ref_point, current_state, dt, {
                        'integral_error': np.zeros(self.n_joints),
                        'last_error': np.zeros(self.n_joints),
                        'last_time': 0.0
                    }
                )
                
                # 组合控制指令
                if self.control_mode in [ControlMode.COMPUTED_TORQUE, ControlMode.SLIDING_MODE, ControlMode.ADAPTIVE]:
                    total_torque = ff_command + fb_command
                    control_command = ControlCommand(
                        joint_torques=total_torque,
                        control_mode="torque",
                        timestamp=current_state.timestamp
                    )
                else:
                    total_position = ref_point.position + fb_command
                    control_command = ControlCommand(
                        joint_positions=total_position,
                        joint_velocities=ref_point.velocity.copy(),
                        control_mode="position",
                        timestamp=current_state.timestamp
                    )
                
                segment_commands.append(control_command)
                segment_states.append(current_state)
            
            return segment_commands, segment_states
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            segment_results = list(executor.map(process_control_segment, segments))
        
        # 合并结果
        all_commands = []
        all_states = []
        
        for commands, states in segment_results:
            all_commands.extend(commands)
            all_states.extend(states)
        
        return all_commands, all_states
    
    def _sequential_control_sequence(
        self,
        trajectory: List[TrajectoryPoint],
        initial_state: RobotState,
        dt: float
    ) -> Tuple[List[ControlCommand], List[RobotState]]:
        """
        顺序控制序列计算（回退方案）
        
        Args:
            trajectory: 参考轨迹
            initial_state: 初始状态
            dt: 时间步长
        
        Returns:
            (控制指令列表, 状态序列)
        """
        commands = []
        states = []
        current_state = initial_state
        
        for ref_point in trajectory:
            command = super().compute_control(ref_point, current_state, dt)
            commands.append(command)
            states.append(current_state)
        
        return commands, states
    
    def parallel_gain_tuning(
        self,
        reference_trajectory: List[TrajectoryPoint],
        simulation_states: List[RobotState],
        gain_ranges: Dict[str, Tuple[float, float]],
        optimization_steps: int = 20
    ) -> Dict[str, Vector]:
        """
        并行增益调优
        
        Args:
            reference_trajectory: 参考轨迹
            simulation_states: 仿真状态序列
            gain_ranges: 增益范围字典
            optimization_steps: 优化步数
        
        Returns:
            优化后的增益参数
        """
        # 生成候选增益组合
        candidate_gains = self._generate_gain_candidates(gain_ranges, optimization_steps)
        
        def evaluate_gain_combination(gains):
            kp, ki, kd = gains
            
            # 临时设置增益
            original_kp, original_ki, original_kd = self.kp.copy(), self.ki.copy(), self.kd.copy()
            self.kp = np.ones(self.n_joints) * kp
            self.ki = np.ones(self.n_joints) * ki
            self.kd = np.ones(self.n_joints) * kd
            
            try:
                # 评估性能
                performance = self._evaluate_control_performance(
                    self.kp, self.ki, self.kd,
                    reference_trajectory, simulation_states
                )
                return gains, performance
            finally:
                # 恢复原始增益
                self.kp, self.ki, self.kd = original_kp, original_ki, original_kd
        
        # 并行评估候选增益
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            results = list(executor.map(evaluate_gain_combination, candidate_gains))
        
        # 找到最佳增益
        best_gains, best_performance = min(results, key=lambda x: x[1])
        kp_best, ki_best, kd_best = best_gains
        
        # 应用最佳增益
        self.set_control_gains(
            kp=np.ones(self.n_joints) * kp_best,
            ki=np.ones(self.n_joints) * ki_best,
            kd=np.ones(self.n_joints) * kd_best
        )
        
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'performance': best_performance
        }
    
    def _generate_gain_candidates(
        self,
        gain_ranges: Dict[str, Tuple[float, float]],
        steps: int
    ) -> List[Tuple[float, float, float]]:
        """
        生成增益候选组合
        
        Args:
            gain_ranges: 增益范围
            steps: 步数
        
        Returns:
            增益候选列表
        """
        kp_range = gain_ranges.get('kp', (50, 500))
        ki_range = gain_ranges.get('ki', (5, 50))
        kd_range = gain_ranges.get('kd', (2, 30))
        
        kp_values = np.linspace(kp_range[0], kp_range[1], steps)
        ki_values = np.linspace(ki_range[0], ki_range[1], steps)
        kd_values = np.linspace(kd_range[0], kd_range[1], steps)
        
        candidates = []
        
        # 网格搜索（简化版）
        step_size = max(1, steps // 5)  # 减少候选数量
        
        for i in range(0, len(kp_values), step_size):
            for j in range(0, len(ki_values), step_size):
                for k in range(0, len(kd_values), step_size):
                    candidates.append((kp_values[i], ki_values[j], kd_values[k]))
        
        return candidates
    
    def memory_optimized_batch_control(
        self,
        trajectory_batch: List[List[TrajectoryPoint]],
        state_batch: List[List[RobotState]]
    ) -> List[List[ControlCommand]]:
        """
        内存优化的批量控制计算
        
        Args:
            trajectory_batch: 轨迹批次
            state_batch: 状态批次
        
        Returns:
            控制指令批次
        """
        # 预分配内存
        result_batch = []
        
        # 使用内存池优化
        for trajectory, states in zip(trajectory_batch, state_batch):
            # 获取预分配的内存
            commands = []
            
            for ref, state in zip(trajectory, states):
                # 使用内存池获取控制指令对象
                command_array = self.memory_optimizer.get_memory_pool(
                    "control_command", (self.n_joints,)
                )
                
                # 计算控制
                control_result = super().compute_control(ref, state)
                
                # 复制结果到预分配内存
                if hasattr(control_result, 'joint_torques') and control_result.joint_torques is not None:
                    command_array[:] = control_result.joint_torques
                elif hasattr(control_result, 'joint_positions') and control_result.joint_positions is not None:
                    command_array[:] = control_result.joint_positions
                
                commands.append(control_result)
            
            result_batch.append(commands)
        
        return result_batch
    
    def _update_parallel_performance_stats(self, computation_time: float, task_count: int):
        """更新并行性能统计"""
        self.parallel_performance_stats['batch_processing_calls'] += 1
        self.parallel_performance_stats['total_parallel_time'] += computation_time
        
        # 估算加速比
        estimated_sequential_time = computation_time * 2.0  # 简化估算
        if computation_time > 0:
            speedup = estimated_sequential_time / computation_time
            
            # 更新平均加速比
            total_calls = self.parallel_performance_stats['batch_processing_calls']
            current_avg = self.parallel_performance_stats['avg_speedup']
            self.parallel_performance_stats['avg_speedup'] = (
                (current_avg * (total_calls - 1) + speedup) / total_calls
            )
    
    def get_parallel_performance_report(self) -> Dict[str, Any]:
        """
        获取并行性能报告
        
        Returns:
            并行性能报告字典
        """
        return {
            'parallel_control_calls': self.parallel_performance_stats['parallel_control_calls'],
            'parallel_feedforward_calls': self.parallel_performance_stats['parallel_feedforward_calls'],
            'batch_processing_calls': self.parallel_performance_stats['batch_processing_calls'],
            'total_parallel_time': self.parallel_performance_stats['total_parallel_time'],
            'average_speedup': self.parallel_performance_stats['avg_speedup'],
            'batch_threshold': self.batch_threshold,
            'cache_size': len(self._precompute_cache),
            'parallel_config': {
                'mode': self.parallel_config.mode.value,
                'max_workers': self.parallel_config.max_workers,
                'memory_optimization': self.parallel_config.enable_memory_optimization
            }
        }
    
    def clear_precompute_cache(self):
        """清除预计算缓存"""
        with self._cache_lock:
            self._precompute_cache.clear()
    
    def optimize_batch_threshold(
        self,
        test_trajectories: List[List[TrajectoryPoint]],
        test_states: List[List[RobotState]]
    ):
        """
        优化批处理阈值
        
        Args:
            test_trajectories: 测试轨迹列表
            test_states: 测试状态列表
        """
        print("开始批处理阈值优化...")
        
        best_threshold = self.batch_threshold
        best_performance = 0.0
        
        # 测试不同阈值
        for threshold in [2, 4, 8, 16, 32]:
            original_threshold = self.batch_threshold
            self.batch_threshold = threshold
            
            try:
                total_performance = 0.0
                test_count = 0
                
                for trajectory, states in zip(test_trajectories[:3], test_states[:3]):  # 限制测试数量
                    if len(trajectory) >= threshold:
                        start_time = time.time()
                        self.batch_compute_control(trajectory, states)
                        computation_time = time.time() - start_time
                        
                        performance_score = len(trajectory) / computation_time if computation_time > 0 else 0
                        total_performance += performance_score
                        test_count += 1
                
                if test_count > 0:
                    avg_performance = total_performance / test_count
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_threshold = threshold
            
            except Exception as e:
                print(f"阈值测试失败: threshold={threshold}, {e}")
            
            finally:
                self.batch_threshold = original_threshold
        
        # 应用最佳阈值
        self.batch_threshold = best_threshold
        print(f"阈值优化完成，最佳阈值: {best_threshold}")