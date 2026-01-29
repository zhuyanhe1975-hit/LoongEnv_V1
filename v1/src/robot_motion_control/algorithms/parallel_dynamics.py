"""
并行动力学计算模块

基于原始动力学引擎实现并行优化版本，提高计算效率。
支持批量计算、内存优化和任务调度。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .dynamics import DynamicsEngine
from ..core.parallel_computing import (
    ParallelComputingManager, ParallelConfig, ParallelMode,
    MemoryOptimizer, parallelize
)
from ..core.types import Vector, Matrix, PayloadInfo, AlgorithmError
from ..core.models import RobotModel


class ParallelOptimizedDynamicsEngine(DynamicsEngine):
    """
    并行优化的动力学引擎
    
    继承原始动力学引擎，添加并行计算优化功能。
    """
    
    def __init__(
        self,
        robot_model: RobotModel,
        parallel_config: Optional[ParallelConfig] = None
    ):
        """
        初始化并行优化动力学引擎
        
        Args:
            robot_model: 机器人模型
            parallel_config: 并行计算配置
        """
        super().__init__(robot_model)
        
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
        self.batch_threshold = 4  # 当任务数量超过此值时使用并行计算
        
        # 性能统计
        self.performance_stats = {
            'parallel_calls': 0,
            'sequential_calls': 0,
            'total_speedup': 0.0,
            'avg_speedup': 1.0
        }
    
    def batch_forward_dynamics(
        self,
        q_list: List[Vector],
        qd_list: List[Vector],
        tau_list: List[Vector]
    ) -> List[Vector]:
        """
        批量正向动力学计算
        
        Args:
            q_list: 关节位置列表
            qd_list: 关节速度列表
            tau_list: 关节力矩列表
        
        Returns:
            关节加速度列表
        """
        if len(q_list) < self.batch_threshold:
            # 任务数量少，使用顺序计算
            return [
                super().forward_dynamics(q, qd, tau)
                for q, qd, tau in zip(q_list, qd_list, tau_list)
            ]
        
        # 使用并行计算
        start_time = time.time()
        
        with self.parallel_manager as manager:
            parallel_dynamics = manager.create_parallel_dynamics_engine(self)
            
            states = list(zip(q_list, qd_list, tau_list))
            results = parallel_dynamics.parallel_forward_dynamics_batch(states)
        
        # 更新性能统计
        self._update_performance_stats(start_time, len(q_list), 'parallel')
        
        return results
    
    def batch_inverse_dynamics(
        self,
        q_list: List[Vector],
        qd_list: List[Vector],
        qdd_list: List[Vector]
    ) -> List[Vector]:
        """
        批量逆向动力学计算
        
        Args:
            q_list: 关节位置列表
            qd_list: 关节速度列表
            qdd_list: 关节加速度列表
        
        Returns:
            关节力矩列表
        """
        if len(q_list) < self.batch_threshold:
            return [
                super().inverse_dynamics(q, qd, qdd)
                for q, qd, qdd in zip(q_list, qd_list, qdd_list)
            ]
        
        start_time = time.time()
        
        with self.parallel_manager as manager:
            parallel_dynamics = manager.create_parallel_dynamics_engine(self)
            
            states = list(zip(q_list, qd_list, qdd_list))
            results = parallel_dynamics.parallel_inverse_dynamics_batch(states)
        
        self._update_performance_stats(start_time, len(q_list), 'parallel')
        
        return results
    
    def batch_jacobian(self, q_list: List[Vector]) -> List[Matrix]:
        """
        批量雅可比矩阵计算
        
        Args:
            q_list: 关节位置列表
        
        Returns:
            雅可比矩阵列表
        """
        if len(q_list) < self.batch_threshold:
            return [super().jacobian(q) for q in q_list]
        
        start_time = time.time()
        
        with self.parallel_manager as manager:
            parallel_dynamics = manager.create_parallel_dynamics_engine(self)
            results = parallel_dynamics.parallel_jacobian_batch(q_list)
        
        self._update_performance_stats(start_time, len(q_list), 'parallel')
        
        return results
    
    def batch_gravity_compensation(self, q_list: List[Vector]) -> List[Vector]:
        """
        批量重力补偿计算
        
        Args:
            q_list: 关节位置列表
        
        Returns:
            重力补偿力矩列表
        """
        if len(q_list) < self.batch_threshold:
            return [super().gravity_compensation(q) for q in q_list]
        
        start_time = time.time()
        
        with self.parallel_manager as manager:
            parallel_dynamics = manager.create_parallel_dynamics_engine(self)
            results = parallel_dynamics.parallel_gravity_compensation_batch(q_list)
        
        self._update_performance_stats(start_time, len(q_list), 'parallel')
        
        return results
    
    @parallelize(mode=ParallelMode.THREAD, max_workers=4)
    def parallel_friction_computation(
        self,
        qd_list: List[Vector],
        temperature: float = 20.0
    ) -> List[Vector]:
        """
        并行摩擦力计算
        
        Args:
            qd_list: 关节速度列表
            temperature: 环境温度
        
        Returns:
            摩擦力矩列表
        """
        return [
            super().compute_friction_torque(qd, temperature)
            for qd in qd_list
        ]
    
    def optimized_mass_matrix_batch(
        self,
        q_list: List[Vector],
        use_cache: bool = True
    ) -> List[Matrix]:
        """
        优化的批量质量矩阵计算
        
        Args:
            q_list: 关节位置列表
            use_cache: 是否使用缓存
        
        Returns:
            质量矩阵列表
        """
        if len(q_list) < self.batch_threshold:
            return [super().compute_mass_matrix(q) for q in q_list]
        
        # 内存优化：预分配结果数组
        results = []
        
        # 分批处理以优化内存使用
        batch_size = 16
        for i in range(0, len(q_list), batch_size):
            batch_q = q_list[i:i + batch_size]
            
            # 并行计算批次
            def compute_mass_matrix_single(q):
                return super(ParallelOptimizedDynamicsEngine, self).compute_mass_matrix(q)
            
            with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
                batch_results = list(executor.map(compute_mass_matrix_single, batch_q))
            
            results.extend(batch_results)
        
        return results
    
    def parallel_payload_effect_analysis(
        self,
        q_list: List[Vector],
        payload_list: List[PayloadInfo]
    ) -> List[Dict[str, Vector]]:
        """
        并行负载影响分析
        
        Args:
            q_list: 关节位置列表
            payload_list: 负载信息列表
        
        Returns:
            负载影响分析结果列表
        """
        def analyze_single_payload_effect(args):
            q, payload = args
            # 临时更新负载
            original_payload = self.robot_model.current_payload
            self.robot_model.current_payload = payload
            
            try:
                result = super(ParallelOptimizedDynamicsEngine, self).get_payload_effect_on_dynamics(q)
                return result
            finally:
                # 恢复原始负载
                self.robot_model.current_payload = original_payload
        
        if len(q_list) < self.batch_threshold:
            return [
                analyze_single_payload_effect((q, payload))
                for q, payload in zip(q_list, payload_list)
            ]
        
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            args_list = list(zip(q_list, payload_list))
            results = list(executor.map(analyze_single_payload_effect, args_list))
        
        return results
    
    def memory_optimized_dynamics_sequence(
        self,
        trajectory_states: List[Tuple[Vector, Vector, Vector]],
        computation_type: str = "forward"
    ) -> List[Vector]:
        """
        内存优化的动力学序列计算
        
        Args:
            trajectory_states: 轨迹状态列表
            computation_type: 计算类型 ("forward", "inverse", "gravity")
        
        Returns:
            计算结果列表
        """
        # 预分配内存
        n_states = len(trajectory_states)
        result_shape = (self.n_joints,)
        
        # 使用内存池
        results = []
        
        # 分块处理以控制内存使用
        chunk_size = min(32, max(4, n_states // 4))
        
        for i in range(0, n_states, chunk_size):
            chunk_states = trajectory_states[i:i + chunk_size]
            
            # 获取预分配的内存
            chunk_results = []
            for _ in range(len(chunk_states)):
                result_array = self.memory_optimizer.get_memory_pool(
                    f"dynamics_{computation_type}", result_shape
                )
                chunk_results.append(result_array)
            
            # 并行计算块
            if computation_type == "forward":
                computed_results = self.batch_forward_dynamics(
                    [s[0] for s in chunk_states],
                    [s[1] for s in chunk_states],
                    [s[2] for s in chunk_states]
                )
            elif computation_type == "inverse":
                computed_results = self.batch_inverse_dynamics(
                    [s[0] for s in chunk_states],
                    [s[1] for s in chunk_states],
                    [s[2] for s in chunk_states]
                )
            elif computation_type == "gravity":
                computed_results = self.batch_gravity_compensation(
                    [s[0] for s in chunk_states]
                )
            else:
                raise ValueError(f"未知的计算类型: {computation_type}")
            
            # 复制结果到预分配的内存
            for j, computed in enumerate(computed_results):
                if j < len(chunk_results):
                    chunk_results[j][:] = computed
            
            results.extend(chunk_results)
        
        return results
    
    def adaptive_parallel_threshold(
        self,
        sample_size: int = 10,
        test_iterations: int = 3
    ) -> int:
        """
        自适应确定并行计算阈值
        
        Args:
            sample_size: 样本大小
            test_iterations: 测试迭代次数
        
        Returns:
            最优并行阈值
        """
        # 生成测试数据
        test_q = [np.random.randn(self.n_joints) for _ in range(sample_size)]
        test_qd = [np.random.randn(self.n_joints) for _ in range(sample_size)]
        test_tau = [np.random.randn(self.n_joints) for _ in range(sample_size)]
        
        best_threshold = self.batch_threshold
        best_performance = 0.0
        
        # 测试不同阈值
        for threshold in [2, 4, 8, 16]:
            if threshold > sample_size:
                continue
            
            # 测试顺序执行
            sequential_times = []
            for _ in range(test_iterations):
                start_time = time.time()
                for q, qd, tau in zip(test_q[:threshold], test_qd[:threshold], test_tau[:threshold]):
                    super().forward_dynamics(q, qd, tau)
                sequential_times.append(time.time() - start_time)
            
            # 测试并行执行
            parallel_times = []
            original_threshold = self.batch_threshold
            self.batch_threshold = 1  # 强制使用并行
            
            for _ in range(test_iterations):
                start_time = time.time()
                self.batch_forward_dynamics(
                    test_q[:threshold], test_qd[:threshold], test_tau[:threshold]
                )
                parallel_times.append(time.time() - start_time)
            
            self.batch_threshold = original_threshold
            
            # 计算性能提升
            avg_sequential = np.mean(sequential_times)
            avg_parallel = np.mean(parallel_times)
            
            if avg_parallel > 0:
                speedup = avg_sequential / avg_parallel
                if speedup > best_performance:
                    best_performance = speedup
                    best_threshold = threshold
        
        self.batch_threshold = best_threshold
        return best_threshold
    
    def _update_performance_stats(
        self,
        start_time: float,
        task_count: int,
        execution_type: str
    ):
        """更新性能统计"""
        execution_time = time.time() - start_time
        
        if execution_type == 'parallel':
            self.performance_stats['parallel_calls'] += 1
            
            # 估算顺序执行时间
            estimated_sequential_time = execution_time * 2.0  # 简化估算
            if execution_time > 0:
                speedup = estimated_sequential_time / execution_time
                self.performance_stats['total_speedup'] += speedup
                
                # 更新平均加速比
                total_calls = self.performance_stats['parallel_calls']
                self.performance_stats['avg_speedup'] = (
                    self.performance_stats['total_speedup'] / total_calls
                )
        else:
            self.performance_stats['sequential_calls'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            性能报告字典
        """
        total_calls = (
            self.performance_stats['parallel_calls'] +
            self.performance_stats['sequential_calls']
        )
        
        parallel_ratio = (
            self.performance_stats['parallel_calls'] / total_calls
            if total_calls > 0 else 0.0
        )
        
        return {
            'parallel_calls': self.performance_stats['parallel_calls'],
            'sequential_calls': self.performance_stats['sequential_calls'],
            'parallel_ratio': parallel_ratio,
            'average_speedup': self.performance_stats['avg_speedup'],
            'batch_threshold': self.batch_threshold,
            'parallel_config': {
                'mode': self.parallel_config.mode.value,
                'max_workers': self.parallel_config.max_workers,
                'memory_optimization': self.parallel_config.enable_memory_optimization
            }
        }
    
    def optimize_for_workload(
        self,
        typical_batch_sizes: List[int],
        optimization_iterations: int = 5
    ):
        """
        针对典型工作负载优化配置
        
        Args:
            typical_batch_sizes: 典型批处理大小列表
            optimization_iterations: 优化迭代次数
        """
        print("开始工作负载优化...")
        
        # 测试不同配置
        best_config = self.parallel_config
        best_performance = 0.0
        
        configs_to_test = [
            ParallelConfig(mode=ParallelMode.THREAD, max_workers=2),
            ParallelConfig(mode=ParallelMode.THREAD, max_workers=4),
            ParallelConfig(mode=ParallelMode.THREAD, max_workers=8),
            ParallelConfig(mode=ParallelMode.PROCESS, max_workers=2),
            ParallelConfig(mode=ParallelMode.PROCESS, max_workers=4),
        ]
        
        for config in configs_to_test:
            total_performance = 0.0
            
            # 临时设置配置
            original_config = self.parallel_config
            self.parallel_config = config
            self.parallel_manager = ParallelComputingManager(config)
            
            try:
                for batch_size in typical_batch_sizes:
                    # 生成测试数据
                    test_q = [np.random.randn(self.n_joints) for _ in range(batch_size)]
                    test_qd = [np.random.randn(self.n_joints) for _ in range(batch_size)]
                    test_tau = [np.random.randn(self.n_joints) for _ in range(batch_size)]
                    
                    # 测试性能
                    start_time = time.time()
                    for _ in range(optimization_iterations):
                        self.batch_forward_dynamics(test_q, test_qd, test_tau)
                    
                    avg_time = (time.time() - start_time) / optimization_iterations
                    performance_score = batch_size / avg_time if avg_time > 0 else 0
                    total_performance += performance_score
                
                if total_performance > best_performance:
                    best_performance = total_performance
                    best_config = config
                    
            except Exception as e:
                print(f"配置测试失败: {config.mode.value}, {e}")
            finally:
                # 恢复原始配置
                self.parallel_config = original_config
                self.parallel_manager = ParallelComputingManager(original_config)
        
        # 应用最佳配置
        self.parallel_config = best_config
        self.parallel_manager = ParallelComputingManager(best_config)
        
        print(f"优化完成，最佳配置: {best_config.mode.value}, "
              f"工作线程数: {best_config.max_workers}")