"""
并行计算优化模块

实现多线程并行计算优化，提高算法计算效率。
支持线程池管理、任务调度和数据结构优化。

主要功能：
- 线程池管理和任务调度
- 并行动力学计算
- 并行轨迹规划
- 并行控制计算
- 内存访问优化
- 性能监控和分析
"""

import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum
import queue
import functools

from .types import Vector, Matrix, RobotState, TrajectoryPoint, ControlCommand


class ParallelMode(Enum):
    """并行计算模式"""
    THREAD = "thread"      # 线程并行（适合I/O密集型）
    PROCESS = "process"    # 进程并行（适合CPU密集型）
    HYBRID = "hybrid"      # 混合模式


@dataclass
class ParallelConfig:
    """并行计算配置"""
    mode: ParallelMode = ParallelMode.THREAD
    max_workers: Optional[int] = None  # None表示自动检测
    chunk_size: int = 1               # 任务分块大小
    enable_memory_optimization: bool = True
    enable_performance_monitoring: bool = True
    thread_affinity: Optional[List[int]] = None  # CPU亲和性


@dataclass
class PerformanceMetrics:
    """性能监控指标"""
    total_execution_time: float = 0.0
    parallel_execution_time: float = 0.0
    sequential_execution_time: float = 0.0
    speedup_ratio: float = 1.0
    efficiency: float = 1.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    task_count: int = 0
    failed_tasks: int = 0


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, config: ParallelConfig):
        """
        初始化任务调度器
        
        Args:
            config: 并行计算配置
        """
        self.config = config
        self.performance_metrics = PerformanceMetrics()
        
        # 确定工作线程数
        if config.max_workers is None:
            if config.mode == ParallelMode.THREAD:
                self.max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
            else:
                self.max_workers = multiprocessing.cpu_count() or 1
        else:
            self.max_workers = config.max_workers
        
        # 线程池
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # 性能监控
        self._lock = threading.Lock()
        self._task_times = []
        
    def __enter__(self):
        """上下文管理器入口"""
        self._initialize_pools()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._cleanup_pools()
    
    def _initialize_pools(self):
        """初始化线程池和进程池"""
        if self.config.mode in [ParallelMode.THREAD, ParallelMode.HYBRID]:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="RobotControl"
            )
        
        if self.config.mode in [ParallelMode.PROCESS, ParallelMode.HYBRID]:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.max_workers
            )
    
    def _cleanup_pools(self):
        """清理线程池和进程池"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
    
    def submit_tasks(
        self,
        func: Callable,
        tasks: List[Any],
        use_processes: bool = False,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        提交并执行任务列表
        
        Args:
            func: 要执行的函数
            tasks: 任务参数列表
            use_processes: 是否使用进程池
            timeout: 超时时间（秒）
        
        Returns:
            任务结果列表
        """
        if not tasks:
            return []
        
        start_time = time.time()
        
        try:
            # 选择执行器
            if use_processes and self._process_pool:
                executor = self._process_pool
            elif self._thread_pool:
                executor = self._thread_pool
            else:
                # 回退到顺序执行
                return [func(task) for task in tasks]
            
            # 提交任务并保持顺序
            future_to_index = {}
            for i, task in enumerate(tasks):
                future = executor.submit(func, task)
                future_to_index[future] = i
            
            # 收集结果并保持原始顺序
            results = [None] * len(tasks)
            completed_count = 0
            failed_count = 0
            
            for future in as_completed(future_to_index.keys(), timeout=timeout):
                try:
                    result = future.result()
                    index = future_to_index[future]
                    results[index] = result
                    completed_count += 1
                except Exception as e:
                    index = future_to_index[future]
                    results[index] = None
                    failed_count += 1
                    print(f"任务{index}执行失败: {e}")
            
            # 更新性能指标
            execution_time = time.time() - start_time
            self._update_performance_metrics(
                execution_time, len(tasks), failed_count
            )
            
            return results
            
        except Exception as e:
            print(f"并行任务执行失败: {e}")
            # 回退到顺序执行
            return [func(task) for task in tasks]
    
    def map_parallel(
        self,
        func: Callable,
        iterable: List[Any],
        chunk_size: Optional[int] = None,
        use_processes: bool = False
    ) -> List[Any]:
        """
        并行映射函数
        
        Args:
            func: 映射函数
            iterable: 输入数据
            chunk_size: 分块大小
            use_processes: 是否使用进程池
        
        Returns:
            映射结果列表
        """
        if not iterable:
            return []
        
        chunk_size = chunk_size or self.config.chunk_size
        
        # 选择执行器
        if use_processes and self._process_pool:
            executor = self._process_pool
        elif self._thread_pool:
            executor = self._thread_pool
        else:
            return list(map(func, iterable))
        
        try:
            # 使用executor.map进行并行映射
            results = list(executor.map(func, iterable, chunksize=chunk_size))
            return results
        except Exception as e:
            print(f"并行映射失败: {e}")
            return list(map(func, iterable))
    
    def _update_performance_metrics(
        self,
        execution_time: float,
        task_count: int,
        failed_count: int
    ):
        """更新性能指标"""
        with self._lock:
            self.performance_metrics.total_execution_time += execution_time
            self.performance_metrics.task_count += task_count
            self.performance_metrics.failed_tasks += failed_count
            
            # 估算顺序执行时间（简化）
            avg_task_time = execution_time / max(task_count, 1)
            sequential_time = avg_task_time * task_count
            
            # 计算加速比
            if sequential_time > 0:
                speedup = sequential_time / execution_time
                self.performance_metrics.speedup_ratio = speedup
                self.performance_metrics.efficiency = speedup / self.max_workers
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        return self.performance_metrics


class ParallelDynamicsEngine:
    """并行动力学计算引擎"""
    
    def __init__(self, dynamics_engine, scheduler: TaskScheduler):
        """
        初始化并行动力学引擎
        
        Args:
            dynamics_engine: 原始动力学引擎
            scheduler: 任务调度器
        """
        self.dynamics_engine = dynamics_engine
        self.scheduler = scheduler
        self.n_joints = dynamics_engine.n_joints
    
    def parallel_forward_dynamics_batch(
        self,
        states: List[Tuple[Vector, Vector, Vector]]
    ) -> List[Vector]:
        """
        批量并行正向动力学计算
        
        Args:
            states: 状态列表 [(q, qd, tau), ...]
        
        Returns:
            加速度列表
        """
        def compute_single_forward_dynamics(state):
            q, qd, tau = state
            return self.dynamics_engine.forward_dynamics(q, qd, tau)
        
        return self.scheduler.submit_tasks(
            compute_single_forward_dynamics,
            states,
            use_processes=True  # CPU密集型，使用进程
        )
    
    def parallel_inverse_dynamics_batch(
        self,
        states: List[Tuple[Vector, Vector, Vector]]
    ) -> List[Vector]:
        """
        批量并行逆向动力学计算
        
        Args:
            states: 状态列表 [(q, qd, qdd), ...]
        
        Returns:
            力矩列表
        """
        def compute_single_inverse_dynamics(state):
            q, qd, qdd = state
            return self.dynamics_engine.inverse_dynamics(q, qd, qdd)
        
        return self.scheduler.submit_tasks(
            compute_single_inverse_dynamics,
            states,
            use_processes=True
        )
    
    def parallel_jacobian_batch(self, positions: List[Vector]) -> List[Matrix]:
        """
        批量并行雅可比矩阵计算
        
        Args:
            positions: 关节位置列表
        
        Returns:
            雅可比矩阵列表
        """
        def compute_single_jacobian(q):
            return self.dynamics_engine.jacobian(q)
        
        return self.scheduler.submit_tasks(
            compute_single_jacobian,
            positions,
            use_processes=True
        )
    
    def parallel_gravity_compensation_batch(
        self,
        positions: List[Vector]
    ) -> List[Vector]:
        """
        批量并行重力补偿计算
        
        Args:
            positions: 关节位置列表
        
        Returns:
            重力补偿力矩列表
        """
        def compute_single_gravity_compensation(q):
            return self.dynamics_engine.gravity_compensation(q)
        
        return self.scheduler.submit_tasks(
            compute_single_gravity_compensation,
            positions,
            use_processes=True
        )


class ParallelTrajectoryPlanner:
    """并行轨迹规划器"""
    
    def __init__(self, trajectory_planner, scheduler: TaskScheduler):
        """
        初始化并行轨迹规划器
        
        Args:
            trajectory_planner: 原始轨迹规划器
            scheduler: 任务调度器
        """
        self.trajectory_planner = trajectory_planner
        self.scheduler = scheduler
    
    def parallel_topp_segments(
        self,
        path_segments: List[Any],
        limits_list: List[Any]
    ) -> List[Any]:
        """
        并行TOPP轨迹规划
        
        Args:
            path_segments: 路径段列表
            limits_list: 限制条件列表
        
        Returns:
            轨迹段列表
        """
        def compute_single_topp_segment(args):
            path_segment, limits = args
            return self.trajectory_planner.generate_topp_trajectory(
                path_segment, limits
            )
        
        segment_args = list(zip(path_segments, limits_list))
        
        return self.scheduler.submit_tasks(
            compute_single_topp_segment,
            segment_args,
            use_processes=True
        )
    
    def parallel_s7_interpolation(
        self,
        waypoint_pairs: List[Tuple[Any, Any]],
        parameters: List[Dict[str, float]]
    ) -> List[Any]:
        """
        并行S7插补
        
        Args:
            waypoint_pairs: 路径点对列表
            parameters: 插补参数列表
        
        Returns:
            插补轨迹列表
        """
        def compute_single_s7_interpolation(args):
            (start_point, end_point), params = args
            return self.trajectory_planner._interpolate_s7_segment(
                start_point, end_point,
                params.get('v_max', 1.0),
                params.get('a_max', 1.0),
                params.get('j_max', 1.0)
            )
        
        interpolation_args = list(zip(waypoint_pairs, parameters))
        
        return self.scheduler.submit_tasks(
            compute_single_s7_interpolation,
            interpolation_args,
            use_processes=True
        )
    
    def parallel_velocity_limits_computation(
        self,
        parameterized_path_segments: List[Any],
        limits: Any
    ) -> List[List[float]]:
        """
        并行计算速度限制
        
        Args:
            parameterized_path_segments: 参数化路径段列表
            limits: 运动学动力学限制
        
        Returns:
            速度限制列表
        """
        def compute_segment_velocity_limits(path_segment):
            return self.trajectory_planner._compute_velocity_limits(
                path_segment, limits
            )
        
        return self.scheduler.submit_tasks(
            compute_segment_velocity_limits,
            parameterized_path_segments,
            use_processes=True
        )


class ParallelPathController:
    """并行路径控制器"""
    
    def __init__(self, path_controller, scheduler: TaskScheduler):
        """
        初始化并行路径控制器
        
        Args:
            path_controller: 原始路径控制器
            scheduler: 任务调度器
        """
        self.path_controller = path_controller
        self.scheduler = scheduler
    
    def parallel_control_computation(
        self,
        reference_trajectory: List[TrajectoryPoint],
        current_states: List[RobotState]
    ) -> List[ControlCommand]:
        """
        并行控制计算
        
        Args:
            reference_trajectory: 参考轨迹列表
            current_states: 当前状态列表
        
        Returns:
            控制指令列表
        """
        def compute_single_control(args):
            reference, current_state = args
            return self.path_controller.compute_control(reference, current_state)
        
        control_args = list(zip(reference_trajectory, current_states))
        
        return self.scheduler.submit_tasks(
            compute_single_control,
            control_args,
            use_processes=False  # 使用线程，因为涉及共享状态
        )
    
    def parallel_feedforward_computation(
        self,
        reference_points: List[TrajectoryPoint]
    ) -> List[Vector]:
        """
        并行前馈控制计算
        
        Args:
            reference_points: 参考点列表
        
        Returns:
            前馈控制输出列表
        """
        def compute_single_feedforward(reference):
            return self.path_controller.feedforward_control(reference)
        
        return self.scheduler.submit_tasks(
            compute_single_feedforward,
            reference_points,
            use_processes=True
        )


class MemoryOptimizer:
    """内存访问优化器"""
    
    def __init__(self):
        """初始化内存优化器"""
        self._memory_pools = {}
        self._lock = threading.Lock()
    
    def get_memory_pool(self, key: str, shape: Tuple[int, ...], dtype=np.float64):
        """
        获取内存池
        
        Args:
            key: 内存池键
            shape: 数组形状
            dtype: 数据类型
        
        Returns:
            预分配的内存数组
        """
        with self._lock:
            if key not in self._memory_pools:
                # 预分配内存池
                pool_size = 100  # 预分配100个数组
                self._memory_pools[key] = [
                    np.zeros(shape, dtype=dtype) for _ in range(pool_size)
                ]
            
            if self._memory_pools[key]:
                return self._memory_pools[key].pop()
            else:
                # 池为空，创建新数组
                return np.zeros(shape, dtype=dtype)
    
    def return_to_pool(self, key: str, array: np.ndarray):
        """
        将数组返回到内存池
        
        Args:
            key: 内存池键
            array: 要返回的数组
        """
        with self._lock:
            if key in self._memory_pools:
                # 清零数组并返回池中
                array.fill(0)
                self._memory_pools[key].append(array)
    
    def optimize_array_layout(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        优化数组内存布局
        
        Args:
            arrays: 输入数组列表
        
        Returns:
            优化后的数组列表
        """
        optimized = []
        
        for array in arrays:
            # 确保数组是C连续的以提高缓存效率
            if not array.flags['C_CONTIGUOUS']:
                optimized_array = np.ascontiguousarray(array)
            else:
                optimized_array = array
            
            optimized.append(optimized_array)
        
        return optimized
    
    def batch_operations(
        self,
        operation: Callable,
        arrays: List[np.ndarray],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        批量操作优化
        
        Args:
            operation: 要执行的操作
            arrays: 输入数组列表
            batch_size: 批处理大小
        
        Returns:
            操作结果列表
        """
        results = []
        
        for i in range(0, len(arrays), batch_size):
            batch = arrays[i:i + batch_size]
            
            # 将批次数据组织为连续内存块
            if batch:
                batch_array = np.stack(batch)
                batch_results = operation(batch_array)
                
                # 分解批次结果
                if isinstance(batch_results, np.ndarray):
                    results.extend([batch_results[j] for j in range(len(batch))])
                else:
                    results.extend(batch_results)
        
        return results


class ParallelComputingManager:
    """并行计算管理器"""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初始化并行计算管理器
        
        Args:
            config: 并行计算配置
        """
        self.config = config or ParallelConfig()
        self.scheduler = TaskScheduler(self.config)
        self.memory_optimizer = MemoryOptimizer()
        
        # 性能监控
        self.performance_history = []
        self._monitoring_enabled = self.config.enable_performance_monitoring
    
    def __enter__(self):
        """上下文管理器入口"""
        self.scheduler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.scheduler.__exit__(exc_type, exc_val, exc_tb)
    
    def create_parallel_dynamics_engine(self, dynamics_engine):
        """创建并行动力学引擎"""
        return ParallelDynamicsEngine(dynamics_engine, self.scheduler)
    
    def create_parallel_trajectory_planner(self, trajectory_planner):
        """创建并行轨迹规划器"""
        return ParallelTrajectoryPlanner(trajectory_planner, self.scheduler)
    
    def create_parallel_path_controller(self, path_controller):
        """创建并行路径控制器"""
        return ParallelPathController(path_controller, self.scheduler)
    
    def benchmark_parallel_performance(
        self,
        func: Callable,
        test_data: List[Any],
        iterations: int = 10
    ) -> Dict[str, float]:
        """
        基准测试并行性能
        
        Args:
            func: 测试函数
            test_data: 测试数据
            iterations: 迭代次数
        
        Returns:
            性能基准结果
        """
        # 顺序执行基准
        sequential_times = []
        for _ in range(iterations):
            start_time = time.time()
            sequential_results = [func(data) for data in test_data]
            sequential_times.append(time.time() - start_time)
        
        # 并行执行基准
        parallel_times = []
        for _ in range(iterations):
            start_time = time.time()
            parallel_results = self.scheduler.submit_tasks(func, test_data)
            parallel_times.append(time.time() - start_time)
        
        # 计算统计数据
        avg_sequential = np.mean(sequential_times)
        avg_parallel = np.mean(parallel_times)
        speedup = avg_sequential / avg_parallel if avg_parallel > 0 else 1.0
        efficiency = speedup / self.scheduler.max_workers
        
        return {
            'sequential_time': avg_sequential,
            'parallel_time': avg_parallel,
            'speedup': speedup,
            'efficiency': efficiency,
            'max_workers': self.scheduler.max_workers
        }
    
    def optimize_task_distribution(
        self,
        task_complexities: List[float],
        target_load_balance: float = 0.9
    ) -> List[List[int]]:
        """
        优化任务分配
        
        Args:
            task_complexities: 任务复杂度列表
            target_load_balance: 目标负载均衡度
        
        Returns:
            任务分配方案（每个工作线程的任务索引列表）
        """
        n_workers = self.scheduler.max_workers
        n_tasks = len(task_complexities)
        
        if n_tasks <= n_workers:
            # 任务数少于工作线程数，每个任务分配一个线程
            return [[i] for i in range(n_tasks)]
        
        # 使用贪心算法进行负载均衡
        worker_loads = [0.0] * n_workers
        worker_tasks = [[] for _ in range(n_workers)]
        
        # 按复杂度降序排序任务
        sorted_tasks = sorted(
            enumerate(task_complexities),
            key=lambda x: x[1],
            reverse=True
        )
        
        for task_idx, complexity in sorted_tasks:
            # 找到负载最轻的工作线程
            min_load_worker = min(range(n_workers), key=lambda i: worker_loads[i])
            
            # 分配任务
            worker_loads[min_load_worker] += complexity
            worker_tasks[min_load_worker].append(task_idx)
        
        return worker_tasks
    
    def get_system_performance_info(self) -> Dict[str, Any]:
        """
        获取系统性能信息
        
        Returns:
            系统性能信息字典
        """
        import psutil
        
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_info': {
                'total': psutil.virtual_memory().total / (1024**3),  # GB
                'available': psutil.virtual_memory().available / (1024**3),  # GB
                'percent': psutil.virtual_memory().percent
            },
            'thread_count': threading.active_count(),
            'scheduler_metrics': self.scheduler.get_performance_metrics()
        }


# 装饰器：自动并行化
def parallelize(
    mode: ParallelMode = ParallelMode.THREAD,
    max_workers: Optional[int] = None,
    chunk_size: int = 1
):
    """
    自动并行化装饰器
    
    Args:
        mode: 并行模式
        max_workers: 最大工作线程数
        chunk_size: 分块大小
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否有可并行化的参数
            parallel_args = []
            for arg in args:
                if isinstance(arg, (list, tuple)) and len(arg) > 1:
                    parallel_args.append(arg)
            
            if not parallel_args:
                # 没有可并行化的参数，直接执行
                return func(*args, **kwargs)
            
            # 创建临时调度器
            config = ParallelConfig(
                mode=mode,
                max_workers=max_workers,
                chunk_size=chunk_size
            )
            
            with TaskScheduler(config) as scheduler:
                # 简化的并行执行（仅处理第一个列表参数）
                first_list_arg = parallel_args[0]
                
                def single_call(item):
                    # 替换第一个列表参数为单个项目
                    new_args = list(args)
                    for i, arg in enumerate(new_args):
                        if arg is first_list_arg:
                            new_args[i] = item
                            break
                    return func(*new_args, **kwargs)
                
                return scheduler.submit_tasks(single_call, first_list_arg)
        
        return wrapper
    return decorator


# 工具函数
def estimate_task_complexity(
    func: Callable,
    sample_data: Any,
    iterations: int = 5
) -> float:
    """
    估算任务复杂度
    
    Args:
        func: 要测试的函数
        sample_data: 样本数据
        iterations: 测试迭代次数
    
    Returns:
        平均执行时间（秒）
    """
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        try:
            func(sample_data)
            times.append(time.time() - start_time)
        except Exception:
            # 如果执行失败，返回默认复杂度
            return 0.001
    
    return np.mean(times) if times else 0.001


def auto_tune_parallel_config(
    test_functions: List[Callable],
    test_data_sets: List[List[Any]]
) -> ParallelConfig:
    """
    自动调优并行配置
    
    Args:
        test_functions: 测试函数列表
        test_data_sets: 测试数据集列表
    
    Returns:
        优化的并行配置
    """
    best_config = ParallelConfig()
    best_performance = 0.0
    
    # 测试不同的配置
    modes = [ParallelMode.THREAD, ParallelMode.PROCESS]
    worker_counts = [2, 4, 8, multiprocessing.cpu_count()]
    
    for mode in modes:
        for workers in worker_counts:
            if workers > multiprocessing.cpu_count():
                continue
            
            config = ParallelConfig(mode=mode, max_workers=workers)
            
            # 测试性能
            total_speedup = 0.0
            test_count = 0
            
            with ParallelComputingManager(config) as manager:
                for func, test_data in zip(test_functions, test_data_sets):
                    try:
                        benchmark = manager.benchmark_parallel_performance(
                            func, test_data, iterations=3
                        )
                        total_speedup += benchmark['speedup']
                        test_count += 1
                    except Exception:
                        continue
            
            if test_count > 0:
                avg_speedup = total_speedup / test_count
                if avg_speedup > best_performance:
                    best_performance = avg_speedup
                    best_config = config
    
    return best_config