"""
并行轨迹规划模块

基于原始轨迹规划器实现并行优化版本，提高TOPP算法和S型插补的计算效率。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

from .trajectory_planning import TrajectoryPlanner, S7SegmentParameters
from ..core.parallel_computing import (
    ParallelComputingManager, ParallelConfig, ParallelMode,
    MemoryOptimizer, parallelize
)
from ..core.types import (
    Path, Trajectory, TrajectoryPoint, KinodynamicLimits, 
    Waypoint, PayloadInfo, Vector
)
from ..core.models import RobotModel


class ParallelOptimizedTrajectoryPlanner(TrajectoryPlanner):
    """
    并行优化的轨迹规划器
    
    继承原始轨迹规划器，添加并行计算优化功能。
    """
    
    def __init__(
        self,
        robot_model: RobotModel,
        parallel_config: Optional[ParallelConfig] = None
    ):
        """
        初始化并行优化轨迹规划器
        
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
        
        # 并行化阈值
        self.parallel_threshold = 8  # 路径点数量超过此值时使用并行计算
        
        # 线程本地存储
        self._thread_local = threading.local()
        
        # 性能统计
        self.performance_stats = {
            'parallel_topp_calls': 0,
            'parallel_s7_calls': 0,
            'total_speedup': 0.0,
            'avg_computation_time': 0.0
        }
    
    def parallel_generate_topp_trajectory(
        self,
        path: Path,
        limits: KinodynamicLimits,
        payload: Optional[PayloadInfo] = None,
        adaptive_envelope: bool = True,
        segment_size: int = 10
    ) -> Trajectory:
        """
        并行TOPP轨迹生成
        
        Args:
            path: 输入路径
            limits: 运动学动力学限制
            payload: 负载信息
            adaptive_envelope: 是否启用自适应包络线
            segment_size: 路径段大小
        
        Returns:
            时间最优轨迹
        """
        if len(path) < self.parallel_threshold:
            # 路径点少，使用原始方法
            return super().generate_topp_trajectory(path, limits, payload, adaptive_envelope)
        
        start_time = time.time()
        
        try:
            # 更新负载信息
            if payload:
                self.robot_model.update_payload(payload)
            
            # 将路径分段进行并行处理
            path_segments = self._split_path_into_segments(path, segment_size)
            
            # 并行计算每个段的TOPP轨迹
            segment_trajectories = self._parallel_topp_segments(
                path_segments, limits, adaptive_envelope
            )
            
            # 合并轨迹段
            merged_trajectory = self._merge_trajectory_segments(segment_trajectories)
            
            # 后处理
            final_trajectory = self._post_process_trajectory(merged_trajectory, limits)
            
            # 更新性能统计
            computation_time = time.time() - start_time
            self.performance_stats['parallel_topp_calls'] += 1
            self.performance_stats['avg_computation_time'] = (
                (self.performance_stats['avg_computation_time'] * 
                 (self.performance_stats['parallel_topp_calls'] - 1) + computation_time) /
                self.performance_stats['parallel_topp_calls']
            )
            
            return final_trajectory
            
        except Exception as e:
            print(f"并行TOPP失败，回退到顺序计算: {e}")
            return super().generate_topp_trajectory(path, limits, payload, adaptive_envelope)
    
    def parallel_interpolate_s7_trajectory(
        self,
        path: Path,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        max_jerk: Optional[float] = None,
        parallel_segments: bool = True
    ) -> Trajectory:
        """
        并行S7插补
        
        Args:
            path: 输入路径
            max_velocity: 最大速度限制
            max_acceleration: 最大加速度限制
            max_jerk: 最大加加速度限制
            parallel_segments: 是否并行处理路径段
        
        Returns:
            S型插补轨迹
        """
        if len(path) < 2:
            return super().interpolate_s7_trajectory(
                path, max_velocity, max_acceleration, max_jerk
            )
        
        if len(path) < self.parallel_threshold or not parallel_segments:
            return super().interpolate_s7_trajectory(
                path, max_velocity, max_acceleration, max_jerk
            )
        
        start_time = time.time()
        
        try:
            # 准备参数
            limits = self.robot_model.kinodynamic_limits
            
            if max_velocity is not None:
                v_max = min(max_velocity, min(limits.max_joint_velocities))
            else:
                v_max = min(limits.max_joint_velocities)
                
            if max_acceleration is not None:
                a_max = min(max_acceleration, min(limits.max_joint_accelerations))
            else:
                a_max = min(limits.max_joint_accelerations)
                
            if max_jerk is not None:
                j_max = min(max_jerk, min(limits.max_joint_jerks))
            else:
                j_max = min(limits.max_joint_jerks)
            
            # 并行处理路径段
            segment_trajectories = self._parallel_s7_segments(
                path, v_max, a_max, j_max
            )
            
            # 合并轨迹段并确保连续性
            merged_trajectory = self._merge_s7_segments(segment_trajectories)
            
            # 更新路径参数
            self._update_path_parameters(merged_trajectory)
            
            # 更新性能统计
            computation_time = time.time() - start_time
            self.performance_stats['parallel_s7_calls'] += 1
            
            return merged_trajectory
            
        except Exception as e:
            print(f"并行S7插补失败，回退到顺序计算: {e}")
            return super().interpolate_s7_trajectory(
                path, max_velocity, max_acceleration, max_jerk
            )
    
    def _split_path_into_segments(
        self,
        path: Path,
        segment_size: int
    ) -> List[Path]:
        """
        将路径分割为段
        
        Args:
            path: 输入路径
            segment_size: 段大小
        
        Returns:
            路径段列表
        """
        segments = []
        
        for i in range(0, len(path), segment_size - 1):  # 重叠一个点确保连续性
            end_idx = min(i + segment_size, len(path))
            segment = path[i:end_idx]
            
            if len(segment) >= 2:  # 确保段至少有两个点
                segments.append(segment)
        
        return segments
    
    def _parallel_topp_segments(
        self,
        path_segments: List[Path],
        limits: KinodynamicLimits,
        adaptive_envelope: bool
    ) -> List[Trajectory]:
        """
        并行处理TOPP路径段
        
        Args:
            path_segments: 路径段列表
            limits: 运动学动力学限制
            adaptive_envelope: 是否启用自适应包络线
        
        Returns:
            轨迹段列表
        """
        def process_single_segment(segment):
            # 为每个线程创建独立的规划器实例
            if not hasattr(self._thread_local, 'planner'):
                self._thread_local.planner = TrajectoryPlanner(self.robot_model)
            
            return self._thread_local.planner.generate_topp_trajectory(
                segment, limits, adaptive_envelope=adaptive_envelope
            )
        
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            segment_trajectories = list(executor.map(process_single_segment, path_segments))
        
        return segment_trajectories
    
    def _parallel_s7_segments(
        self,
        path: Path,
        v_max: float,
        a_max: float,
        j_max: float
    ) -> List[Trajectory]:
        """
        并行处理S7插补段
        
        Args:
            path: 输入路径
            v_max: 最大速度
            a_max: 最大加速度
            j_max: 最大加加速度
        
        Returns:
            轨迹段列表
        """
        # 创建路径段对
        segment_pairs = []
        for i in range(len(path) - 1):
            segment_pairs.append((path[i], path[i + 1]))
        
        def process_single_s7_segment(segment_pair):
            start_point, end_point = segment_pair
            
            # 为每个线程创建独立的规划器实例
            if not hasattr(self._thread_local, 'planner'):
                self._thread_local.planner = TrajectoryPlanner(self.robot_model)
            
            return self._thread_local.planner._interpolate_s7_segment(
                start_point, end_point, v_max, a_max, j_max
            )
        
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            segment_trajectories = list(executor.map(process_single_s7_segment, segment_pairs))
        
        return segment_trajectories
    
    def _merge_trajectory_segments(
        self,
        segment_trajectories: List[Trajectory]
    ) -> Trajectory:
        """
        合并轨迹段
        
        Args:
            segment_trajectories: 轨迹段列表
        
        Returns:
            合并后的轨迹
        """
        if not segment_trajectories:
            return []
        
        merged_trajectory = []
        current_time_offset = 0.0
        
        for i, segment in enumerate(segment_trajectories):
            if not segment:
                continue
            
            # 调整时间偏移
            adjusted_segment = []
            for point in segment:
                adjusted_point = TrajectoryPoint(
                    position=point.position.copy(),
                    velocity=point.velocity.copy(),
                    acceleration=point.acceleration.copy(),
                    jerk=point.jerk.copy(),
                    time=point.time + current_time_offset,
                    path_parameter=point.path_parameter
                )
                adjusted_segment.append(adjusted_point)
            
            # 添加到合并轨迹
            if i == 0:
                merged_trajectory.extend(adjusted_segment)
            else:
                # 跳过第一个点以避免重复，但确保连续性
                if len(merged_trajectory) > 0 and len(adjusted_segment) > 0:
                    # 检查连续性
                    last_point = merged_trajectory[-1]
                    first_point = adjusted_segment[0]
                    
                    # 如果位置不连续，进行平滑过渡
                    pos_diff = np.linalg.norm(
                        first_point.position - last_point.position
                    )
                    
                    if pos_diff > 1e-3:  # 位置不连续
                        # 插入过渡点
                        transition_point = TrajectoryPoint(
                            position=(last_point.position + first_point.position) / 2,
                            velocity=(last_point.velocity + first_point.velocity) / 2,
                            acceleration=np.zeros(self.n_joints),
                            jerk=np.zeros(self.n_joints),
                            time=(last_point.time + first_point.time) / 2,
                            path_parameter=(last_point.path_parameter + first_point.path_parameter) / 2
                        )
                        merged_trajectory.append(transition_point)
                
                merged_trajectory.extend(adjusted_segment[1:])  # 跳过第一个点
            
            # 更新时间偏移
            if adjusted_segment:
                current_time_offset = adjusted_segment[-1].time
        
        return merged_trajectory
    
    def _merge_s7_segments(
        self,
        segment_trajectories: List[Trajectory]
    ) -> Trajectory:
        """
        合并S7插补段，确保连续性
        
        Args:
            segment_trajectories: 轨迹段列表
        
        Returns:
            合并后的轨迹
        """
        if not segment_trajectories:
            return []
        
        merged_trajectory = []
        current_time = 0.0
        
        for i, segment in enumerate(segment_trajectories):
            if not segment:
                continue
            
            # 调整时间戳
            adjusted_segment = []
            for j, point in enumerate(segment):
                adjusted_point = TrajectoryPoint(
                    position=point.position.copy(),
                    velocity=point.velocity.copy(),
                    acceleration=point.acceleration.copy(),
                    jerk=point.jerk.copy(),
                    time=current_time + point.time,
                    path_parameter=point.path_parameter
                )
                adjusted_segment.append(adjusted_point)
            
            # 添加到合并轨迹
            if i == 0:
                merged_trajectory.extend(adjusted_segment)
            else:
                # 确保连续性：调整第一个点以匹配上一段的最后一个点
                if len(merged_trajectory) > 0 and len(adjusted_segment) > 0:
                    last_point = merged_trajectory[-1]
                    
                    # 调整第一个点的位置和速度以确保连续性
                    adjusted_segment[0].position = last_point.position.copy()
                    adjusted_segment[0].velocity = last_point.velocity.copy()
                
                merged_trajectory.extend(adjusted_segment[1:])  # 跳过第一个点
            
            # 更新当前时间
            if adjusted_segment:
                current_time = adjusted_segment[-1].time
        
        return merged_trajectory
    
    @parallelize(mode=ParallelMode.THREAD, max_workers=4)
    def parallel_velocity_limits_computation(
        self,
        parameterized_path_points: List[Tuple[float, Vector, Vector]],
        limits: KinodynamicLimits
    ) -> List[float]:
        """
        并行计算速度限制
        
        Args:
            parameterized_path_points: 参数化路径点列表
            limits: 运动学动力学限制
        
        Returns:
            速度限制列表
        """
        def compute_single_velocity_limit(path_point):
            s, position, tangent = path_point
            
            # 运动学速度限制
            kinematic_limit = self._compute_kinematic_velocity_limit(tangent, limits)
            
            # 动力学速度限制
            dynamic_limit = self._compute_dynamic_velocity_limit(position, tangent, limits)
            
            # 曲率限制（简化计算）
            curvature_limit = float('inf')  # 简化为无限制
            
            return min(kinematic_limit, dynamic_limit, curvature_limit)
        
        return [compute_single_velocity_limit(point) for point in parameterized_path_points]
    
    def parallel_trajectory_optimization(
        self,
        initial_trajectory: Trajectory,
        optimization_params: Dict[str, Any]
    ) -> Trajectory:
        """
        并行轨迹优化
        
        Args:
            initial_trajectory: 初始轨迹
            optimization_params: 优化参数
        
        Returns:
            优化后的轨迹
        """
        if len(initial_trajectory) < self.parallel_threshold:
            return initial_trajectory
        
        # 将轨迹分段进行并行优化
        segment_size = optimization_params.get('segment_size', 20)
        segments = []
        
        for i in range(0, len(initial_trajectory), segment_size - 1):
            end_idx = min(i + segment_size, len(initial_trajectory))
            segment = initial_trajectory[i:end_idx]
            segments.append(segment)
        
        def optimize_single_segment(segment):
            # 简化的轨迹优化：平滑处理
            if len(segment) < 3:
                return segment
            
            optimized_segment = []
            
            for i, point in enumerate(segment):
                if i == 0 or i == len(segment) - 1:
                    # 保持端点不变
                    optimized_segment.append(point)
                else:
                    # 平滑中间点
                    prev_point = segment[i - 1]
                    next_point = segment[i + 1]
                    
                    smoothed_velocity = (
                        prev_point.velocity + point.velocity + next_point.velocity
                    ) / 3.0
                    
                    smoothed_acceleration = (
                        prev_point.acceleration + point.acceleration + next_point.acceleration
                    ) / 3.0
                    
                    optimized_point = TrajectoryPoint(
                        position=point.position,
                        velocity=smoothed_velocity,
                        acceleration=smoothed_acceleration,
                        jerk=point.jerk,
                        time=point.time,
                        path_parameter=point.path_parameter
                    )
                    
                    optimized_segment.append(optimized_point)
            
            return optimized_segment
        
        # 并行优化各段
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            optimized_segments = list(executor.map(optimize_single_segment, segments))
        
        # 合并优化后的段
        return self._merge_trajectory_segments(optimized_segments)
    
    def batch_trajectory_generation(
        self,
        path_list: List[Path],
        limits_list: List[KinodynamicLimits],
        method: str = "topp"
    ) -> List[Trajectory]:
        """
        批量轨迹生成
        
        Args:
            path_list: 路径列表
            limits_list: 限制条件列表
            method: 生成方法 ("topp" 或 "s7")
        
        Returns:
            轨迹列表
        """
        if len(path_list) != len(limits_list):
            raise ValueError("路径列表和限制列表长度不匹配")
        
        def generate_single_trajectory(args):
            path, limits = args
            
            if method == "topp":
                return self.generate_topp_trajectory(path, limits)
            elif method == "s7":
                return self.interpolate_s7_trajectory(path)
            else:
                raise ValueError(f"未知的生成方法: {method}")
        
        trajectory_args = list(zip(path_list, limits_list))
        
        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            trajectories = list(executor.map(generate_single_trajectory, trajectory_args))
        
        return trajectories
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            性能报告字典
        """
        return {
            'parallel_topp_calls': self.performance_stats['parallel_topp_calls'],
            'parallel_s7_calls': self.performance_stats['parallel_s7_calls'],
            'average_computation_time': self.performance_stats['avg_computation_time'],
            'parallel_threshold': self.parallel_threshold,
            'parallel_config': {
                'mode': self.parallel_config.mode.value,
                'max_workers': self.parallel_config.max_workers,
                'memory_optimization': self.parallel_config.enable_memory_optimization
            }
        }
    
    def auto_tune_parallel_parameters(
        self,
        sample_paths: List[Path],
        sample_limits: List[KinodynamicLimits]
    ):
        """
        自动调优并行参数
        
        Args:
            sample_paths: 样本路径列表
            sample_limits: 样本限制列表
        """
        print("开始并行参数自动调优...")
        
        best_threshold = self.parallel_threshold
        best_workers = self.parallel_config.max_workers
        best_performance = 0.0
        
        # 测试不同的并行阈值
        for threshold in [4, 8, 16, 32]:
            # 测试不同的工作线程数
            for workers in [2, 4, 8]:
                if workers > self.parallel_config.max_workers:
                    continue
                
                # 临时设置参数
                original_threshold = self.parallel_threshold
                original_config = self.parallel_config
                
                self.parallel_threshold = threshold
                self.parallel_config = ParallelConfig(
                    mode=self.parallel_config.mode,
                    max_workers=workers,
                    enable_memory_optimization=True
                )
                self.parallel_manager = ParallelComputingManager(self.parallel_config)
                
                try:
                    # 测试性能
                    total_time = 0.0
                    test_count = 0
                    
                    for path, limits in zip(sample_paths[:3], sample_limits[:3]):  # 限制测试数量
                        if len(path) >= threshold:
                            start_time = time.time()
                            self.parallel_generate_topp_trajectory(path, limits)
                            total_time += time.time() - start_time
                            test_count += 1
                    
                    if test_count > 0:
                        avg_time = total_time / test_count
                        performance_score = 1.0 / avg_time if avg_time > 0 else 0
                        
                        if performance_score > best_performance:
                            best_performance = performance_score
                            best_threshold = threshold
                            best_workers = workers
                
                except Exception as e:
                    print(f"参数测试失败: threshold={threshold}, workers={workers}, {e}")
                
                finally:
                    # 恢复原始参数
                    self.parallel_threshold = original_threshold
                    self.parallel_config = original_config
                    self.parallel_manager = ParallelComputingManager(original_config)
        
        # 应用最佳参数
        self.parallel_threshold = best_threshold
        self.parallel_config = ParallelConfig(
            mode=self.parallel_config.mode,
            max_workers=best_workers,
            enable_memory_optimization=True
        )
        self.parallel_manager = ParallelComputingManager(self.parallel_config)
        
        print(f"调优完成，最佳参数: threshold={best_threshold}, workers={best_workers}")