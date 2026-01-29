"""
主控制器模块

实现机器人运动控制系统的主控制器，集成轨迹规划、路径控制、抑振算法等功能。
提供统一的控制接口和算法协调管理。
"""

from typing import List, Optional, Dict, Any
import numpy as np
import time
from dataclasses import dataclass

from .models import RobotModel, RobotStateHistory
from .types import (
    RobotState, TrajectoryPoint, ControlCommand, Trajectory,
    Path, Waypoint, PerformanceMetrics, PayloadInfo
)
from .parallel_computing import ParallelComputingManager, ParallelConfig, ParallelMode


@dataclass
class ControllerConfig:
    """控制器配置参数"""
    control_frequency: float = 1000.0  # 控制频率 [Hz]
    enable_feedforward: bool = True    # 是否启用前馈控制
    enable_vibration_suppression: bool = True  # 是否启用抑振
    enable_payload_adaptation: bool = True     # 是否启用负载自适应
    safety_check_enabled: bool = True          # 是否启用安全检查
    max_tracking_error: float = 0.001          # 最大跟踪误差 [m]
    max_vibration_amplitude: float = 0.00005   # 最大振动幅度 [m]
    
    # 并行计算配置
    enable_parallel_computing: bool = True     # 是否启用并行计算
    parallel_mode: ParallelMode = ParallelMode.THREAD  # 并行模式
    max_parallel_workers: Optional[int] = None # 最大并行工作线程数
    parallel_batch_threshold: int = 4          # 并行计算阈值


class RobotMotionController:
    """
    机器人运动控制器主类
    
    集成轨迹规划、路径控制、抑振算法等功能，提供统一的控制接口。
    实现高精度、高速度、低振动的机器人运动控制。
    """
    
    def __init__(
        self,
        robot_model: RobotModel,
        config: Optional[ControllerConfig] = None
    ):
        """
        初始化控制器
        
        Args:
            robot_model: 机器人模型
            config: 控制器配置（可选）
        """
        self.robot_model = robot_model
        self.config = config or ControllerConfig()
        
        # 状态管理
        self.state_history = RobotStateHistory()
        self.current_trajectory: Optional[Trajectory] = None
        self.trajectory_index = 0
        
        # 性能监控
        self.performance_metrics = PerformanceMetrics(
            computation_time=0.0,
            memory_usage=0.0,
            tracking_error=0.0,
            vibration_amplitude=0.0,
            success_rate=1.0
        )
        
        # 算法组件（延迟初始化）
        self._trajectory_planner = None
        self._path_controller = None
        self._vibration_suppressor = None
        self._dynamics_engine = None
        self._payload_identifier = None
        self._safety_monitor = None
        self._collision_monitor = None
        
        # 控制状态
        self.is_active = False
        self.emergency_stop = False
        
        # 并行计算管理器
        self.parallel_manager = None
        if self.config.enable_parallel_computing:
            parallel_config = ParallelConfig(
                mode=self.config.parallel_mode,
                max_workers=self.config.max_parallel_workers,
                enable_memory_optimization=True,
                enable_performance_monitoring=True
            )
            self.parallel_manager = ParallelComputingManager(parallel_config)
        
        # 并行优化的算法组件
        self._parallel_dynamics_engine = None
        self._parallel_trajectory_planner = None
        self._parallel_path_controller = None
        
    @property
    def trajectory_planner(self):
        """延迟初始化轨迹规划器"""
        if self._trajectory_planner is None:
            if self.config.enable_parallel_computing and self.parallel_manager:
                from ..algorithms.parallel_trajectory_planning import ParallelOptimizedTrajectoryPlanner
                self._trajectory_planner = ParallelOptimizedTrajectoryPlanner(
                    self.robot_model, self.parallel_manager.config
                )
            else:
                from ..algorithms.trajectory_planning import TrajectoryPlanner
                self._trajectory_planner = TrajectoryPlanner(self.robot_model)
        return self._trajectory_planner
    
    @property
    def path_controller(self):
        """延迟初始化路径控制器"""
        if self._path_controller is None:
            if self.config.enable_parallel_computing and self.parallel_manager:
                from ..algorithms.parallel_path_control import ParallelOptimizedPathController
                self._path_controller = ParallelOptimizedPathController(
                    self.robot_model, parallel_config=self.parallel_manager.config
                )
            else:
                from ..algorithms.path_control import PathController
                self._path_controller = PathController(self.robot_model)
        return self._path_controller
    
    @property
    def vibration_suppressor(self):
        """延迟初始化抑振控制器"""
        if self._vibration_suppressor is None:
            from ..algorithms.vibration_suppression import VibrationSuppressor
            self._vibration_suppressor = VibrationSuppressor(self.robot_model)
        return self._vibration_suppressor
    
    @property
    def dynamics_engine(self):
        """延迟初始化动力学引擎"""
        if self._dynamics_engine is None:
            if self.config.enable_parallel_computing and self.parallel_manager:
                from ..algorithms.parallel_dynamics import ParallelOptimizedDynamicsEngine
                self._dynamics_engine = ParallelOptimizedDynamicsEngine(
                    self.robot_model, self.parallel_manager.config
                )
            else:
                from ..algorithms.dynamics import DynamicsEngine
                self._dynamics_engine = DynamicsEngine(self.robot_model)
        return self._dynamics_engine
    
    @property
    def payload_identifier(self):
        """延迟初始化负载识别器"""
        if self._payload_identifier is None:
            from ..algorithms.payload_identification import PayloadIdentifier
            self._payload_identifier = PayloadIdentifier(self.robot_model)
        return self._payload_identifier
    
    @property
    def safety_monitor(self):
        """延迟初始化安全监控器"""
        if self._safety_monitor is None:
            from ..algorithms.safety import SafetyMonitor
            self._safety_monitor = SafetyMonitor(self.robot_model)
        return self._safety_monitor
    
    @property
    def collision_monitor(self):
        """延迟初始化碰撞监控器"""
        if self._collision_monitor is None:
            from ..algorithms.collision_detection import CollisionMonitor
            self._collision_monitor = CollisionMonitor(self.robot_model)
        return self._collision_monitor
    
    def plan_trajectory(
        self,
        waypoints: List[Waypoint],
        optimize_time: bool = True,
        payload: Optional[PayloadInfo] = None
    ) -> Trajectory:
        """
        规划轨迹
        
        Args:
            waypoints: 路径点列表
            optimize_time: 是否进行时间优化
            payload: 负载信息（可选）
        
        Returns:
            规划的轨迹
        """
        start_time = time.time()
        
        try:
            # 更新负载信息
            if payload:
                self.robot_model.update_payload(payload)
            
            # 创建路径
            path = waypoints
            
            # 轨迹规划
            if optimize_time:
                trajectory = self.trajectory_planner.generate_topp_trajectory(
                    path, self.robot_model.kinodynamic_limits
                )
            else:
                trajectory = self.trajectory_planner.interpolate_s7_trajectory(path)
            
            # 存储轨迹
            self.current_trajectory = trajectory
            self.trajectory_index = 0
            
            # 更新性能指标
            self.performance_metrics.computation_time = time.time() - start_time
            
            return trajectory
            
        except Exception as e:
            self.performance_metrics.success_rate = 0.0
            raise RuntimeError(f"轨迹规划失败: {e}")
    
    def compute_control(
        self,
        current_state: RobotState,
        target_time: Optional[float] = None
    ) -> ControlCommand:
        """
        计算控制指令
        
        Args:
            current_state: 当前机器人状态
            target_time: 目标时间（可选，默认使用当前时间戳）
        
        Returns:
            控制指令
        """
        start_time = time.time()
        
        try:
            # 记录当前状态
            self.state_history.add_state(current_state)
            
            # 安全检查
            if self.config.safety_check_enabled:
                safety_status = self.safety_monitor.check_safety(current_state)
                if not safety_status.is_safe:
                    self.emergency_stop = True
                    return self._generate_emergency_stop_command()
            
            # 获取参考轨迹点
            if target_time is None:
                target_time = current_state.timestamp
            
            reference_point = self._get_reference_point(target_time)
            if reference_point is None:
                return self._generate_hold_position_command(current_state)
            
            # 负载自适应
            if self.config.enable_payload_adaptation:
                self._update_payload_if_needed(current_state)
            
            # 路径控制
            control_command = self.path_controller.compute_control(
                reference_point, current_state
            )
            
            # 碰撞检测和避让
            collisions, avoidance_command = self.collision_monitor.update(
                current_state, control_command
            )
            
            # 应用碰撞避让
            if avoidance_command and avoidance_command.priority > 0.1:
                # 根据优先级混合原始指令和避让指令
                blend_factor = min(avoidance_command.priority, 1.0)
                
                if control_command.joint_velocities is not None:
                    control_command.joint_velocities = (
                        (1 - blend_factor) * control_command.joint_velocities +
                        blend_factor * avoidance_command.joint_velocities
                    )
                
                if control_command.joint_accelerations is not None:
                    control_command.joint_accelerations = (
                        (1 - blend_factor) * control_command.joint_accelerations +
                        blend_factor * avoidance_command.joint_accelerations
                    )
            
            # 抑振控制
            if self.config.enable_vibration_suppression:
                control_command = self.vibration_suppressor.apply_input_shaping(
                    control_command
                )
            
            # 安全限制
            control_command = self._apply_safety_limits(control_command)
            
            # 更新性能指标
            self._update_performance_metrics(current_state, reference_point, start_time)
            
            return control_command
            
        except Exception as e:
            self.performance_metrics.success_rate = 0.0
            return self._generate_emergency_stop_command()
    
    def _get_reference_point(self, target_time: float) -> Optional[TrajectoryPoint]:
        """
        获取参考轨迹点
        
        Args:
            target_time: 目标时间
        
        Returns:
            参考轨迹点
        """
        if not self.current_trajectory:
            return None
        
        # 找到最接近目标时间的轨迹点
        closest_point = None
        min_time_diff = float('inf')
        
        for point in self.current_trajectory:
            time_diff = abs(point.time - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_point = point
        
        return closest_point
    
    def _update_payload_if_needed(self, current_state: RobotState) -> None:
        """
        根据需要更新负载参数
        
        Args:
            current_state: 当前状态
        """
        # 检查是否需要重新识别负载
        if self.payload_identifier.should_reidentify(current_state):
            try:
                # 收集运动数据
                motion_data = self.state_history.states[-10:]  # 使用最近10个状态
                
                # 识别负载
                new_payload = self.payload_identifier.identify_payload(motion_data)
                
                # 更新模型
                if new_payload.identification_confidence > 0.8:
                    self.robot_model.update_payload(new_payload)
                    
            except Exception as e:
                # 负载识别失败，继续使用当前参数
                pass
    
    def _apply_safety_limits(self, command: ControlCommand) -> ControlCommand:
        """
        应用安全限制
        
        Args:
            command: 原始控制指令
        
        Returns:
            限制后的控制指令
        """
        # 位置限制
        if command.joint_positions is not None:
            min_pos, max_pos = self.robot_model.get_joint_limits()
            command.joint_positions = np.clip(
                command.joint_positions, min_pos, max_pos
            )
        
        # 速度限制
        if command.joint_velocities is not None:
            max_vel = self.robot_model.get_velocity_limits()
            command.joint_velocities = np.clip(
                command.joint_velocities, -max_vel, max_vel
            )
        
        # 力矩限制
        if command.joint_torques is not None:
            max_torque = self.robot_model.get_torque_limits()
            command.joint_torques = np.clip(
                command.joint_torques, -max_torque, max_torque
            )
        
        return command
    
    def _generate_emergency_stop_command(self) -> ControlCommand:
        """
        生成紧急停止指令
        
        Returns:
            紧急停止控制指令
        """
        return ControlCommand(
            joint_velocities=np.zeros(self.robot_model.n_joints),
            joint_accelerations=np.zeros(self.robot_model.n_joints),
            control_mode="velocity",
            timestamp=time.time()
        )
    
    def _generate_hold_position_command(self, current_state: RobotState) -> ControlCommand:
        """
        生成保持位置指令
        
        Args:
            current_state: 当前状态
        
        Returns:
            保持位置控制指令
        """
        return ControlCommand(
            joint_positions=current_state.joint_positions.copy(),
            joint_velocities=np.zeros(self.robot_model.n_joints),
            control_mode="position",
            timestamp=time.time()
        )
    
    def _update_performance_metrics(
        self,
        current_state: RobotState,
        reference_point: TrajectoryPoint,
        start_time: float
    ) -> None:
        """
        更新性能指标
        
        Args:
            current_state: 当前状态
            reference_point: 参考点
            start_time: 计算开始时间
        """
        # 计算时间
        self.performance_metrics.computation_time = time.time() - start_time
        
        # 跟踪误差
        if reference_point:
            error = np.linalg.norm(
                current_state.joint_positions - reference_point.position
            )
            self.performance_metrics.tracking_error = error
        
        # 振动幅度（简化计算）
        if len(self.state_history.states) >= 3:
            recent_positions = [
                state.joint_positions for state in self.state_history.states[-3:]
            ]
            vibration = np.std(recent_positions, axis=0).max()
            self.performance_metrics.vibration_amplitude = vibration
    
    def start_control(self) -> None:
        """启动控制"""
        self.is_active = True
        self.emergency_stop = False
    
    def stop_control(self) -> None:
        """停止控制"""
        self.is_active = False
    
    def emergency_stop_control(self) -> None:
        """紧急停止"""
        self.emergency_stop = True
        self.is_active = False
    
    def reset_controller(self) -> None:
        """重置控制器"""
        self.is_active = False
        self.emergency_stop = False
        self.current_trajectory = None
        self.trajectory_index = 0
        self.state_history.clear()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        获取性能指标
        
        Returns:
            当前性能指标
        """
        return self.performance_metrics
    
    def get_controller_status(self) -> Dict[str, Any]:
        """
        获取控制器状态
        
        Returns:
            控制器状态字典
        """
        status = {
            "is_active": self.is_active,
            "emergency_stop": self.emergency_stop,
            "has_trajectory": self.current_trajectory is not None,
            "trajectory_progress": (
                self.trajectory_index / len(self.current_trajectory)
                if self.current_trajectory else 0.0
            ),
            "performance_metrics": self.performance_metrics,
            "state_history_length": len(self.state_history.states),
            "parallel_computing_enabled": self.config.enable_parallel_computing
        }
        
        # 添加碰撞监控状态
        if hasattr(self, '_collision_monitor') and self._collision_monitor:
            status["collision_statistics"] = self.collision_monitor.get_collision_statistics()
        
        # 添加并行计算性能统计
        if self.config.enable_parallel_computing and self.parallel_manager:
            status["parallel_performance"] = self._get_parallel_performance_summary()
        
        return status
    
    def _get_parallel_performance_summary(self) -> Dict[str, Any]:
        """获取并行计算性能摘要"""
        summary = {
            "parallel_config": {
                "mode": self.parallel_manager.config.mode.value,
                "max_workers": self.parallel_manager.config.max_workers,
                "memory_optimization": self.parallel_manager.config.enable_memory_optimization
            }
        }
        
        # 动力学引擎性能
        if hasattr(self._dynamics_engine, 'get_performance_report'):
            summary["dynamics_performance"] = self._dynamics_engine.get_performance_report()
        
        # 轨迹规划器性能
        if hasattr(self._trajectory_planner, 'get_performance_report'):
            summary["trajectory_planner_performance"] = self._trajectory_planner.get_performance_report()
        
        # 路径控制器性能
        if hasattr(self._path_controller, 'get_parallel_performance_report'):
            summary["path_controller_performance"] = self._path_controller.get_parallel_performance_report()
        
        return summary
    
    def batch_compute_control_sequence(
        self,
        trajectory_batch: List[Trajectory],
        initial_states: List[RobotState],
        dt: float = 0.001
    ) -> List[List[ControlCommand]]:
        """
        批量计算控制序列
        
        Args:
            trajectory_batch: 轨迹批次
            initial_states: 初始状态列表
            dt: 时间步长
        
        Returns:
            控制指令序列列表
        """
        if not self.config.enable_parallel_computing or not self.parallel_manager:
            # 顺序处理
            return [
                self._compute_single_control_sequence(trajectory, initial_state, dt)
                for trajectory, initial_state in zip(trajectory_batch, initial_states)
            ]
        
        # 并行处理
        def compute_sequence_task(args):
            trajectory, initial_state = args
            return self._compute_single_control_sequence(trajectory, initial_state, dt)
        
        sequence_args = list(zip(trajectory_batch, initial_states))
        
        with self.parallel_manager as manager:
            scheduler = manager.scheduler
            control_sequences = scheduler.submit_tasks(
                compute_sequence_task,
                sequence_args,
                use_processes=False  # 使用线程以共享状态
            )
        
        return control_sequences
    
    def _compute_single_control_sequence(
        self,
        trajectory: Trajectory,
        initial_state: RobotState,
        dt: float
    ) -> List[ControlCommand]:
        """
        计算单个控制序列
        
        Args:
            trajectory: 轨迹
            initial_state: 初始状态
            dt: 时间步长
        
        Returns:
            控制指令列表
        """
        commands = []
        current_state = initial_state
        
        for reference_point in trajectory:
            command = self.compute_control(current_state, reference_point.time)
            commands.append(command)
            
            # 简化的状态更新（实际应用中需要更复杂的状态预测）
            if hasattr(command, 'joint_positions') and command.joint_positions is not None:
                current_state.joint_positions = command.joint_positions.copy()
        
        return commands
    
    def optimize_parallel_performance(
        self,
        sample_trajectories: List[Trajectory],
        sample_states: List[RobotState],
        optimization_iterations: int = 5
    ):
        """
        优化并行计算性能
        
        Args:
            sample_trajectories: 样本轨迹列表
            sample_states: 样本状态列表
            optimization_iterations: 优化迭代次数
        """
        if not self.config.enable_parallel_computing or not self.parallel_manager:
            print("并行计算未启用，跳过性能优化")
            return
        
        print("开始并行计算性能优化...")
        
        # 优化动力学引擎
        if hasattr(self._dynamics_engine, 'optimize_for_workload'):
            typical_batch_sizes = [len(traj) for traj in sample_trajectories[:5]]
            self._dynamics_engine.optimize_for_workload(
                typical_batch_sizes, optimization_iterations
            )
        
        # 优化轨迹规划器
        if hasattr(self._trajectory_planner, 'auto_tune_parallel_parameters'):
            sample_paths = [
                [point for point in traj] for traj in sample_trajectories[:3]
            ]
            sample_limits = [self.robot_model.kinodynamic_limits] * len(sample_paths)
            self._trajectory_planner.auto_tune_parallel_parameters(
                sample_paths, sample_limits
            )
        
        # 优化路径控制器
        if hasattr(self._path_controller, 'optimize_batch_threshold'):
            test_trajectories = [
                [point for point in traj] for traj in sample_trajectories[:3]
            ]
            test_states = [
                [state] * len(traj) for traj, state in zip(sample_trajectories[:3], sample_states[:3])
            ]
            self._path_controller.optimize_batch_threshold(test_trajectories, test_states)
        
        print("并行计算性能优化完成")
    
    def enable_parallel_computing(self, enable: bool = True):
        """
        启用/禁用并行计算
        
        Args:
            enable: 是否启用并行计算
        """
        self.config.enable_parallel_computing = enable
        
        if enable and self.parallel_manager is None:
            # 创建并行计算管理器
            parallel_config = ParallelConfig(
                mode=self.config.parallel_mode,
                max_workers=self.config.max_parallel_workers,
                enable_memory_optimization=True,
                enable_performance_monitoring=True
            )
            self.parallel_manager = ParallelComputingManager(parallel_config)
        elif not enable:
            # 清理并行计算管理器
            if self.parallel_manager:
                self.parallel_manager = None
        
        # 重置算法组件以使用新配置
        self._trajectory_planner = None
        self._path_controller = None
        self._dynamics_engine = None
    
    def set_parallel_config(
        self,
        mode: Optional[ParallelMode] = None,
        max_workers: Optional[int] = None,
        batch_threshold: Optional[int] = None
    ):
        """
        设置并行计算配置
        
        Args:
            mode: 并行模式
            max_workers: 最大工作线程数
            batch_threshold: 批处理阈值
        """
        if mode is not None:
            self.config.parallel_mode = mode
        
        if max_workers is not None:
            self.config.max_parallel_workers = max_workers
        
        if batch_threshold is not None:
            self.config.parallel_batch_threshold = batch_threshold
        
        # 重新创建并行计算管理器
        if self.config.enable_parallel_computing:
            parallel_config = ParallelConfig(
                mode=self.config.parallel_mode,
                max_workers=self.config.max_parallel_workers,
                enable_memory_optimization=True,
                enable_performance_monitoring=True
            )
            self.parallel_manager = ParallelComputingManager(parallel_config)
            
            # 重置算法组件
            self._trajectory_planner = None
            self._path_controller = None
            self._dynamics_engine = None