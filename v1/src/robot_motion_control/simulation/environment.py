"""
仿真环境模块

提供机器人仿真环境，支持状态更新和轨迹执行
"""

import numpy as np
import time
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..core.models import RobotModel
from ..core.types import RobotState, TrajectoryPoint, Trajectory


@dataclass
class SimulationConfig:
    """仿真配置"""
    update_frequency: float = 1000.0  # Hz
    enable_noise: bool = True
    noise_level: float = 0.001
    enable_dynamics: bool = True


class SimulationEnvironment:
    """
    机器人仿真环境
    
    提供虚拟机器人的状态仿真和轨迹执行功能
    """
    
    def __init__(
        self, 
        robot_model: RobotModel,
        config: Optional[SimulationConfig] = None
    ):
        """
        初始化仿真环境
        
        Args:
            robot_model: 机器人模型
            config: 仿真配置
        """
        self.robot_model = robot_model
        self.config = config or SimulationConfig()
        
        # 仿真状态
        self.is_running = False
        self.current_state = RobotState(
            joint_positions=np.zeros(robot_model.n_joints),
            joint_velocities=np.zeros(robot_model.n_joints),
            joint_accelerations=np.zeros(robot_model.n_joints),
            joint_torques=np.zeros(robot_model.n_joints),
            end_effector_transform=np.eye(4),
            timestamp=time.time()
        )
        
        # 目标状态
        self.target_state = self.current_state
        
        # 仿真线程
        self.simulation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 轨迹执行
        self.current_trajectory: Optional[Trajectory] = None
        self.trajectory_start_time = 0.0
        
        # 运动参数
        self.max_joint_velocities = np.ones(robot_model.n_joints) * 2.0  # rad/s
        self.max_joint_accelerations = np.ones(robot_model.n_joints) * 5.0  # rad/s²
        
        # 仿真时间
        self.simulation_time = 0.0
        self.start_time = time.time()
    
    def start_simulation(self, duration: Optional[float] = None):
        """
        启动仿真
        
        Args:
            duration: 仿真持续时间（秒），None表示无限运行
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.simulation_time = 0.0
        
        # 启动仿真线程
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(duration,),
            daemon=True
        )
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """停止仿真"""
        self.is_running = False
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
    
    def set_target_state(self, target_state: RobotState):
        """设置目标状态"""
        self.target_state = target_state
    
    def execute_trajectory(self, trajectory: Trajectory):
        """
        执行轨迹
        
        Args:
            trajectory: 要执行的轨迹
        """
        self.current_trajectory = trajectory
        self.trajectory_start_time = self.simulation_time
    
    def get_current_state(self) -> RobotState:
        """获取当前状态"""
        return RobotState(
            joint_positions=self.current_state.joint_positions.copy(),
            joint_velocities=self.current_state.joint_velocities.copy(),
            joint_accelerations=self.current_state.joint_accelerations.copy(),
            joint_torques=self.current_state.joint_torques.copy(),
            end_effector_transform=self.current_state.end_effector_transform.copy(),
            timestamp=self.current_state.timestamp
        )
    
    def _simulation_loop(self, duration: Optional[float]):
        """仿真主循环"""
        dt = 1.0 / self.config.update_frequency
        
        while self.is_running and not self.stop_event.is_set():
            loop_start = time.time()
            
            # 更新仿真时间
            self.simulation_time = time.time() - self.start_time
            
            # 检查持续时间
            if duration and self.simulation_time >= duration:
                break
            
            # 更新机器人状态
            self._update_robot_state(dt)
            
            # 更新时间戳
            self.current_state.timestamp = time.time()
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.is_running = False
    
    def _update_robot_state(self, dt: float):
        """
        更新机器人状态
        
        Args:
            dt: 时间步长
        """
        # 如果有轨迹在执行，跟随轨迹
        if self.current_trajectory:
            trajectory_time = self.simulation_time - self.trajectory_start_time
            target_point = self._get_trajectory_point_at_time(trajectory_time)
            
            if target_point:
                self.target_state = RobotState(
                    joint_positions=target_point.position,
                    joint_velocities=target_point.velocity,
                    joint_accelerations=target_point.acceleration,
                    joint_torques=np.zeros(self.robot_model.n_joints),
                    end_effector_transform=np.eye(4),
                    timestamp=time.time()
                )
            else:
                # 轨迹执行完成
                self.current_trajectory = None
        
        # 简化的运动控制：向目标状态移动
        position_error = self.target_state.joint_positions - self.current_state.joint_positions
        velocity_error = self.target_state.joint_velocities - self.current_state.joint_velocities
        
        # PD控制器
        kp = 10.0
        kd = 2.0
        
        desired_acceleration = kp * position_error + kd * velocity_error
        
        # 限制加速度
        desired_acceleration = np.clip(
            desired_acceleration,
            -self.max_joint_accelerations,
            self.max_joint_accelerations
        )
        
        # 更新速度
        new_velocity = self.current_state.joint_velocities + desired_acceleration * dt
        
        # 限制速度
        new_velocity = np.clip(
            new_velocity,
            -self.max_joint_velocities,
            self.max_joint_velocities
        )
        
        # 更新位置
        new_position = self.current_state.joint_positions + new_velocity * dt
        
        # 添加噪声
        if self.config.enable_noise:
            noise = np.random.normal(0, self.config.noise_level, self.robot_model.n_joints)
            new_position += noise
            new_velocity += noise * 0.1
        
        # 应用关节限制
        new_position = self._apply_joint_limits(new_position)
        
        # 计算力矩（简化）
        joint_torques = desired_acceleration * 0.1  # 简化的惯性模型
        
        # 更新状态
        self.current_state.joint_positions = new_position
        self.current_state.joint_velocities = new_velocity
        self.current_state.joint_accelerations = desired_acceleration
        self.current_state.joint_torques = joint_torques
    
    def _get_trajectory_point_at_time(self, t: float) -> Optional[TrajectoryPoint]:
        """
        获取指定时间的轨迹点
        
        Args:
            t: 时间
            
        Returns:
            轨迹点或None
        """
        if not self.current_trajectory:
            return None
        
        # 找到最接近的轨迹点
        best_point = None
        min_time_diff = float('inf')
        
        for point in self.current_trajectory:
            time_diff = abs(point.time - t)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_point = point
        
        return best_point
    
    def _apply_joint_limits(self, positions: np.ndarray) -> np.ndarray:
        """
        应用关节限制
        
        Args:
            positions: 关节位置
            
        Returns:
            限制后的关节位置
        """
        # 简化的关节限制
        joint_limits = np.pi * np.ones(self.robot_model.n_joints)
        
        return np.clip(positions, -joint_limits, joint_limits)
    
    def reset(self):
        """重置仿真环境"""
        self.stop_simulation()
        
        self.current_state = RobotState(
            joint_positions=np.zeros(self.robot_model.n_joints),
            joint_velocities=np.zeros(self.robot_model.n_joints),
            joint_accelerations=np.zeros(self.robot_model.n_joints),
            joint_torques=np.zeros(self.robot_model.n_joints),
            end_effector_transform=np.eye(4),
            timestamp=time.time()
        )
        
        self.target_state = self.current_state
        self.current_trajectory = None
        self.simulation_time = 0.0
    
    def simulate_step(self, control_command, dt: float) -> RobotState:
        """
        仿真单步执行
        
        Args:
            control_command: 控制指令
            dt: 时间步长
            
        Returns:
            更新后的机器人状态
        """
        # 简化的动力学仿真
        # 假设控制指令直接作用于关节加速度
        if hasattr(control_command, 'joint_torques'):
            # 简化的力矩到加速度转换
            desired_acceleration = control_command.joint_torques * 0.1
        elif hasattr(control_command, 'joint_accelerations'):
            desired_acceleration = control_command.joint_accelerations
        else:
            # 默认零加速度
            desired_acceleration = np.zeros(self.robot_model.n_joints)
        
        # 限制加速度
        desired_acceleration = np.clip(
            desired_acceleration,
            -self.max_joint_accelerations,
            self.max_joint_accelerations
        )
        
        # 更新速度
        new_velocity = self.current_state.joint_velocities + desired_acceleration * dt
        
        # 限制速度
        new_velocity = np.clip(
            new_velocity,
            -self.max_joint_velocities,
            self.max_joint_velocities
        )
        
        # 更新位置
        new_position = self.current_state.joint_positions + new_velocity * dt
        
        # 添加噪声
        if self.config.enable_noise:
            noise = np.random.normal(0, self.config.noise_level, self.robot_model.n_joints)
            new_position += noise
            new_velocity += noise * 0.1
        
        # 应用关节限制
        new_position = self._apply_joint_limits(new_position)
        
        # 计算力矩（简化）
        joint_torques = desired_acceleration * 0.1
        
        # 更新状态
        self.current_state.joint_positions = new_position
        self.current_state.joint_velocities = new_velocity
        self.current_state.joint_accelerations = desired_acceleration
        self.current_state.joint_torques = joint_torques
        self.current_state.timestamp = time.time()
        
        return self.get_current_state()
    
    def add_noise_and_disturbances(self):
        """添加噪声和干扰"""
        self.config.enable_noise = True
        self.config.noise_level = 0.002  # 增加噪声水平
        
        # 可以添加其他干扰，如外力、传感器噪声等
        print("已启用噪声和干扰")

    def get_simulation_info(self) -> Dict[str, Any]:
        """获取仿真信息"""
        return {
            'is_running': self.is_running,
            'simulation_time': self.simulation_time,
            'update_frequency': self.config.update_frequency,
            'has_trajectory': self.current_trajectory is not None,
            'trajectory_points': len(self.current_trajectory) if self.current_trajectory else 0
        }