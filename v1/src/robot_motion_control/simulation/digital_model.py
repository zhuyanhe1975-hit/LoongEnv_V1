"""
数字化机器人模型

实现高保真的机器人数字化仿真模型，用于算法验证和测试。
支持完整的动力学仿真、传感器噪声模拟和物理特性建模。
"""

from typing import Optional, Dict, Any
import numpy as np
import time
from dataclasses import dataclass

from ..core.models import RobotModel
from ..core.types import (
    RobotState, ControlCommand, NoiseConfig, 
    SimulationConfig, Vector, AlgorithmError
)
from ..algorithms.dynamics import DynamicsEngine


@dataclass
class SensorData:
    """传感器数据结构"""
    joint_positions: Vector
    joint_velocities: Vector
    joint_torques: Vector
    end_effector_pose: np.ndarray
    timestamp: float
    noise_level: float = 0.0
    # 新增末端执行器传感器数据
    end_effector_position: Optional[Vector] = None
    end_effector_velocity: Optional[Vector] = None
    end_effector_acceleration: Optional[Vector] = None
    end_effector_force: Optional[Vector] = None
    end_effector_torque: Optional[Vector] = None


class RobotDigitalModel:
    """
    数字化机器人模型类
    
    提供高保真的机器人物理仿真，包括动力学计算、
    传感器噪声模拟和物理特性参数化。
    """
    
    def __init__(
        self,
        robot_model: RobotModel,
        config: Optional[SimulationConfig] = None,
        noise_config: Optional[NoiseConfig] = None
    ):
        """
        初始化数字化机器人模型
        
        Args:
            robot_model: 机器人模型
            config: 仿真配置
            noise_config: 噪声配置
        """
        self.robot_model = robot_model
        self.config = config or SimulationConfig()
        self.noise_config = noise_config or NoiseConfig()
        
        # 动力学引擎
        self.dynamics_engine = DynamicsEngine(robot_model)
        
        # 当前状态
        self.current_state = self._initialize_state()
        
        # 仿真时间
        self.simulation_time = 0.0
        self.time_step = self.config.simulation_time_step
        
        # 传感器延迟缓冲区
        self.sensor_buffer = []
        self.max_buffer_size = int(
            self.noise_config.sensor_delay / self.time_step
        ) + 1
        
        # 噪声生成器
        self.noise_generator = np.random.RandomState(42)  # 固定种子以便重现
    
    def _initialize_state(self) -> RobotState:
        """初始化机器人状态"""
        n_joints = self.robot_model.n_joints
        
        return RobotState(
            joint_positions=np.zeros(n_joints),
            joint_velocities=np.zeros(n_joints),
            joint_accelerations=np.zeros(n_joints),
            joint_torques=np.zeros(n_joints),
            end_effector_pose=np.eye(4),
            timestamp=0.0
        )
    
    def simulate_step(self, command: ControlCommand, dt: float) -> RobotState:
        """
        执行一步仿真
        
        Args:
            command: 控制指令
            dt: 时间步长
        
        Returns:
            更新后的机器人状态
        """
        try:
            # 根据控制模式执行仿真
            if command.control_mode == "position":
                new_state = self._simulate_position_control(command, dt)
            elif command.control_mode == "velocity":
                new_state = self._simulate_velocity_control(command, dt)
            elif command.control_mode == "torque":
                new_state = self._simulate_torque_control(command, dt)
            else:
                raise ValueError(f"不支持的控制模式: {command.control_mode}")
            
            # 添加噪声
            if self.config.enable_noise:
                new_state = self._add_noise(new_state)
            
            # 更新当前状态
            self.current_state = new_state
            self.simulation_time += dt
            
            # 更新传感器缓冲区
            self._update_sensor_buffer(new_state)
            
            return new_state
            
        except Exception as e:
            raise AlgorithmError(f"仿真步骤执行失败: {e}")
    
    def _simulate_position_control(self, command: ControlCommand, dt: float) -> RobotState:
        """位置控制仿真"""
        if command.joint_positions is None:
            return self.current_state
        
        # 简化的位置控制仿真：一阶系统
        target_positions = command.joint_positions
        current_positions = self.current_state.joint_positions
        
        # 计算位置变化（简化为指数收敛）
        position_gain = 5.0  # 位置控制增益
        position_change = position_gain * (target_positions - current_positions) * dt
        
        # 更新位置和速度
        new_positions = current_positions + position_change
        new_velocities = position_change / dt
        
        # 计算加速度
        new_accelerations = (new_velocities - self.current_state.joint_velocities) / dt
        
        # 通过逆动力学计算所需力矩
        new_torques = self.dynamics_engine.inverse_dynamics(
            new_positions, new_velocities, new_accelerations
        )
        
        return RobotState(
            joint_positions=new_positions,
            joint_velocities=new_velocities,
            joint_accelerations=new_accelerations,
            joint_torques=new_torques,
            end_effector_pose=self._compute_forward_kinematics(new_positions),
            timestamp=self.simulation_time + dt
        )
    
    def _simulate_velocity_control(self, command: ControlCommand, dt: float) -> RobotState:
        """速度控制仿真"""
        if command.joint_velocities is None:
            return self.current_state
        
        # 积分得到位置
        new_positions = self.current_state.joint_positions + command.joint_velocities * dt
        new_velocities = command.joint_velocities.copy()
        
        # 计算加速度
        new_accelerations = (new_velocities - self.current_state.joint_velocities) / dt
        
        # 通过逆动力学计算所需力矩
        new_torques = self.dynamics_engine.inverse_dynamics(
            new_positions, new_velocities, new_accelerations
        )
        
        return RobotState(
            joint_positions=new_positions,
            joint_velocities=new_velocities,
            joint_accelerations=new_accelerations,
            joint_torques=new_torques,
            end_effector_pose=self._compute_forward_kinematics(new_positions),
            timestamp=self.simulation_time + dt
        )
    
    def _simulate_torque_control(self, command: ControlCommand, dt: float) -> RobotState:
        """力矩控制仿真"""
        if command.joint_torques is None:
            return self.current_state
        
        # 通过正向动力学计算加速度
        new_accelerations = self.dynamics_engine.forward_dynamics(
            self.current_state.joint_positions,
            self.current_state.joint_velocities,
            command.joint_torques
        )
        
        # 积分得到速度和位置
        new_velocities = self.current_state.joint_velocities + new_accelerations * dt
        new_positions = self.current_state.joint_positions + new_velocities * dt
        
        return RobotState(
            joint_positions=new_positions,
            joint_velocities=new_velocities,
            joint_accelerations=new_accelerations,
            joint_torques=command.joint_torques.copy(),
            end_effector_pose=self._compute_forward_kinematics(new_positions),
            timestamp=self.simulation_time + dt
        )
    
    def _compute_forward_kinematics(self, joint_positions: Vector) -> np.ndarray:
        """计算正向运动学"""
        # 简化实现：返回单位矩阵
        # TODO: 实现完整的正向运动学计算
        return np.eye(4)
    
    def _add_noise(self, state: RobotState) -> RobotState:
        """添加传感器噪声"""
        # 位置噪声
        position_noise = self.noise_generator.normal(
            0, self.noise_config.position_noise_std, len(state.joint_positions)
        )
        
        # 速度噪声
        velocity_noise = self.noise_generator.normal(
            0, self.noise_config.velocity_noise_std, len(state.joint_velocities)
        )
        
        # 力矩噪声
        torque_noise = self.noise_generator.normal(
            0, self.noise_config.torque_noise_std, len(state.joint_torques)
        )
        
        return RobotState(
            joint_positions=state.joint_positions + position_noise,
            joint_velocities=state.joint_velocities + velocity_noise,
            joint_accelerations=state.joint_accelerations,  # 不添加噪声
            joint_torques=state.joint_torques + torque_noise,
            end_effector_pose=state.end_effector_pose,
            timestamp=state.timestamp
        )
    
    def _update_sensor_buffer(self, state: RobotState) -> None:
        """更新传感器缓冲区"""
        sensor_data = SensorData(
            joint_positions=state.joint_positions.copy(),
            joint_velocities=state.joint_velocities.copy(),
            joint_torques=state.joint_torques.copy(),
            end_effector_pose=state.end_effector_pose.copy(),
            timestamp=state.timestamp,
            noise_level=self.noise_config.position_noise_std
        )
        
        self.sensor_buffer.append(sensor_data)
        
        # 限制缓冲区大小
        if len(self.sensor_buffer) > self.max_buffer_size:
            self.sensor_buffer.pop(0)
    
    def get_sensor_data(self) -> SensorData:
        """
        获取传感器数据（包含延迟）
        
        Returns:
            传感器数据
        """
        if not self.sensor_buffer:
            # 返回当前状态作为传感器数据
            return SensorData(
                joint_positions=self.current_state.joint_positions.copy(),
                joint_velocities=self.current_state.joint_velocities.copy(),
                joint_torques=self.current_state.joint_torques.copy(),
                end_effector_pose=self.current_state.end_effector_pose.copy(),
                timestamp=self.current_state.timestamp
            )
        
        # 考虑传感器延迟
        delay_steps = int(self.noise_config.sensor_delay / self.time_step)
        if delay_steps >= len(self.sensor_buffer):
            return self.sensor_buffer[0]
        else:
            return self.sensor_buffer[-(delay_steps + 1)]
    
    def reset_simulation(self) -> None:
        """重置仿真"""
        self.current_state = self._initialize_state()
        self.simulation_time = 0.0
        self.sensor_buffer.clear()
    
    def set_robot_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置机器人参数
        
        Args:
            params: 参数字典
        """
        # 更新动力学参数
        if "masses" in params:
            self.robot_model.dynamics_params.masses = params["masses"]
        
        if "friction_coeffs" in params:
            self.robot_model.dynamics_params.friction_coeffs = params["friction_coeffs"]
        
        # 重新初始化动力学引擎
        self.dynamics_engine = DynamicsEngine(self.robot_model)
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """
        获取仿真状态
        
        Returns:
            仿真状态字典
        """
        return {
            "current_state": self.current_state,
            "simulation_time": self.simulation_time,
            "time_step": self.time_step,
            "sensor_buffer_size": len(self.sensor_buffer),
            "noise_enabled": self.config.enable_noise
        }