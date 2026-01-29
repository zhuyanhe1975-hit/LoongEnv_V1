"""
核心数据类型定义

定义机器人运动控制系统中使用的所有核心数据结构和类型注解。
基于设计文档中的数据模型实现，提供类型安全和数据验证。
"""

from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator


# 基础数据类型别名
Vector = np.ndarray  # 一维向量
Matrix = np.ndarray  # 二维矩阵
Vector3 = np.ndarray  # 3D向量
Matrix3 = np.ndarray  # 3x3矩阵


@dataclass
class Pose:
    """位姿数据结构"""
    position: Vector3        # 位置 [x, y, z]
    orientation: Matrix3     # 旋转矩阵 (3x3)
    
    def __post_init__(self):
        """验证数据维度"""
        assert len(self.position) == 3, "位置必须是3D向量"
        assert self.orientation.shape == (3, 3), "旋转矩阵必须是3x3"


@dataclass
class RobotState:
    """
    机器人状态数据结构
    
    包含机器人在某一时刻的完整状态信息，包括关节空间和笛卡尔空间的位置、速度、加速度等。
    """
    joint_positions: Vector      # 关节位置 [rad]
    joint_velocities: Vector     # 关节速度 [rad/s]
    joint_accelerations: Vector  # 关节加速度 [rad/s²]
    joint_torques: Vector        # 关节力矩 [Nm]
    end_effector_transform: Matrix  # 末端位姿 (4x4齐次变换矩阵)
    timestamp: float             # 时间戳 [s]
    
    def __post_init__(self):
        """验证数据维度一致性"""
        n_joints = len(self.joint_positions)
        assert len(self.joint_velocities) == n_joints, "速度维度不匹配"
        assert len(self.joint_accelerations) == n_joints, "加速度维度不匹配"
        assert len(self.joint_torques) == n_joints, "力矩维度不匹配"
        assert self.end_effector_transform.shape == (4, 4), "末端位姿必须是4x4矩阵"


@dataclass
class TrajectoryPoint:
    """
    轨迹点数据结构
    
    描述轨迹上某一点的运动状态，包括位置、速度、加速度、加加速度等信息。
    """
    position: Vector             # 位置 (关节空间或笛卡尔空间)
    velocity: Vector             # 速度
    acceleration: Vector         # 加速度
    jerk: Vector                 # 加加速度
    time: float                  # 时间 [s]
    path_parameter: float        # 路径参数 [0, 1]
    
    def __post_init__(self):
        """验证数据维度一致性"""
        n_dof = len(self.position)
        assert len(self.velocity) == n_dof, "速度维度不匹配"
        assert len(self.acceleration) == n_dof, "加速度维度不匹配"
        assert len(self.jerk) == n_dof, "加加速度维度不匹配"
        assert 0.0 <= self.path_parameter <= 1.0, "路径参数必须在[0,1]范围内"


class DynamicsParameters(BaseModel):
    """
    动力学参数数据结构
    
    包含机器人的质量、惯性、摩擦等物理参数。
    使用Pydantic进行数据验证和序列化。
    """
    masses: List[float] = Field(..., description="连杆质量 [kg]")
    centers_of_mass: List[List[float]] = Field(..., description="质心位置 [m]")
    inertias: List[List[List[float]]] = Field(..., description="惯量张量 [kg⋅m²]")
    friction_coeffs: List[float] = Field(..., description="摩擦系数")
    gravity: List[float] = Field(default=[0.0, 0.0, -9.81], description="重力向量 [m/s²]")
    
    @validator('centers_of_mass')
    def validate_com_dimensions(cls, v):
        """验证质心位置为3D向量"""
        for com in v:
            if len(com) != 3:
                raise ValueError("质心位置必须是3D向量")
        return v
    
    @validator('inertias')
    def validate_inertia_dimensions(cls, v):
        """验证惯量张量为3x3矩阵"""
        for inertia in v:
            if len(inertia) != 3 or any(len(row) != 3 for row in inertia):
                raise ValueError("惯量张量必须是3x3矩阵")
        return v
    
    @validator('gravity')
    def validate_gravity_dimension(cls, v):
        """验证重力向量为3D"""
        if len(v) != 3:
            raise ValueError("重力向量必须是3D")
        return v


class PayloadInfo(BaseModel):
    """
    负载信息数据结构
    
    描述机器人末端负载的物理特性，用于动力学计算和控制参数调整。
    """
    mass: float = Field(..., ge=0.0, description="负载质量 [kg]")
    center_of_mass: List[float] = Field(..., description="负载质心 [m]")
    inertia: List[List[float]] = Field(..., description="负载惯量张量 [kg⋅m²]")
    identification_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="识别置信度 [0-1]"
    )
    
    @validator('center_of_mass')
    def validate_com_dimension(cls, v):
        """验证质心为3D向量"""
        if len(v) != 3:
            raise ValueError("质心必须是3D向量")
        return v
    
    @validator('inertia')
    def validate_inertia_dimension(cls, v):
        """验证惯量张量为3x3矩阵"""
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("惯量张量必须是3x3矩阵")
        return v


class KinodynamicLimits(BaseModel):
    """
    运动学和动力学限制
    
    定义机器人各关节的位置、速度、加速度、加加速度和力矩限制。
    """
    max_joint_positions: List[float] = Field(..., description="最大关节位置 [rad]")
    min_joint_positions: List[float] = Field(..., description="最小关节位置 [rad]")
    max_joint_velocities: List[float] = Field(..., description="最大关节速度 [rad/s]")
    max_joint_accelerations: List[float] = Field(..., description="最大关节加速度 [rad/s²]")
    max_joint_jerks: List[float] = Field(..., description="最大关节加加速度 [rad/s³]")
    max_joint_torques: List[float] = Field(..., description="最大关节力矩 [Nm]")
    
    @validator('max_joint_velocities', 'max_joint_accelerations', 'max_joint_jerks', 'max_joint_torques')
    def validate_positive_limits(cls, v):
        """验证限制值为正数"""
        if any(val <= 0 for val in v):
            raise ValueError("运动限制必须为正数")
        return v
    
    def validate_dimensions_consistency(self):
        """验证所有限制的维度一致性"""
        n_joints = len(self.max_joint_positions)
        limits = [
            self.min_joint_positions,
            self.max_joint_velocities,
            self.max_joint_accelerations,
            self.max_joint_jerks,
            self.max_joint_torques
        ]
        
        for limit in limits:
            if len(limit) != n_joints:
                raise ValueError(f"所有限制的维度必须一致，期望{n_joints}，实际{len(limit)}")


@dataclass
class ControlCommand:
    """
    控制指令数据结构
    
    包含发送给机器人的控制指令，可以是位置、速度、力矩或混合控制模式。
    """
    joint_positions: Optional[Vector] = None     # 目标关节位置 [rad]
    joint_velocities: Optional[Vector] = None    # 目标关节速度 [rad/s]
    joint_accelerations: Optional[Vector] = None # 目标关节加速度 [rad/s²]
    joint_torques: Optional[Vector] = None       # 目标关节力矩 [Nm]
    control_mode: str = "position"               # 控制模式: position, velocity, torque, hybrid
    timestamp: float = 0.0                       # 指令时间戳 [s]
    
    def __post_init__(self):
        """验证控制指令的有效性"""
        valid_modes = ["position", "velocity", "torque", "hybrid"]
        if self.control_mode not in valid_modes:
            raise ValueError(f"无效的控制模式: {self.control_mode}")
        
        # 根据控制模式验证必要的字段
        if self.control_mode == "position" and self.joint_positions is None:
            raise ValueError("位置控制模式需要提供目标位置")
        elif self.control_mode == "velocity" and self.joint_velocities is None:
            raise ValueError("速度控制模式需要提供目标速度")
        elif self.control_mode == "torque" and self.joint_torques is None:
            raise ValueError("力矩控制模式需要提供目标力矩")


# 轨迹类型定义
Trajectory = List[TrajectoryPoint]  # 轨迹为轨迹点的序列


# 路径类型定义 (用于轨迹规划)
@dataclass
class Waypoint:
    """路径点数据结构"""
    position: Vector        # 位置
    velocity: Optional[Vector] = None  # 可选的速度约束
    acceleration: Optional[Vector] = None  # 可选的加速度约束
    time_constraint: Optional[float] = None  # 可选的时间约束


Path = List[Waypoint]  # 路径为路径点的序列


# 仿真相关数据结构
class SimulationConfig(BaseModel):
    """仿真配置参数"""
    control_frequency: float = Field(default=1000.0, gt=0, description="控制频率 [Hz]")
    simulation_time_step: float = Field(default=0.001, gt=0, description="仿真时间步长 [s]")
    enable_noise: bool = Field(default=False, description="是否启用噪声")
    noise_std: float = Field(default=0.01, ge=0, description="噪声标准差")
    enable_visualization: bool = Field(default=False, description="是否启用可视化")


class NoiseConfig(BaseModel):
    """噪声配置参数"""
    position_noise_std: float = Field(default=0.001, ge=0, description="位置噪声标准差 [rad]")
    velocity_noise_std: float = Field(default=0.01, ge=0, description="速度噪声标准差 [rad/s]")
    torque_noise_std: float = Field(default=0.1, ge=0, description="力矩噪声标准差 [Nm]")
    sensor_delay: float = Field(default=0.001, ge=0, description="传感器延迟 [s]")


# 性能监控相关数据结构
@dataclass
class PerformanceMetrics:
    """算法性能指标"""
    computation_time: float      # 计算时间 [s]
    memory_usage: float          # 内存使用 [MB]
    tracking_error: float        # 跟踪误差 [m]
    vibration_amplitude: float   # 振动幅度 [m]
    success_rate: float          # 成功率 [0-1]


# 错误处理相关类型
class AlgorithmError(Exception):
    """算法计算异常基类"""
    pass


class NumericalInstabilityError(AlgorithmError):
    """数值不稳定异常"""
    pass


class ConvergenceError(AlgorithmError):
    """收敛失败异常"""
    pass


class SafetyViolationError(AlgorithmError):
    """安全限制违反异常"""
    pass