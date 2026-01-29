"""数据模型"""
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator

Vector = np.ndarray
Matrix = np.ndarray

@dataclass
class RobotState:
    """机器人状态"""
    joint_positions: Vector
    joint_velocities: Vector
    joint_accelerations: Vector
    joint_torques: Vector
    timestamp: float = 0.0

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: Vector
    velocity: Vector
    acceleration: Vector
    jerk: Vector
    time: float
    path_parameter: float

Trajectory = List[TrajectoryPoint]

class DynamicsParameters(BaseModel):
    """动力学参数"""
    masses: List[float]
    centers_of_mass: List[List[float]]
    inertias: List[List[List[float]]]
    friction_coeffs: List[float]
    gravity: List[float] = [0.0, 0.0, -9.81]

class KinodynamicLimits(BaseModel):
    """运动学动力学限制"""
    max_joint_positions: List[float]
    min_joint_positions: List[float]
    max_joint_velocities: List[float]
    max_joint_accelerations: List[float]
    max_joint_jerks: List[float]
    max_joint_torques: List[float]
    
    def validate_dimensions_consistency(self):
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
                raise ValueError(f"维度不一致")

class RobotModel:
    """机器人模型"""
    def __init__(self, name: str, n_joints: int, 
                 dynamics_params: DynamicsParameters,
                 kinodynamic_limits: KinodynamicLimits):
        self.name = name
        self.n_joints = n_joints
        self.dynamics_params = dynamics_params
        self.kinodynamic_limits = kinodynamic_limits
    
    @classmethod
    def create_test_model(cls, n_joints: int = 6):
        """创建测试模型"""
        dynamics_params = DynamicsParameters(
            masses=[5.0 + i * 2.0 for i in range(n_joints)],
            centers_of_mass=[[0.0, 0.0, 0.1 + i * 0.05] for i in range(n_joints)],
            inertias=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] for i in range(n_joints)],
            friction_coeffs=[0.1 + i * 0.02 for i in range(n_joints)]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi] * n_joints,
            min_joint_positions=[-np.pi] * n_joints,
            max_joint_velocities=[2.0] * n_joints,
            max_joint_accelerations=[10.0] * n_joints,
            max_joint_jerks=[50.0] * n_joints,
            max_joint_torques=[100.0] * n_joints
        )
        
        return cls("TestRobot", n_joints, dynamics_params, kinodynamic_limits)


@dataclass
class ControlCommand:
    """控制指令"""
    joint_positions: Optional[Vector] = None
    joint_velocities: Optional[Vector] = None
    joint_accelerations: Optional[Vector] = None
    joint_torques: Optional[Vector] = None
    timestamp: float = 0.0

@dataclass
class PayloadInfo:
    """负载信息"""
    mass: float = 0.0
    center_of_mass: Vector = None
    inertia: Matrix = None
    identification_confidence: float = 0.0
    
    def __post_init__(self):
        if self.center_of_mass is None:
            self.center_of_mass = np.zeros(3)
        if self.inertia is None:
            self.inertia = np.eye(3)

class AlgorithmError(Exception):
    """算法异常"""
    pass

class PerformanceMetrics(BaseModel):
    """性能指标"""
    tracking_accuracy: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    energy_efficiency: float = 0.0
    vibration_suppression: float = 0.0
    safety_margin: float = 0.0
