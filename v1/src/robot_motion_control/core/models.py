"""
核心模型定义

定义机器人模型、状态管理等核心数据模型类。
提供机器人几何、动力学参数的封装和管理功能。
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

from .types import (
    Vector, Matrix, Pose, DynamicsParameters, KinodynamicLimits,
    RobotState, TrajectoryPoint, PayloadInfo
)


class RobotModel:
    """
    机器人模型类
    
    封装机器人的几何、动力学参数和运动学模型。
    支持从URDF、MJCF等格式加载机器人模型。
    """
    
    def __init__(
        self,
        name: str,
        n_joints: int,
        dynamics_params: DynamicsParameters,
        kinodynamic_limits: KinodynamicLimits,
        urdf_path: Optional[str] = None,
        mjcf_path: Optional[str] = None
    ):
        """
        初始化机器人模型
        
        Args:
            name: 机器人名称
            n_joints: 关节数量
            dynamics_params: 动力学参数
            kinodynamic_limits: 运动学动力学限制
            urdf_path: URDF文件路径（可选）
            mjcf_path: MJCF文件路径（可选）
        """
        self.name = name
        self.n_joints = n_joints
        self.dynamics_params = dynamics_params
        self.kinodynamic_limits = kinodynamic_limits
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        
        # 验证参数一致性
        self._validate_parameters()
        
        # 当前负载信息
        self.current_payload: Optional[PayloadInfo] = None
        
        # 模型元数据
        self.metadata: Dict[str, Any] = {}
    
    def _validate_parameters(self) -> None:
        """验证模型参数的一致性"""
        # 验证动力学参数维度
        if len(self.dynamics_params.masses) != self.n_joints:
            raise ValueError(f"质量参数数量({len(self.dynamics_params.masses)})与关节数({self.n_joints})不匹配")
        
        if len(self.dynamics_params.centers_of_mass) != self.n_joints:
            raise ValueError(f"质心参数数量({len(self.dynamics_params.centers_of_mass)})与关节数({self.n_joints})不匹配")
        
        if len(self.dynamics_params.inertias) != self.n_joints:
            raise ValueError(f"惯量参数数量({len(self.dynamics_params.inertias)})与关节数({self.n_joints})不匹配")
        
        # 验证运动学限制维度
        self.kinodynamic_limits.validate_dimensions_consistency()
    
    @classmethod
    def from_urdf(cls, urdf_path: str, name: Optional[str] = None) -> 'RobotModel':
        """
        从URDF文件加载机器人模型
        
        Args:
            urdf_path: URDF文件路径
            name: 机器人名称（可选，默认使用文件名）
        
        Returns:
            RobotModel实例
        """
        if name is None:
            name = Path(urdf_path).stem
        
        # TODO: 实际实现需要解析URDF文件
        # 这里提供一个示例实现框架
        
        # 示例参数（实际应从URDF解析）
        n_joints = 6  # 假设6轴机器人
        
        dynamics_params = DynamicsParameters(
            masses=[10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            centers_of_mass=[[0.0, 0.0, 0.1]] * n_joints,
            inertias=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * n_joints,
            friction_coeffs=[0.1] * n_joints,
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi] * n_joints,
            min_joint_positions=[-np.pi] * n_joints,
            max_joint_velocities=[2.0] * n_joints,
            max_joint_accelerations=[10.0] * n_joints,
            max_joint_jerks=[50.0] * n_joints,
            max_joint_torques=[100.0] * n_joints
        )
        
        return cls(
            name=name,
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits,
            urdf_path=urdf_path
        )
    
    @classmethod
    def from_mjcf(cls, mjcf_path: str, name: Optional[str] = None) -> 'RobotModel':
        """
        从MJCF文件加载机器人模型
        
        Args:
            mjcf_path: MJCF文件路径
            name: 机器人名称（可选，默认使用文件名）
        
        Returns:
            RobotModel实例
        """
        from .mjcf_parser import MJCFParser
        
        # 解析MJCF文件
        parser = MJCFParser(mjcf_path)
        
        if name is None:
            name = parser.model_name or Path(mjcf_path).stem
        
        # 从MJCF提取参数
        dynamics_params = parser.extract_dynamics_parameters()
        kinodynamic_limits = parser.extract_kinodynamic_limits()
        n_joints = parser.get_joint_count()
        
        model = cls(
            name=name,
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits,
            mjcf_path=mjcf_path
        )
        
        # 添加解析器信息到元数据
        model.metadata.update(parser.get_model_info())
        
        return model
    
    @classmethod
    def create_er15_1400(cls, mjcf_path: Optional[str] = None) -> 'RobotModel':
        """
        创建ER15-1400机械臂模型
        
        Args:
            mjcf_path: MJCF文件路径（可选，使用默认路径）
        
        Returns:
            ER15-1400 RobotModel实例
        """
        if mjcf_path is None:
            # 使用默认的ER15-1400模型路径
            mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
        
        return cls.from_mjcf(mjcf_path, name="ER15-1400")
    
    @classmethod
    def create_test_model(cls, n_joints: int = 6, name: str = "TestRobot") -> 'RobotModel':
        """
        创建测试用机器人模型
        
        Args:
            n_joints: 关节数量
            name: 机器人名称
        
        Returns:
            测试用RobotModel实例
        """
        # 创建测试用动力学参数
        dynamics_params = DynamicsParameters(
            masses=[5.0 + i * 2.0 for i in range(n_joints)],  # 递增质量
            centers_of_mass=[[0.0, 0.0, 0.1 + i * 0.05] for i in range(n_joints)],  # 递增质心高度
            inertias=[[[1.0 + i * 0.5, 0.0, 0.0], [0.0, 1.0 + i * 0.5, 0.0], [0.0, 0.0, 1.0 + i * 0.5]] for i in range(n_joints)],  # 递增惯量
            friction_coeffs=[0.1 + i * 0.02 for i in range(n_joints)],  # 递增摩擦系数
            gravity=[0.0, 0.0, -9.81]
        )
        
        # 创建测试用运动学限制
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[np.pi * (1.0 + i * 0.1) for i in range(n_joints)],  # 递增位置限制
            min_joint_positions=[-np.pi * (1.0 + i * 0.1) for i in range(n_joints)],
            max_joint_velocities=[2.0 + i * 0.5 for i in range(n_joints)],  # 递增速度限制
            max_joint_accelerations=[10.0 + i * 2.0 for i in range(n_joints)],  # 递增加速度限制
            max_joint_jerks=[50.0 + i * 10.0 for i in range(n_joints)],  # 递增加加速度限制
            max_joint_torques=[100.0 + i * 20.0 for i in range(n_joints)]  # 递增力矩限制
        )
        
        return cls(
            name=name,
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
    
    def update_payload(self, payload: PayloadInfo) -> None:
        """
        更新负载信息
        
        Args:
            payload: 新的负载信息
        """
        self.current_payload = payload
    
    def get_joint_limits(self) -> tuple[Vector, Vector]:
        """
        获取关节位置限制
        
        Returns:
            (最小位置, 最大位置)
        """
        return (
            np.array(self.kinodynamic_limits.min_joint_positions),
            np.array(self.kinodynamic_limits.max_joint_positions)
        )
    
    def get_velocity_limits(self) -> Vector:
        """
        获取关节速度限制
        
        Returns:
            最大关节速度
        """
        return np.array(self.kinodynamic_limits.max_joint_velocities)
    
    def get_acceleration_limits(self) -> Vector:
        """
        获取关节加速度限制
        
        Returns:
            最大关节加速度
        """
        return np.array(self.kinodynamic_limits.max_joint_accelerations)
    
    def get_torque_limits(self) -> Vector:
        """
        获取关节力矩限制
        
        Returns:
            最大关节力矩
        """
        return np.array(self.kinodynamic_limits.max_joint_torques)
    
    def is_configuration_valid(self, joint_positions: Vector) -> bool:
        """
        检查关节配置是否在有效范围内
        
        Args:
            joint_positions: 关节位置
        
        Returns:
            是否有效
        """
        min_pos, max_pos = self.get_joint_limits()
        return np.all(joint_positions >= min_pos) and np.all(joint_positions <= max_pos)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型转换为字典格式
        
        Returns:
            模型字典表示
        """
        return {
            "name": self.name,
            "n_joints": self.n_joints,
            "dynamics_params": self.dynamics_params.dict(),
            "kinodynamic_limits": self.kinodynamic_limits.dict(),
            "urdf_path": self.urdf_path,
            "mjcf_path": self.mjcf_path,
            "current_payload": self.current_payload.dict() if self.current_payload else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobotModel':
        """
        从字典创建模型实例
        
        Args:
            data: 模型字典数据
        
        Returns:
            RobotModel实例
        """
        dynamics_params = DynamicsParameters(**data["dynamics_params"])
        kinodynamic_limits = KinodynamicLimits(**data["kinodynamic_limits"])
        
        model = cls(
            name=data["name"],
            n_joints=data["n_joints"],
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits,
            urdf_path=data.get("urdf_path"),
            mjcf_path=data.get("mjcf_path")
        )
        
        if data.get("current_payload"):
            model.current_payload = PayloadInfo(**data["current_payload"])
        
        model.metadata = data.get("metadata", {})
        
        return model


@dataclass
class RobotStateHistory:
    """
    机器人状态历史记录
    
    用于存储和管理机器人的历史状态数据，支持轨迹分析和性能评估。
    """
    states: List[RobotState] = field(default_factory=list)
    max_history_length: int = 1000
    
    def add_state(self, state: RobotState) -> None:
        """
        添加新的状态记录
        
        Args:
            state: 机器人状态
        """
        self.states.append(state)
        
        # 限制历史记录长度
        if len(self.states) > self.max_history_length:
            self.states.pop(0)
    
    def get_latest_state(self) -> Optional[RobotState]:
        """
        获取最新状态
        
        Returns:
            最新的机器人状态，如果没有记录则返回None
        """
        return self.states[-1] if self.states else None
    
    def get_state_at_time(self, timestamp: float, tolerance: float = 0.001) -> Optional[RobotState]:
        """
        获取指定时间的状态
        
        Args:
            timestamp: 目标时间戳
            tolerance: 时间容差
        
        Returns:
            最接近指定时间的状态
        """
        if not self.states:
            return None
        
        # 找到最接近的时间戳
        closest_state = min(
            self.states,
            key=lambda s: abs(s.timestamp - timestamp)
        )
        
        if abs(closest_state.timestamp - timestamp) <= tolerance:
            return closest_state
        
        return None
    
    def get_time_range(self) -> tuple[float, float]:
        """
        获取时间范围
        
        Returns:
            (开始时间, 结束时间)
        """
        if not self.states:
            return (0.0, 0.0)
        
        timestamps = [state.timestamp for state in self.states]
        return (min(timestamps), max(timestamps))
    
    def clear(self) -> None:
        """清空历史记录"""
        self.states.clear()
    
    def get_position_trajectory(self) -> np.ndarray:
        """
        获取位置轨迹
        
        Returns:
            位置轨迹矩阵 (n_states, n_joints)
        """
        if not self.states:
            return np.array([])
        
        return np.array([state.joint_positions for state in self.states])
    
    def get_velocity_trajectory(self) -> np.ndarray:
        """
        获取速度轨迹
        
        Returns:
            速度轨迹矩阵 (n_states, n_joints)
        """
        if not self.states:
            return np.array([])
        
        return np.array([state.joint_velocities for state in self.states])
    
    def compute_tracking_error(self, reference_trajectory: List[TrajectoryPoint]) -> np.ndarray:
        """
        计算轨迹跟踪误差
        
        Args:
            reference_trajectory: 参考轨迹
        
        Returns:
            跟踪误差数组
        """
        if not self.states or not reference_trajectory:
            return np.array([])
        
        errors = []
        for state in self.states:
            # 找到最接近的参考点
            closest_ref = min(
                reference_trajectory,
                key=lambda p: abs(p.time - state.timestamp)
            )
            
            # 计算位置误差
            error = np.linalg.norm(state.joint_positions - closest_ref.position)
            errors.append(error)
        
        return np.array(errors)