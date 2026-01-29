"""
核心模块 - 定义系统的基础数据结构、类型和主控制器
"""

from .controller import RobotMotionController
from .models import RobotModel, RobotState, TrajectoryPoint
from .types import (
    ControlCommand,
    DynamicsParameters,
    KinodynamicLimits,
    PayloadInfo,
    Trajectory,
)

__all__ = [
    "RobotMotionController",
    "RobotModel",
    "RobotState", 
    "TrajectoryPoint",
    "ControlCommand",
    "DynamicsParameters",
    "KinodynamicLimits",
    "PayloadInfo",
    "Trajectory",
]