"""
机器人运动控制系统 (Robot Motion Control System)

一套对标ABB QuickMove、TrueMove并加入StableMove功能的机器人控制算法库。
实现从运动学控制向全动力学控制的跨越，提供高精度、高速度、低振动的机器人运动控制能力。
"""

__version__ = "0.1.0"
__author__ = "Robot Motion Control Team"
__email__ = "team@robotics.com"

# 核心组件导入
from .core.controller import RobotMotionController
from .core.models import RobotModel, RobotState, TrajectoryPoint
from .core.types import (
    ControlCommand,
    DynamicsParameters,
    KinodynamicLimits,
    PayloadInfo,
    Trajectory,
)

# 主要算法模块
from .algorithms.dynamics import DynamicsEngine
from .algorithms.path_control import PathController
from .algorithms.trajectory_planning import TrajectoryPlanner
from .algorithms.vibration_suppression import VibrationSuppressor

# 仿真和测试
from .simulation.digital_model import RobotDigitalModel
from .simulation.environment import SimulationEnvironment

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    # 核心组件
    "RobotMotionController",
    "RobotModel",
    "RobotState",
    "TrajectoryPoint",
    # 数据类型
    "ControlCommand",
    "DynamicsParameters",
    "KinodynamicLimits",
    "PayloadInfo",
    "Trajectory",
    # 算法模块
    "DynamicsEngine",
    "PathController",
    "TrajectoryPlanner",
    "VibrationSuppressor",
    # 仿真模块
    "RobotDigitalModel",
    "SimulationEnvironment",
]