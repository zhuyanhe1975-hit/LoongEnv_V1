"""PerfOpt - 机器人参数性能优化工具"""

from .models import (
    RobotModel, 
    RobotState, 
    TrajectoryPoint, 
    Trajectory,
    ControlCommand,
    PayloadInfo,
    PerformanceMetrics,
    DynamicsParameters,
    KinodynamicLimits,
    AlgorithmError
)
from .optimizer import ParameterTuner, TuningReportGenerator, OptimizationConfig, PerformanceWeights
from .dynamics import DynamicsEngine
from .controller import PathController

# 为了兼容性，提供别名
ParameterOptimizer = ParameterTuner

__version__ = "1.0.0"
__all__ = [
    "RobotModel",
    "RobotState", 
    "TrajectoryPoint",
    "Trajectory",
    "ControlCommand",
    "PayloadInfo",
    "PerformanceMetrics",
    "DynamicsParameters",
    "KinodynamicLimits",
    "AlgorithmError",
    "ParameterTuner",
    "ParameterOptimizer",
    "TuningReportGenerator",
    "OptimizationConfig",
    "PerformanceWeights",
    "DynamicsEngine",
    "PathController"
]
