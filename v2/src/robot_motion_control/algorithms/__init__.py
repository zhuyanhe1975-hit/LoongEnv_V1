"""
算法模块 - 实现机器人运动控制的核心算法

包含以下主要算法组件：
- 动力学引擎：正向/逆向动力学计算
- 轨迹规划：TOPP算法和S型插补
- 路径控制：高精度路径跟踪控制
- 抑振算法：输入整形和柔性补偿
- 负载识别：在线负载参数识别
- 安全监控：算法异常检测和保护
- 碰撞检测：基于距离的碰撞检测和避让
"""

from .dynamics import DynamicsEngine
from .trajectory_planning import TrajectoryPlanner
from .path_control import PathController
from .vibration_suppression import VibrationSuppressor
from .collision_detection import CollisionMonitor, CollisionDetector, CollisionAvoidance

__all__ = [
    "DynamicsEngine",
    "TrajectoryPlanner", 
    "PathController",
    "VibrationSuppressor",
    "CollisionMonitor",
    "CollisionDetector", 
    "CollisionAvoidance",
]