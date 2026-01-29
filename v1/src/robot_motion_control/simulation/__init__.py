"""
仿真模块 - 提供机器人数字化仿真和验证环境

包含以下主要组件：
- 数字化机器人模型：高保真物理仿真
- 仿真环境：完整的仿真执行环境
- 传感器模拟：噪声和延迟模拟
- 可视化接口：仿真结果可视化
"""

from .digital_model import RobotDigitalModel
from .environment import SimulationEnvironment

__all__ = [
    "RobotDigitalModel",
    "SimulationEnvironment",
]