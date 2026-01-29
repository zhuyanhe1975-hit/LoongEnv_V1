#!/bin/bash
# 自动设置PerfOpt项目的脚本

set -e

SOURCE_DIR="/home/yhzhu/LoongEnv/v1"
TARGET_DIR="/home/yhzhu/LoongEnv/PerfOpt"

echo "=========================================="
echo "设置PerfOpt项目"
echo "=========================================="

# 创建目标目录结构
echo "创建目录结构..."
mkdir -p "$TARGET_DIR"/{perfopt,examples,models,reports}

# 复制核心文件
echo "复制核心模块..."

# 1. 复制models.py（精简版）
cat > "$TARGET_DIR/perfopt/models.py" << 'EOF'
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
EOF

# 2. 复制dynamics.py（从原项目提取）
cp "$SOURCE_DIR/src/robot_motion_control/algorithms/dynamics.py" "$TARGET_DIR/perfopt/dynamics.py"

# 3. 复制controller.py（精简版）
cp "$SOURCE_DIR/src/robot_motion_control/algorithms/path_control.py" "$TARGET_DIR/perfopt/controller.py"

# 4. 复制optimizer.py（从parameter_tuning.py提取）
cp "$SOURCE_DIR/src/robot_motion_control/algorithms/parameter_tuning.py" "$TARGET_DIR/perfopt/optimizer.py"

# 5. 创建__init__.py
cat > "$TARGET_DIR/perfopt/__init__.py" << 'EOF'
"""PerfOpt - 机器人参数性能优化工具"""

from .models import RobotModel, RobotState, TrajectoryPoint, Trajectory
from .optimizer import ParameterOptimizer
from .dynamics import DynamicsEngine

__version__ = "1.0.0"
__all__ = [
    "RobotModel",
    "RobotState", 
    "TrajectoryPoint",
    "Trajectory",
    "ParameterOptimizer",
    "DynamicsEngine"
]
EOF

# 6. 复制ER15模型文件
if [ -f "$SOURCE_DIR/models/ER15-1400.urdf" ]; then
    cp "$SOURCE_DIR/models/ER15-1400.urdf" "$TARGET_DIR/models/"
    echo "✓ 复制ER15-1400.urdf"
fi

# 7. 复制STL文件
if [ -d "$SOURCE_DIR/models" ]; then
    cp "$SOURCE_DIR/models"/*.STL "$TARGET_DIR/models/" 2>/dev/null || true
    echo "✓ 复制STL文件"
fi

# 8. 复制README和requirements
cp "$SOURCE_DIR/PerfOpt_temp/README.md" "$TARGET_DIR/"
cp "$SOURCE_DIR/PerfOpt_temp/requirements.txt" "$TARGET_DIR/"

echo ""
echo "=========================================="
echo "✓ PerfOpt项目设置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. cd $TARGET_DIR"
echo "2. pip install -r requirements.txt"
echo "3. python examples/optimize_er15.py"
echo ""
