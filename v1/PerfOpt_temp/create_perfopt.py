#!/usr/bin/env python3
"""
自动创建PerfOpt项目的Python脚本
"""

import os
import shutil
from pathlib import Path

SOURCE_DIR = Path("/home/yhzhu/LoongEnv/v1")
TARGET_DIR = Path("/home/yhzhu/LoongEnv/PerfOpt")

def create_directory_structure():
    """创建目录结构"""
    print("创建目录结构...")
    dirs = [
        TARGET_DIR,
        TARGET_DIR / "perfopt",
        TARGET_DIR / "examples",
        TARGET_DIR / "models",
        TARGET_DIR / "reports",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")


def copy_core_files():
    """复制核心文件"""
    print("\n复制核心文件...")
    
    # 复制文件映射
    files_to_copy = [
        # (源文件, 目标文件)
        (SOURCE_DIR / "src/robot_motion_control/algorithms/dynamics.py", 
         TARGET_DIR / "perfopt/dynamics.py"),
        (SOURCE_DIR / "src/robot_motion_control/algorithms/path_control.py", 
         TARGET_DIR / "perfopt/controller.py"),
        (SOURCE_DIR / "src/robot_motion_control/algorithms/parameter_tuning.py", 
         TARGET_DIR / "perfopt/optimizer.py"),
    ]
    
    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")
        else:
            print(f"  ✗ {src} 不存在")


def copy_model_files():
    """复制模型文件"""
    print("\n复制模型文件...")
    
    model_dir = SOURCE_DIR / "models"
    if model_dir.exists():
        # 复制URDF文件
        urdf_file = model_dir / "ER15-1400.urdf"
        if urdf_file.exists():
            shutil.copy2(urdf_file, TARGET_DIR / "models/")
            print(f"  ✓ ER15-1400.urdf")
        
        # 复制STL文件
        for stl_file in model_dir.glob("*.STL"):
            shutil.copy2(stl_file, TARGET_DIR / "models/")
            print(f"  ✓ {stl_file.name}")


def create_init_file():
    """创建__init__.py"""
    print("\n创建__init__.py...")
    
    init_content = '''"""PerfOpt - 机器人参数性能优化工具"""

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
'''
    
    init_file = TARGET_DIR / "perfopt/__init__.py"
    init_file.write_text(init_content)
    print(f"  ✓ {init_file}")


def create_models_file():
    """创建精简的models.py"""
    print("\n创建models.py...")
    
    models_content = '''"""数据模型"""
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
'''
    
    models_file = TARGET_DIR / "perfopt/models.py"
    models_file.write_text(models_content)
    print(f"  ✓ {models_file}")


def copy_documentation():
    """复制文档"""
    print("\n复制文档...")
    
    docs = [
        (SOURCE_DIR / "PerfOpt_temp/README.md", TARGET_DIR / "README.md"),
        (SOURCE_DIR / "PerfOpt_temp/requirements.txt", TARGET_DIR / "requirements.txt"),
    ]
    
    for src, dst in docs:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {dst.name}")


def copy_examples():
    """复制示例"""
    print("\n复制示例...")
    
    example_file = SOURCE_DIR / "PerfOpt_temp/examples/optimize_er15.py"
    if example_file.exists():
        shutil.copy2(example_file, TARGET_DIR / "examples/")
        print(f"  ✓ optimize_er15.py")


def main():
    """主函数"""
    print("=" * 60)
    print("创建PerfOpt项目")
    print("=" * 60)
    print()
    
    try:
        create_directory_structure()
        create_models_file()
        create_init_file()
        copy_core_files()
        copy_model_files()
        copy_documentation()
        copy_examples()
        
        print()
        print("=" * 60)
        print("✓ PerfOpt项目创建完成！")
        print("=" * 60)
        print()
        print("下一步:")
        print(f"1. cd {TARGET_DIR}")
        print("2. pip install -r requirements.txt")
        print("3. python examples/optimize_er15.py")
        print()
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
