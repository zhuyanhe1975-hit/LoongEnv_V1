#!/usr/bin/env python3
"""
修复PerfOpt项目的导入依赖问题
"""

import re
from pathlib import Path

PERFOPT_DIR = Path("/home/yhzhu/LoongEnv/PerfOpt")

def fix_optimizer_imports():
    """修复optimizer.py的导入"""
    print("修复 optimizer.py...")
    
    file_path = PERFOPT_DIR / "perfopt/optimizer.py"
    content = file_path.read_text()
    
    # 移除字体配置导入
    content = re.sub(
        r'# 导入字体配置.*?label_map = \{\}',
        '# 字体配置已移除（精简版不需要）',
        content,
        flags=re.DOTALL
    )
    
    # 修复导入语句
    content = content.replace(
        'from ..core.types import (',
        'from .models import ('
    )
    content = content.replace(
        'from ..core.models import RobotModel',
        'from .models import RobotModel'
    )
    
    file_path.write_text(content)
    print("  ✓ optimizer.py 已修复")


def fix_controller_imports():
    """修复controller.py的导入"""
    print("修复 controller.py...")
    
    file_path = PERFOPT_DIR / "perfopt/controller.py"
    content = file_path.read_text()
    
    # 修复导入语句
    content = content.replace(
        'from ..core.models import RobotModel',
        'from .models import RobotModel'
    )
    content = content.replace(
        'from ..core.types import RobotState, TrajectoryPoint, ControlCommand, Vector, AlgorithmError',
        'from .models import RobotState, TrajectoryPoint, ControlCommand, Vector, AlgorithmError'
    )
    
    file_path.write_text(content)
    print("  ✓ controller.py 已修复")


def fix_dynamics_imports():
    """修复dynamics.py的导入"""
    print("修复 dynamics.py...")
    
    file_path = PERFOPT_DIR / "perfopt/dynamics.py"
    content = file_path.read_text()
    
    # 修复导入语句
    content = content.replace(
        'from ..core.models import RobotModel',
        'from .models import RobotModel'
    )
    content = content.replace(
        'from ..core.types import Vector, Matrix, PayloadInfo, AlgorithmError',
        'from .models import Vector, Matrix, PayloadInfo, AlgorithmError'
    )
    
    # 移除MJCF解析器导入（精简版不需要）
    content = re.sub(
        r'from \.\.core\.mjcf_parser import MJCFParser',
        '# MJCF解析器已移除（精简版不需要）',
        content
    )
    
    file_path.write_text(content)
    print("  ✓ dynamics.py 已修复")


def enhance_models():
    """增强models.py，添加缺失的类型"""
    print("增强 models.py...")
    
    file_path = PERFOPT_DIR / "perfopt/models.py"
    content = file_path.read_text()
    
    # 在文件末尾添加缺失的类型定义
    additional_types = '''

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
'''
    
    content += additional_types
    file_path.write_text(content)
    print("  ✓ models.py 已增强")


def update_init():
    """更新__init__.py"""
    print("更新 __init__.py...")
    
    init_content = '''"""PerfOpt - 机器人参数性能优化工具"""

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
from .optimizer import ParameterOptimizer
from .dynamics import DynamicsEngine
from .controller import PathController

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
    "ParameterOptimizer",
    "DynamicsEngine",
    "PathController"
]
'''
    
    init_file = PERFOPT_DIR / "perfopt/__init__.py"
    init_file.write_text(init_content)
    print("  ✓ __init__.py 已更新")


def main():
    """主函数"""
    print("=" * 60)
    print("修复PerfOpt项目")
    print("=" * 60)
    print()
    
    try:
        fix_optimizer_imports()
        fix_controller_imports()
        fix_dynamics_imports()
        enhance_models()
        update_init()
        
        print()
        print("=" * 60)
        print("✓ PerfOpt项目修复完成！")
        print("=" * 60)
        print()
        print("测试导入:")
        print("  python -c \"import sys; sys.path.insert(0, '/home/yhzhu/LoongEnv/PerfOpt'); from perfopt import ParameterOptimizer; print('✓ 导入成功')\"")
        print()
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
