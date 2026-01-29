#!/usr/bin/env python3
"""
测试Pinocchio动力学在参数调优中的使用
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.robot_motion_control.core.models import RobotModel
from src.robot_motion_control.algorithms.dynamics import DynamicsEngine

def test_pinocchio_forward_dynamics():
    """测试Pinocchio正向动力学"""
    print("=" * 60)
    print("测试Pinocchio正向动力学")
    print("=" * 60)
    
    # 创建机器人模型
    robot_model = RobotModel.create_test_model(n_joints=6)
    
    # 创建动力学引擎
    dynamics = DynamicsEngine(robot_model)
    
    print(f"\nPinocchio模型可用: {dynamics.pinocchio_model is not None}")
    
    if dynamics.pinocchio_model is None:
        print("⚠️  Pinocchio模型未初始化，将使用简化模型")
        return False
    
    # 测试不同力矩下的加速度
    q = np.zeros(6)
    v = np.zeros(6)
    
    test_cases = [
        ("零力矩", np.zeros(6)),
        ("小力矩", np.ones(6) * 1.0),
        ("中等力矩", np.ones(6) * 10.0),
        ("大力矩", np.ones(6) * 50.0),
    ]
    
    print("\n测试不同力矩下的加速度响应：\n")
    
    accelerations = []
    for name, tau in test_cases:
        try:
            qdd = dynamics.forward_dynamics(q, v, tau)
            accelerations.append(qdd)
            print(f"{name:12s}: qdd = {qdd}")
        except Exception as e:
            print(f"{name:12s}: 错误 - {e}")
            return False
    
    # 检查加速度是否随力矩变化
    print("\n分析结果：")
    
    acc_norms = [np.linalg.norm(a) for a in accelerations]
    print(f"加速度范数: {acc_norms}")
    
    acc_range = max(acc_norms) - min(acc_norms)
    print(f"加速度范围: {acc_range:.6f}")
    
    if acc_range < 0.01:
        print("⚠️  警告：加速度变化太小，Pinocchio可能未正常工作")
        return False
    else:
        print("✓ Pinocchio正向动力学工作正常")
        return True


if __name__ == "__main__":
    success = test_pinocchio_forward_dynamics()
    sys.exit(0 if success else 1)
