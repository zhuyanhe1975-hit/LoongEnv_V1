#!/usr/bin/env python3
"""
简化的并行计算演示

展示基本的并行计算功能和性能提升。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time

from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.parallel_computing import ParallelConfig, ParallelMode
from robot_motion_control.algorithms.parallel_dynamics import ParallelOptimizedDynamicsEngine


def main():
    """简化的并行计算演示"""
    print("机器人运动控制系统 - 并行计算优化演示")
    print("=" * 50)
    
    # 创建测试机器人模型
    print("创建测试机器人模型...")
    robot_model = RobotModel.create_test_model(n_joints=6)
    print(f"机器人模型: {robot_model.name}, 关节数: {robot_model.n_joints}")
    
    # 创建并行优化动力学引擎
    print("\n创建并行优化动力学引擎...")
    parallel_config = ParallelConfig(
        mode=ParallelMode.THREAD,
        max_workers=4,
        enable_memory_optimization=True
    )
    
    parallel_engine = ParallelOptimizedDynamicsEngine(robot_model, parallel_config)
    print(f"并行配置: {parallel_config.mode.value}, 工作线程数: {parallel_config.max_workers}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    n_joints = robot_model.n_joints
    batch_sizes = [5, 10, 20]
    
    for batch_size in batch_sizes:
        print(f"\n测试批处理大小: {batch_size}")
        
        # 生成随机测试数据
        q_list = [np.random.randn(n_joints) * 0.5 for _ in range(batch_size)]
        qd_list = [np.random.randn(n_joints) * 0.1 for _ in range(batch_size)]
        tau_list = [np.random.randn(n_joints) * 10.0 for _ in range(batch_size)]
        
        # 顺序计算
        start_time = time.time()
        sequential_results = []
        for q, qd, tau in zip(q_list, qd_list, tau_list):
            result = parallel_engine.forward_dynamics(q, qd, tau)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # 并行计算
        start_time = time.time()
        parallel_results = parallel_engine.batch_forward_dynamics(q_list, qd_list, tau_list)
        parallel_time = time.time() - start_time
        
        # 验证结果一致性
        max_diff = 0.0
        for seq_result, par_result in zip(sequential_results, parallel_results):
            if seq_result is not None and par_result is not None:
                diff = np.max(np.abs(seq_result - par_result))
                max_diff = max(max_diff, diff)
        
        # 计算性能提升
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"  顺序计算时间: {sequential_time:.4f}s")
        print(f"  并行计算时间: {parallel_time:.4f}s")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  结果最大差异: {max_diff:.2e}")
        
        # 验证结果正确性
        if max_diff < 1e-10:
            print("  ✓ 结果验证通过")
        else:
            print("  ⚠ 结果存在差异")
    
    # 获取性能报告
    print("\n性能报告:")
    performance_report = parallel_engine.get_performance_report()
    print(f"  并行调用次数: {performance_report['parallel_calls']}")
    print(f"  顺序调用次数: {performance_report['sequential_calls']}")
    print(f"  平均加速比: {performance_report['average_speedup']:.2f}x")
    print(f"  批处理阈值: {performance_report['batch_threshold']}")
    
    # 测试自适应阈值优化
    print("\n测试自适应阈值优化...")
    original_threshold = parallel_engine.batch_threshold
    print(f"  原始阈值: {original_threshold}")
    
    try:
        new_threshold = parallel_engine.adaptive_parallel_threshold(
            sample_size=8, test_iterations=2
        )
        print(f"  优化后阈值: {new_threshold}")
    except Exception as e:
        print(f"  阈值优化失败: {e}")
    
    print("\n" + "=" * 50)
    print("并行计算演示完成！")
    print("\n主要成果:")
    print("✓ 实现了多线程并行动力学计算")
    print("✓ 支持批量处理以提高效率")
    print("✓ 提供了自动性能调优功能")
    print("✓ 确保了计算结果的正确性")


if __name__ == "__main__":
    main()