#!/usr/bin/env python3
"""
ER15-1400机械臂Pinocchio动力学库集成演示

本示例展示如何使用Pinocchio动力学库与ER15-1400机械臂模型进行动力学计算。
包括正向动力学、逆向动力学、雅可比矩阵计算等功能的演示。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robot_motion_control.core.models import RobotModel
from robot_motion_control.algorithms.dynamics import DynamicsEngine
from robot_motion_control.core.types import PayloadInfo


def main():
    """主演示函数"""
    print("=" * 60)
    print("ER15-1400机械臂Pinocchio动力学库集成演示")
    print("=" * 60)
    
    # 1. 加载ER15-1400机器人模型
    print("\n1. 加载ER15-1400机器人模型...")
    
    mjcf_path = "models/ER15-1400-mjcf/er15-1400.mjcf.xml"
    if not Path(mjcf_path).exists():
        print(f"错误: MJCF文件不存在: {mjcf_path}")
        print("请确保ER15-1400模型文件已正确放置在models目录中")
        return
    
    try:
        robot_model = RobotModel.create_er15_1400(mjcf_path)
        print(f"✓ 成功加载机器人模型: {robot_model.name}")
        print(f"  - 关节数量: {robot_model.n_joints}")
        print(f"  - MJCF文件: {robot_model.mjcf_path}")
        
        # 显示关节名称
        if "joint_names" in robot_model.metadata:
            joint_names = robot_model.metadata["joint_names"]
            print(f"  - 关节名称: {', '.join(joint_names)}")
        
    except Exception as e:
        print(f"✗ 加载机器人模型失败: {e}")
        return
    
    # 2. 创建动力学引擎
    print("\n2. 创建动力学引擎...")
    
    try:
        dynamics_engine = DynamicsEngine(robot_model)
        print("✓ 动力学引擎创建成功")
        
        # 检查Pinocchio是否可用
        pinocchio_model = dynamics_engine.pinocchio_model
        if pinocchio_model is not None:
            print("✓ Pinocchio库集成成功")
            print(f"  - 配置空间维度: {pinocchio_model.nq}")
            print(f"  - 速度空间维度: {pinocchio_model.nv}")
        else:
            print("⚠ 使用简化动力学实现（Pinocchio不可用）")
            
    except Exception as e:
        print(f"✗ 动力学引擎创建失败: {e}")
        return
    
    # 3. 动力学计算演示
    print("\n3. 动力学计算演示...")
    
    # 定义测试关节配置
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 关节位置 [rad]
    qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 关节速度 [rad/s]
    qdd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 关节加速度 [rad/s²]
    
    print(f"测试关节配置:")
    print(f"  - 位置 (rad): {q}")
    print(f"  - 速度 (rad/s): {qd}")
    print(f"  - 加速度 (rad/s²): {qdd}")
    
    try:
        # 3.1 逆向动力学计算
        print("\n3.1 逆向动力学计算...")
        tau = dynamics_engine.inverse_dynamics(q, qd, qdd)
        print(f"✓ 所需关节力矩 (Nm): {tau}")
        
        # 3.2 正向动力学计算
        print("\n3.2 正向动力学计算...")
        qdd_computed = dynamics_engine.forward_dynamics(q, qd, tau)
        print(f"✓ 计算得到的关节加速度 (rad/s²): {qdd_computed}")
        
        # 验证一致性
        error = np.linalg.norm(qdd - qdd_computed)
        print(f"✓ 动力学一致性误差: {error:.6f}")
        
        # 3.3 雅可比矩阵计算
        print("\n3.3 雅可比矩阵计算...")
        jacobian = dynamics_engine.jacobian(q)
        print(f"✓ 雅可比矩阵形状: {jacobian.shape}")
        print(f"✓ 雅可比矩阵条件数: {np.linalg.cond(jacobian):.2f}")
        
        # 3.4 重力补偿计算
        print("\n3.4 重力补偿计算...")
        g = dynamics_engine.gravity_compensation(q)
        print(f"✓ 重力补偿力矩 (Nm): {g}")
        
        # 3.5 质量矩阵计算
        print("\n3.5 质量矩阵计算...")
        M = dynamics_engine.compute_mass_matrix(q)
        print(f"✓ 质量矩阵形状: {M.shape}")
        print(f"✓ 质量矩阵条件数: {np.linalg.cond(M):.2f}")
        
        # 3.6 科里奥利矩阵计算
        print("\n3.6 科里奥利矩阵计算...")
        C = dynamics_engine.compute_coriolis_matrix(q, qd)
        print(f"✓ 科里奥利矩阵形状: {C.shape}")
        
    except Exception as e:
        print(f"✗ 动力学计算失败: {e}")
        return
    
    # 4. 负载更新演示
    print("\n4. 负载更新演示...")
    
    try:
        # 创建负载信息
        payload = PayloadInfo(
            mass=5.0,  # 5kg负载
            center_of_mass=[0.0, 0.0, 0.1],  # 质心位置
            inertia=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],  # 惯量矩阵
            identification_confidence=0.95
        )
        
        print(f"添加负载:")
        print(f"  - 质量: {payload.mass} kg")
        print(f"  - 质心: {payload.center_of_mass}")
        print(f"  - 识别置信度: {payload.identification_confidence}")
        
        # 更新负载
        dynamics_engine.update_payload(payload)
        print("✓ 负载更新成功")
        
        # 重新计算重力补偿（应该包含负载影响）
        g_with_payload = dynamics_engine.gravity_compensation(q)
        print(f"✓ 带负载的重力补偿力矩 (Nm): {g_with_payload}")
        
        # 比较负载前后的差异
        g_diff = g_with_payload - g
        print(f"✓ 负载引起的重力补偿变化 (Nm): {g_diff}")
        
    except Exception as e:
        print(f"✗ 负载更新失败: {e}")
    
    # 5. 性能测试
    print("\n5. 性能测试...")
    
    try:
        import time
        
        # 测试计算性能
        n_iterations = 1000
        
        # 正向动力学性能测试
        start_time = time.time()
        for _ in range(n_iterations):
            dynamics_engine.forward_dynamics(q, qd, tau)
        fd_time = (time.time() - start_time) / n_iterations * 1000  # ms
        
        # 逆向动力学性能测试
        start_time = time.time()
        for _ in range(n_iterations):
            dynamics_engine.inverse_dynamics(q, qd, qdd)
        id_time = (time.time() - start_time) / n_iterations * 1000  # ms
        
        # 雅可比矩阵性能测试
        start_time = time.time()
        for _ in range(n_iterations):
            dynamics_engine.jacobian(q)
        jac_time = (time.time() - start_time) / n_iterations * 1000  # ms
        
        print(f"✓ 性能测试结果 ({n_iterations}次迭代平均):")
        print(f"  - 正向动力学: {fd_time:.3f} ms")
        print(f"  - 逆向动力学: {id_time:.3f} ms")
        print(f"  - 雅可比矩阵: {jac_time:.3f} ms")
        
    except Exception as e:
        print(f"⚠ 性能测试失败: {e}")
    
    # 6. 轨迹动力学分析
    print("\n6. 轨迹动力学分析...")
    
    try:
        # 生成简单的正弦轨迹
        t = np.linspace(0, 2*np.pi, 100)
        amplitude = 0.5
        
        # 计算轨迹上每个点的动力学
        positions = []
        velocities = []
        accelerations = []
        torques = []
        
        for i, time_step in enumerate(t):
            # 正弦轨迹
            q_traj = amplitude * np.sin(time_step) * np.ones(6)
            qd_traj = amplitude * np.cos(time_step) * np.ones(6)
            qdd_traj = -amplitude * np.sin(time_step) * np.ones(6)
            
            # 计算所需力矩
            tau_traj = dynamics_engine.inverse_dynamics(q_traj, qd_traj, qdd_traj)
            
            positions.append(q_traj)
            velocities.append(qd_traj)
            accelerations.append(qdd_traj)
            torques.append(tau_traj)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        torques = np.array(torques)
        
        print("✓ 轨迹动力学分析完成")
        print(f"  - 轨迹点数: {len(t)}")
        print(f"  - 最大关节力矩: {np.max(np.abs(torques)):.2f} Nm")
        print(f"  - 平均关节力矩: {np.mean(np.abs(torques)):.2f} Nm")
        
        # 绘制结果
        create_dynamics_plots(t, positions, velocities, accelerations, torques)
        
    except Exception as e:
        print(f"✗ 轨迹动力学分析失败: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def create_dynamics_plots(t, positions, velocities, accelerations, torques):
    """创建动力学分析图表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ER15-1400机械臂动力学分析', fontsize=14)
        
        # 位置图
        axes[0, 0].plot(t, positions[:, 0], label='关节1')
        axes[0, 0].plot(t, positions[:, 1], label='关节2')
        axes[0, 0].plot(t, positions[:, 2], label='关节3')
        axes[0, 0].set_title('关节位置')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('位置 (rad)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 速度图
        axes[0, 1].plot(t, velocities[:, 0], label='关节1')
        axes[0, 1].plot(t, velocities[:, 1], label='关节2')
        axes[0, 1].plot(t, velocities[:, 2], label='关节3')
        axes[0, 1].set_title('关节速度')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('速度 (rad/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 加速度图
        axes[1, 0].plot(t, accelerations[:, 0], label='关节1')
        axes[1, 0].plot(t, accelerations[:, 1], label='关节2')
        axes[1, 0].plot(t, accelerations[:, 2], label='关节3')
        axes[1, 0].set_title('关节加速度')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('加速度 (rad/s²)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 力矩图
        axes[1, 1].plot(t, torques[:, 0], label='关节1')
        axes[1, 1].plot(t, torques[:, 1], label='关节2')
        axes[1, 1].plot(t, torques[:, 2], label='关节3')
        axes[1, 1].set_title('关节力矩')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('力矩 (Nm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = "examples/er15_1400_dynamics_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 动力学分析图表已保存: {output_path}")
        
        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            pass  # 在非交互环境中忽略显示错误
            
    except Exception as e:
        print(f"⚠ 图表创建失败: {e}")


if __name__ == "__main__":
    main()