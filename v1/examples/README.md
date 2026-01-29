# 示例代码

本目录包含各种功能模块的使用示例。

## 示例列表

### 基础使用

#### basic_usage.py
基本的机器人控制示例。

**功能**:
- 创建机器人模型
- 初始化控制器
- 规划简单轨迹
- 执行运动控制

**运行**:
```bash
python examples/basic_usage.py
```

### 动力学分析

#### er15_1400_pinocchio_demo.py
使用Pinocchio进行动力学分析。

**功能**:
- 正向运动学
- 逆向动力学
- 雅可比矩阵计算
- 质量矩阵计算

**运行**:
```bash
python examples/er15_1400_pinocchio_demo.py
```

**输出**:
- `er15_1400_dynamics_analysis.png`: 动力学分析图表

#### gravity_friction_demo.py
重力和摩擦力建模示例。

**功能**:
- 重力补偿
- 摩擦力模型
- 力矩分析

**运行**:
```bash
python examples/gravity_friction_demo.py
```

**输出**:
- `gravity_friction_analysis.png`: 重力摩擦分析图

### 轨迹规划

#### trajectory_interpolation_demo.py
轨迹插值算法示例。

**功能**:
- S7曲线插值
- 速度和加速度规划
- 轨迹平滑

**运行**:
```bash
python examples/trajectory_interpolation_demo.py
```

**输出**:
- `trajectory_results.png`: 轨迹规划结果

#### topp_algorithm_demo.py
TOPP时间最优规划示例。

**功能**:
- 时间最优路径参数化
- 速度限制处理
- 加速度限制处理

**运行**:
```bash
python examples/topp_algorithm_demo.py
```

**输出**:
- `topp_results.png`: TOPP算法结果

### 路径控制

#### path_controller_demo.py
路径控制器示例。

**功能**:
- PID控制
- 前馈控制
- 轨迹跟踪

**运行**:
```bash
python examples/path_controller_demo.py
```

**输出**:
- `path_controller_results.png`: 控制器性能图

### 参数优化

#### parameter_tuning_demo.py
参数调优示例。

**功能**:
- 差分进化优化
- 性能评估
- 报告生成

**运行**:
```bash
python examples/parameter_tuning_demo.py
```

**输出**:
- 调优报告（在 `tuning_reports/` 目录）

### 高级功能

#### flexible_joint_compensation_demo.py
柔性关节补偿示例。

**功能**:
- 关节柔性建模
- 补偿控制
- 性能对比

**运行**:
```bash
python examples/flexible_joint_compensation_demo.py
```

#### collision_detection_demo.py
碰撞检测示例。

**功能**:
- 自碰撞检测
- 环境碰撞检测
- 避障规划

**运行**:
```bash
python examples/collision_detection_demo.py
```

### 并行计算

#### parallel_computing_demo.py
并行计算示例。

**功能**:
- 多进程并行
- 性能对比
- 加速比分析

**运行**:
```bash
python examples/parallel_computing_demo.py
```

#### simple_parallel_demo.py
简单的并行计算示例。

**功能**:
- 基础并行任务
- 进程池使用
- 结果聚合

**运行**:
```bash
python examples/simple_parallel_demo.py
```

### 综合测试

#### comprehensive_test_demo.py
综合测试场景示例。

**功能**:
- 多种测试场景
- 性能基准测试
- 结果分析

**运行**:
```bash
python examples/comprehensive_test_demo.py
```

## 使用指南

### 运行示例

1. 确保虚拟环境已激活
2. 安装所有依赖
3. 运行示例脚本

```bash
source venv/bin/activate
python examples/<example_name>.py
```

### 修改示例

所有示例都可以作为模板进行修改：

1. 复制示例文件
2. 修改参数和配置
3. 运行并查看结果

### 创建新示例

```python
#!/usr/bin/env python3
"""
示例名称 - 简短描述

详细说明示例的功能和用途。
"""

from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.controller import RobotMotionController

def main():
    # 创建机器人模型
    robot = RobotModel(name="ER15-1400", n_joints=6)
    
    # 创建控制器
    controller = RobotMotionController(robot)
    
    # 你的代码...
    
    print("示例完成！")

if __name__ == '__main__':
    main()
```

## 输出文件

示例生成的图片和数据文件：

- `*.png`: 可视化图表
- `*.json`: 数据文件
- `tuning_reports/`: 调优报告

## 依赖

所有示例需要以下依赖：

- NumPy
- Matplotlib
- Pinocchio
- SciPy

## 常见问题

### Q: 示例运行失败？

A: 检查：
1. 虚拟环境是否激活
2. 依赖是否完整安装
3. 模型文件是否存在

### Q: 图片不显示？

A: 确认：
1. Matplotlib后端配置
2. 显示环境是否支持
3. 使用 `plt.savefig()` 保存图片

### Q: 性能较慢？

A: 尝试：
1. 减少数据点数量
2. 使用并行计算
3. 优化算法参数

## 相关文档

- [源代码文档](../src/README.md)
- [测试文档](../tests/README.md)
- [项目README](../README.md)

## 更新日期

2026-01-29
