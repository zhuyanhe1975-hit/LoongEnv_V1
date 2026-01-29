# 源代码目录

本目录包含机器人运动控制系统的核心源代码。

## 目录结构

```
src/robot_motion_control/
├── algorithms/         # 算法实现
│   ├── collision_detection.py      # 碰撞检测
│   ├── dynamics.py                 # 动力学计算
│   ├── parallel_dynamics.py        # 并行动力学
│   ├── parallel_path_control.py    # 并行路径控制
│   ├── parallel_trajectory_planning.py  # 并行轨迹规划
│   ├── parameter_tuning.py         # 参数调优
│   ├── path_control.py             # 路径控制
│   ├── payload_identification.py   # 负载识别
│   ├── safety.py                   # 安全监控
│   ├── trajectory_planning.py      # 轨迹规划
│   └── vibration_suppression.py    # 振动抑制
├── core/              # 核心模块
│   ├── controller.py              # 运动控制器
│   ├── mjcf_parser.py            # MJCF解析器
│   ├── models.py                 # 机器人模型
│   ├── parallel_computing.py     # 并行计算框架
│   └── types.py                  # 数据类型定义
├── simulation/        # 仿真环境
│   ├── digital_model.py          # 数字孪生模型
│   └── environment.py            # 仿真环境
└── utils/            # 工具函数
    └── font_config.py            # 字体配置
```

## 模块说明

### algorithms/ - 算法实现

#### 轨迹规划
- `trajectory_planning.py`: S7曲线插值、TOPP时间最优规划
- `parallel_trajectory_planning.py`: 并行轨迹规划加速

#### 路径控制
- `path_control.py`: PID控制、前馈控制
- `parallel_path_control.py`: 并行路径控制

#### 动力学
- `dynamics.py`: 正向/逆向动力学、雅可比矩阵
- `parallel_dynamics.py`: 并行动力学计算

#### 参数优化
- `parameter_tuning.py`: 差分进化算法、性能评估、报告生成

#### 其他算法
- `collision_detection.py`: 碰撞检测和避障
- `vibration_suppression.py`: 振动抑制控制
- `payload_identification.py`: 负载参数识别
- `safety.py`: 安全监控和限制

### core/ - 核心模块

#### 控制器
- `controller.py`: 主控制器，集成所有功能模块

#### 模型
- `models.py`: 机器人模型定义（ER15-1400）
- `mjcf_parser.py`: MuJoCo MJCF格式解析

#### 类型定义
- `types.py`: 数据结构定义（状态、轨迹点、路径点等）

#### 并行计算
- `parallel_computing.py`: 多进程并行计算框架

### simulation/ - 仿真环境

- `environment.py`: 仿真环境管理
- `digital_model.py`: 数字孪生模型

### utils/ - 工具函数

- `font_config.py`: 中文字体配置（用于图表）

## 使用示例

### 基本使用

```python
from robot_motion_control.core.models import RobotModel
from robot_motion_control.core.controller import RobotMotionController

# 创建机器人模型
robot = RobotModel(name="ER15-1400", n_joints=6)

# 创建控制器
controller = RobotMotionController(robot)

# 规划轨迹
waypoints = [...]
trajectory = controller.plan_trajectory(waypoints)

# 执行控制
for point in trajectory:
    control_cmd = controller.execute_trajectory_point(point)
```

### 参数优化

```python
from robot_motion_control.algorithms.parameter_tuning import ParameterTuner

# 创建调优器
tuner = ParameterTuner(robot)

# 执行优化
results = tuner.comprehensive_tuning(
    reference_trajectory,
    test_scenarios,
    parameter_types
)
```

### 并行计算

```python
from robot_motion_control.core.parallel_computing import ParallelComputing

# 创建并行计算实例
parallel = ParallelComputing(n_workers=4)

# 并行执行任务
results = parallel.map(compute_function, data_list)
```

## 开发指南

### 添加新算法

1. 在 `algorithms/` 目录创建新文件
2. 实现算法类和函数
3. 在 `__init__.py` 中导出
4. 添加单元测试
5. 更新文档

### 修改核心模块

1. 确保向后兼容
2. 更新类型定义
3. 运行完整测试套件
4. 更新API文档

### 性能优化

1. 使用 `parallel_computing.py` 进行并行化
2. 使用NumPy向量化操作
3. 避免不必要的内存分配
4. 使用性能分析工具

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_trajectory_planning.py

# 性能测试
python scripts/performance_analysis.py
```

## 依赖

- NumPy: 数值计算
- SciPy: 优化算法
- Pinocchio: 机器人动力学
- Matplotlib: 可视化

## 更新日期

2026-01-29
