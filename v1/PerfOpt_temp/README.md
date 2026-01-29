# PerfOpt - 机器人参数性能优化工具

精简的多目标参数优化工具，专注于ER15-1400机器人的控制参数优化。

## 特性

- 多目标优化（跟踪精度、稳定时间、超调、能效、振动抑制）
- 支持多种优化算法（差分进化、Basin Hopping、梯度下降）
- 基于Pinocchio的精确动力学模型
- 自动生成优化报告和可视化图表

## 项目结构

```
PerfOpt/
├── perfopt/              # 核心优化模块
│   ├── __init__.py
│   ├── optimizer.py      # 参数优化器
│   ├── evaluator.py      # 性能评估器
│   ├── dynamics.py       # 动力学引擎
│   ├── controller.py     # 控制器
│   ├── models.py         # 数据模型
│   └── reporter.py       # 报告生成器
├── examples/             # 示例脚本
│   └── optimize_er15.py  # ER15优化示例
├── models/               # 机器人模型文件
│   └── ER15-1400.urdf
├── reports/              # 优化报告输出
├── requirements.txt      # 依赖包
└── README.md
```

## 安装

```bash
cd /home/yhzhu/LoongEnv/PerfOpt
pip install -r requirements.txt
```

## 快速开始

```python
from perfopt import ParameterOptimizer, RobotModel

# 加载机器人模型
robot = RobotModel.from_urdf("models/ER15-1400.urdf")

# 创建优化器
optimizer = ParameterOptimizer(robot)

# 运行优化
results = optimizer.optimize(
    trajectory=test_trajectory,
    max_iterations=50,
    method="differential_evolution"
)

# 查看结果
print(f"最优性能: {results.best_performance}")
print(f"优化参数: {results.optimal_parameters}")
```

## 命令行使用

```bash
# 运行ER15优化示例
python examples/optimize_er15.py

# 自定义参数
python examples/optimize_er15.py --iterations 100 --method basin_hopping
```

## 优化算法

- `differential_evolution`: 差分进化（默认，适合全局优化）
- `basin_hopping`: Basin Hopping（适合复杂地形）
- `gradient_descent`: 梯度下降（适合局部优化）

## 性能指标

优化器会综合考虑以下指标：

1. **跟踪精度** (40%): RMS跟踪误差
2. **稳定时间** (20%): 达到稳态所需时间
3. **超调量** (15%): 最大超调百分比
4. **能效** (10%): 归一化能耗
5. **振动抑制** (10%): 平均振动水平
6. **安全裕度** (5%): 安全约束满足度

## 输出

优化完成后会生成：

- `reports/optimization_report_YYYYMMDD_HHMMSS.json` - JSON格式报告
- `reports/optimization_report_YYYYMMDD_HHMMSS.md` - Markdown格式报告
- `reports/optimization_history_YYYYMMDD_HHMMSS.png` - 优化历史曲线
- `reports/performance_comparison_YYYYMMDD_HHMMSS.png` - 性能对比图

## 依赖

- Python >= 3.8
- NumPy
- SciPy
- Pinocchio (机器人动力学)
- Matplotlib (可视化)

## 许可

MIT License
