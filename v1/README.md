# LoongEnv - 机器人运动控制系统

一个完整的机器人运动控制系统，支持轨迹规划、参数优化、实时监控和可视化。

## 项目概述

本项目为ER15-1400六轴工业机器人提供了完整的运动控制解决方案，包括：

- 🎯 **轨迹规划**: S7曲线插值、TOPP时间最优规划
- ⚙️ **参数优化**: 差分进化算法自动调优PID参数
- 📊 **性能监控**: 实时跟踪误差、能耗、振动等指标
- 🎨 **可视化界面**: React + TypeScript构建的现代化Web UI
- 🔧 **动力学仿真**: 基于Pinocchio的高精度动力学计算

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 18+
- 虚拟环境（已配置共享venv）

### 安装依赖

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装Python依赖
pip install -r requirements.txt

# 安装前端依赖
cd ui
npm install
```

### 启动系统

```bash
# 方式1: 使用启动脚本（推荐）
python tools/start_system.py

# 方式2: 手动启动
# 终端1 - 启动后端
python ui/backend_api.py

# 终端2 - 启动前端
cd ui
npm run dev
```

访问 http://localhost:5173 查看Web界面。

## 项目结构

```
.
├── src/                    # 核心源代码
│   └── robot_motion_control/
│       ├── algorithms/     # 算法实现（轨迹规划、参数优化等）
│       ├── core/          # 核心模块（控制器、模型等）
│       ├── simulation/    # 仿真环境
│       └── utils/         # 工具函数
├── ui/                    # Web前端界面
│   ├── src/              # React源代码
│   ├── public/           # 静态资源
│   └── backend_api.py    # Flask后端API
├── tests/                # 测试代码
├── examples/             # 示例代码
├── models/               # 机器人模型文件（URDF、STL）
├── docs/                 # 项目文档
│   ├── implementation/   # 实现文档
│   ├── ui/              # UI设计文档
│   ├── fixes/           # 问题修复记录
│   ├── reports/         # 功能报告
│   └── images/          # 文档图片
├── tools/               # 工具脚本
│   └── diagnostics/     # 诊断工具
├── scripts/             # 性能分析脚本
├── reports/             # 性能报告
└── tuning_reports/      # 参数调优报告
```

## 核心功能

### 1. 轨迹规划

- S7曲线插值：平滑的加加速度连续轨迹
- TOPP算法：时间最优路径参数化
- 碰撞检测：实时避障功能

### 2. 参数优化

- 差分进化算法优化PID参数
- 7个性能指标综合评估
- 自动生成优化报告和可视化图表

### 3. Web界面

- **仪表盘**: 系统状态总览、3D机器人可视化
- **轨迹规划**: 路径点编辑、轨迹预览
- **实时监控**: 关节状态、性能指标
- **参数调优**: 算法配置、优化进度、性能对比
- **系统设置**: 主题切换、参数配置

### 4. 性能分析

- 并行计算加速（多进程）
- 实时性能监控
- 详细的性能报告生成

## 技术栈

### 后端
- Python 3.12
- Flask (Web框架)
- NumPy (数值计算)
- Pinocchio (机器人动力学)
- SciPy (优化算法)

### 前端
- React 18
- TypeScript
- Material-UI
- Three.js (3D可视化)
- Redux Toolkit (状态管理)

## 文档

- [虚拟环境配置](VENV_SHARED_SETUP.md) - 共享venv设置说明
- [实现文档](docs/implementation/) - 各功能模块的实现细节
- [UI文档](docs/ui/) - 界面设计和实现
- [修复记录](docs/fixes/) - 问题修复历史
- [功能报告](docs/reports/) - 功能开发报告

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_parameter_tuning.py

# 运行诊断工具
python tools/diagnostics/diagnose_tuning_crash.py
```

## 性能

- 轨迹规划: ~10ms (1000个点)
- 参数优化: ~2-5分钟 (50次迭代)
- 实时控制: 1000Hz控制频率
- 并行加速: 2-4倍性能提升

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目仓库: https://github.com/zhuyanhe1975-hit/LoongEnv_V1
- 作者: zhuyanhe1975-hit

## 更新日志

### v1.0.0 (2026-01-29)
- ✅ 完整的机器人运动控制系统
- ✅ 参数优化和性能对比功能
- ✅ 现代化Web界面
- ✅ 共享虚拟环境配置
- ✅ 项目结构优化和文档完善
