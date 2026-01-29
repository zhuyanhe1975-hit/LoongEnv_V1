# 机器人运动控制系统 (Robot Motion Control System)

一套对标ABB QuickMove（速度）、TrueMove（精度）并加入StableMove（抑振）的国产机器人控制算法库。

## 核心特性

- **高精度路径控制** (TrueMove): ≤0.1mm轨迹跟踪精度
- **时间最优轨迹规划** (QuickMove): TOPP算法实现最快运动
- **主动振动抑制** (StableMove): 输入整形技术消除振动
- **全动力学建模**: 基于Pinocchio的完整动力学计算
- **数字化仿真**: 高保真机器人数字模型验证

## 系统架构

```
应用层: 算法测试界面 | 参数配置 | 标定算法
控制算法层: 轨迹规划 | 路径控制 | 抑振算法 | 安全监控
计算引擎层: 动力学引擎 | 运动学引擎 | 负载识别
仿真层: 机器人数字模型 | 物理仿真 | 传感器模拟
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/robotics/robot-motion-control.git
cd robot-motion-control

# 安装依赖
pip install -e ".[dev]"
```

### 基本使用

```python
from robot_motion_control import RobotMotionController
from robot_motion_control.models import RobotModel

# 加载机器人模型
robot_model = RobotModel.from_urdf("path/to/robot.urdf")

# 创建控制器
controller = RobotMotionController(robot_model)

# 规划轨迹
trajectory = controller.plan_trajectory(waypoints, constraints)

# 执行控制
for point in trajectory:
    control_command = controller.compute_control(point, current_state)
    # 发送控制指令到机器人
```

## 开发

### 运行测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest -m unit

# 运行属性测试
pytest -m property

# 运行性能测试
pytest -m performance
```

### 代码质量

```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# 代码检查
flake8 src/ tests/
```

## 文档

- [API文档](docs/api.md)
- [算法设计](docs/algorithms.md)
- [使用指南](docs/usage.md)
- [开发指南](docs/development.md)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。