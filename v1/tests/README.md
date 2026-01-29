# 测试目录

本目录包含项目的所有测试代码，使用pytest框架。

## 测试结构

```
tests/
├── conftest.py                      # pytest配置和fixtures
├── test_collision_detection.py      # 碰撞检测测试
├── test_comprehensive_scenarios.py  # 综合场景测试
├── test_core_types.py              # 核心类型测试
├── test_extreme_conditions.py      # 极端条件测试
├── test_flexible_joint_compensation.py  # 柔性关节补偿测试
├── test_gravity_friction_modeling.py    # 重力摩擦建模测试
├── test_integration_basic.py       # 基础集成测试
├── test_parallel_computing.py      # 并行计算测试
├── test_parameter_tuning.py        # 参数调优测试
├── test_path_controller.py         # 路径控制器测试
├── test_performance_benchmarks.py  # 性能基准测试
├── test_pinocchio_integration.py   # Pinocchio集成测试
├── test_topp_algorithm.py          # TOPP算法测试
├── test_trajectory_interpolation.py # 轨迹插值测试
└── run_comprehensive_tests.py      # 综合测试运行脚本
```

## 测试类型

### 单元测试

测试单个函数或类的功能：

- `test_core_types.py`: 数据类型验证
- `test_collision_detection.py`: 碰撞检测算法
- `test_trajectory_interpolation.py`: 轨迹插值算法

### 集成测试

测试多个模块的协同工作：

- `test_integration_basic.py`: 基础功能集成
- `test_pinocchio_integration.py`: Pinocchio库集成
- `test_path_controller.py`: 控制器集成

### 性能测试

测试系统性能和效率：

- `test_performance_benchmarks.py`: 性能基准测试
- `test_parallel_computing.py`: 并行计算性能

### 场景测试

测试实际应用场景：

- `test_comprehensive_scenarios.py`: 综合应用场景
- `test_extreme_conditions.py`: 极端条件处理

### 属性测试

使用Hypothesis进行基于属性的测试：

- 自动生成测试数据
- 验证算法的通用性质
- 发现边界情况

## 运行测试

### 运行所有测试

```bash
pytest
```

### 运行特定测试文件

```bash
pytest tests/test_parameter_tuning.py
```

### 运行特定测试函数

```bash
pytest tests/test_trajectory_interpolation.py::test_s7_interpolation
```

### 运行带标记的测试

```bash
# 只运行快速测试
pytest -m "not slow"

# 只运行集成测试
pytest -m integration
```

### 查看测试覆盖率

```bash
pytest --cov=src/robot_motion_control --cov-report=html
```

### 运行综合测试

```bash
python tests/run_comprehensive_tests.py
```

## 测试配置

### pytest.ini

项目根目录的 `pytest.ini` 配置文件：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: 标记慢速测试
    integration: 标记集成测试
    performance: 标记性能测试
```

### conftest.py

包含共享的fixtures和配置：

- `robot_model`: 机器人模型fixture
- `controller`: 控制器fixture
- `test_trajectory`: 测试轨迹fixture

## 编写测试

### 测试命名规范

- 测试文件: `test_<module_name>.py`
- 测试类: `Test<ClassName>`
- 测试函数: `test_<function_name>`

### 测试结构

```python
import pytest
from robot_motion_control.algorithms import trajectory_planning

def test_s7_interpolation():
    """测试S7曲线插值"""
    # Arrange - 准备测试数据
    waypoints = [...]
    
    # Act - 执行测试
    trajectory = trajectory_planning.interpolate_s7(waypoints)
    
    # Assert - 验证结果
    assert len(trajectory) > 0
    assert trajectory[0].position == waypoints[0].position
```

### 使用Fixtures

```python
def test_controller_with_fixture(controller):
    """使用fixture的测试"""
    result = controller.plan_trajectory(waypoints)
    assert result is not None
```

### 参数化测试

```python
@pytest.mark.parametrize("input,expected", [
    (0.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.5),
])
def test_with_parameters(input, expected):
    result = function(input)
    assert result == expected
```

## 测试最佳实践

1. **独立性**: 每个测试应该独立运行
2. **可重复性**: 测试结果应该一致
3. **清晰性**: 测试意图应该明确
4. **快速性**: 单元测试应该快速执行
5. **覆盖性**: 覆盖正常和异常情况

## 持续集成

测试在以下情况自动运行：

- 提交代码时
- 创建Pull Request时
- 合并到主分支前

## 测试报告

测试结果保存在：

- `comprehensive_test_report.json`: JSON格式报告
- `htmlcov/`: HTML覆盖率报告

## 调试测试

### 打印调试信息

```bash
pytest -s  # 显示print输出
```

### 进入调试器

```bash
pytest --pdb  # 失败时进入pdb
```

### 只运行失败的测试

```bash
pytest --lf  # last failed
```

## 更新日期

2026-01-29
