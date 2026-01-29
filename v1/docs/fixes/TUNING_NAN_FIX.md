# 参数调优NaN错误修复

## 问题描述

在运行参数调优时，前端显示错误：
```
调优状态: 错误"Unexpected token 'N', ..."ormance": NaN, "... is not valid JSON
```

这表明调优结果中包含 `NaN`（Not a Number）值，导致 JSON 序列化失败。

## 根本原因

1. **优化器可能返回NaN**：
   - 差分进化算法在某些情况下可能产生 NaN 结果
   - 目标函数评估可能产生 NaN（例如除以零、sqrt负数等）

2. **JSON不支持NaN**：
   - JavaScript的 `JSON.parse()` 无法解析 `NaN`、`Infinity` 等特殊值
   - Python的 `json.dumps()` 会将 NaN 序列化为 `NaN`（不是有效的JSON）

3. **NumPy类型问题**：
   - NumPy的 `np.nan` 和 `np.inf` 在JSON序列化时会产生问题
   - NumPy数组需要转换为Python列表

## 解决方案

### 1. 在优化结果中添加NaN检查

在 `parameter_tuning.py` 的 `_optimize_parameters` 方法中：

```python
# 检查并处理NaN值
if np.isnan(best_score) or np.isinf(best_score):
    warnings.warn(f"优化结果包含无效值 (NaN或Inf): {best_score}，使用默认值")
    best_score = 1e6  # 使用大的惩罚值
    success = False
    message = f"优化结果无效: {message}"

return TuningResult(
    optimal_parameters=optimal_params,
    best_performance=float(best_score),  # 确保是Python float
    ...
)
```

### 2. 在评估函数中添加NaN检查

在所有评估函数（`_evaluate_control_performance`、`_evaluate_trajectory_performance`、`_evaluate_vibration_performance`）的返回处：

```python
# 计算平均分数
final_score = total_score / max(scenario_count, 1)

# 检查并处理NaN或Inf
if np.isnan(final_score) or np.isinf(final_score):
    warnings.warn(f"性能评估产生无效值: {final_score}，返回惩罚值")
    return 1e6

return float(final_score)
```

### 3. 在后端API中添加安全转换

在 `backend_api.py` 中添加 `safe_convert` 函数：

```python
def safe_convert(value):
    """安全转换数值，处理NaN和Inf"""
    if isinstance(value, (np.ndarray, list)):
        return [safe_convert(v) for v in value]
    elif isinstance(value, dict):
        return {k: safe_convert(v) for k, v in value.items()}
    elif isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None  # 转换为null
        return float(value)
    elif isinstance(value, (int, np.integer)):
        return int(value)
    else:
        return value
```

然后在构建调优结果时使用：

```python
tuning_status["results"] = {
    "success": True,
    "overallImprovement": safe_convert(report.overall_performance_improvement),
    "results": {
        param_type.value: {
            "success": result.success,
            "bestPerformance": safe_convert(result.best_performance),
            "computationTime": safe_convert(result.computation_time),
            "optimalParameters": {
                k: safe_convert(v)
                for k, v in result.optimal_parameters.items()
            }
        }
        for param_type, result in results.items()
    },
    ...
}
```

## 验证

创建了测试脚本 `tools/diagnostics/test_nan_handling.py` 来验证：

```bash
python3 tools/diagnostics/test_nan_handling.py
```

测试结果：
- ✓ `safe_convert` 正确将 NaN 转换为 `null`
- ✓ `safe_convert` 正确将 Inf 转换为 `null`
- ✓ 复杂嵌套对象可以正确序列化为JSON

## 预期效果

修复后：
1. 即使优化过程产生 NaN，也会被转换为有效值（1e6 或 null）
2. JSON 序列化不会失败
3. 前端可以正确接收和显示调优结果
4. NaN 值会在前端显示为 `null` 或 "N/A"

## 相关文件

- `src/robot_motion_control/algorithms/parameter_tuning.py` - 添加NaN检查
- `ui/backend_api.py` - 添加safe_convert函数
- `tools/diagnostics/test_nan_handling.py` - NaN处理测试

## 日期

2026-01-29
