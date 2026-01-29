# 工具脚本目录

本目录包含项目的各种工具脚本和诊断工具。

## 目录结构

```
tools/
├── diagnostics/              # 诊断工具
│   ├── diagnose_tuning_crash.py   # 参数调优崩溃诊断
│   └── test_tuning_restart.py     # 调优重启测试
├── start_backend_safe.sh     # 安全启动后端脚本
└── start_system.py          # 系统启动脚本
```

## 工具说明

### 系统启动

#### start_system.py

一键启动整个系统（后端+前端）。

**使用方法**:
```bash
python tools/start_system.py
```

**功能**:
- 自动检查依赖
- 启动Flask后端服务
- 启动React前端开发服务器
- 提供统一的日志输出

#### start_backend_safe.sh

安全启动后端服务，带自动重启和日志记录。

**使用方法**:
```bash
./tools/start_backend_safe.sh
```

**功能**:
- 自动记录日志到文件
- 崩溃时自动重启（最多5次）
- 提供详细的时间戳和退出码
- 便于调试和问题追踪

### 诊断工具

#### diagnostics/diagnose_tuning_crash.py

诊断参数调优功能的崩溃问题。

**使用方法**:
```bash
python tools/diagnostics/diagnose_tuning_crash.py
```

**功能**:
- 测试调优器初始化
- 验证第一次调优
- 验证状态重置
- 验证第二次调优
- 生成诊断报告

**适用场景**:
- 调优功能出现500错误
- 第二次调优失败
- 进程崩溃问题

#### diagnostics/test_tuning_restart.py

测试调优重启功能的API测试脚本。

**使用方法**:
```bash
python tools/diagnostics/test_tuning_restart.py
```

**功能**:
- 通过API测试调优启动
- 验证调优状态轮询
- 测试连续多次调优
- 验证结果正确性

**适用场景**:
- 验证调优修复效果
- API集成测试
- 回归测试

## 使用场景

### 日常开发

```bash
# 启动开发环境
python tools/start_system.py
```

### 生产部署

```bash
# 使用安全启动脚本
./tools/start_backend_safe.sh
```

### 问题诊断

```bash
# 诊断调优问题
python tools/diagnostics/diagnose_tuning_crash.py

# 测试调优重启
python tools/diagnostics/test_tuning_restart.py
```

## 添加新工具

### 创建新工具脚本

1. 在适当的子目录创建脚本
2. 添加清晰的文档字符串
3. 提供使用示例
4. 更新本README

### 脚本模板

```python
#!/usr/bin/env python3
"""
工具名称 - 简短描述

详细说明工具的功能和用途。

使用方法:
    python tools/your_tool.py [options]

示例:
    python tools/your_tool.py --option value
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='工具描述')
    parser.add_argument('--option', help='选项说明')
    args = parser.parse_args()
    
    # 工具逻辑
    print("工具执行中...")

if __name__ == '__main__':
    main()
```

## 工具开发指南

### 最佳实践

1. **清晰的文档**: 每个工具都应有详细的文档字符串
2. **错误处理**: 优雅地处理错误情况
3. **日志记录**: 提供详细的日志输出
4. **参数验证**: 验证输入参数的有效性
5. **退出码**: 使用标准退出码（0=成功，非0=失败）

### 日志规范

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("工具启动")
logger.warning("警告信息")
logger.error("错误信息")
```

### 配置管理

```python
import json
from pathlib import Path

def load_config(config_file='config.json'):
    """加载配置文件"""
    config_path = Path(__file__).parent / config_file
    with open(config_path) as f:
        return json.load(f)
```

## 常见问题

### Q: 启动脚本失败怎么办？

A: 检查以下几点：
1. 虚拟环境是否激活
2. 依赖是否完整安装
3. 端口是否被占用
4. 查看日志文件获取详细错误

### Q: 诊断工具报错？

A: 确保：
1. 后端服务正在运行
2. 网络连接正常
3. API端点可访问
4. 查看详细的错误堆栈

### Q: 如何添加自定义工具？

A: 参考"添加新工具"部分，创建脚本并更新文档。

## 相关文档

- [项目README](../README.md)
- [修复记录](../docs/fixes/)
- [诊断指南](../docs/fixes/TUNING_RESTART_FIX.md)

## 更新日期

2026-01-29
