# 共享虚拟环境配置说明

## 概述

为了节省磁盘空间，v1、v2等不同版本现在共用同一个Python虚拟环境。

## 目录结构

```
/home/yhzhu/LoongEnv/
├── venv-shared/          # 共享的Python虚拟环境 (770MB)
├── v0/                   # 版本0（无venv）
├── v1/
│   └── venv -> ../venv-shared  # 符号链接
├── v2/
│   └── venv -> ../venv-shared  # 符号链接
└── .gitignore           # 已添加venv-shared/到忽略列表
```

## 空间节省

- **之前**: v1/venv (770MB) + v2/venv (770MB) = 1540MB
- **现在**: venv-shared (770MB) = 770MB
- **节省**: 770MB (50%)

## 使用方法

在任何版本目录下激活虚拟环境：

```bash
# 在v1目录
cd /home/yhzhu/LoongEnv/v1
source venv/bin/activate

# 在v2目录
cd /home/yhzhu/LoongEnv/v2
source venv/bin/activate
```

两个版本都会使用同一个共享的虚拟环境。

## 安装新包

在任何版本中安装的包都会对所有版本生效：

```bash
# 在v1中安装
cd /home/yhzhu/LoongEnv/v1
source venv/bin/activate
pip install some-package

# 在v2中也能使用
cd /home/yhzhu/LoongEnv/v2
source venv/bin/activate
python -c "import some_package"  # 可以正常导入
```

## 注意事项

1. **依赖冲突**: 如果不同版本需要不同版本的包，可能会产生冲突
2. **独立环境**: 如果某个版本需要独立的环境，可以删除符号链接并创建新的venv
3. **Git忽略**: venv-shared已添加到.gitignore，不会被提交到仓库

## 恢复独立环境

如果需要为某个版本创建独立的虚拟环境：

```bash
# 删除符号链接
cd /home/yhzhu/LoongEnv/v1
rm venv

# 创建新的虚拟环境
python3 -m venv venv

# 激活并安装依赖
source venv/bin/activate
pip install -r requirements.txt
```

## 验证配置

检查符号链接是否正确：

```bash
ls -lh /home/yhzhu/LoongEnv/v*/venv
```

应该显示：
```
lrwxrwxrwx 1 yhzhu yhzhu 14 Jan 29 08:21 /home/yhzhu/LoongEnv/v1/venv -> ../venv-shared
lrwxrwxrwx 1 yhzhu yhzhu 14 Jan 29 08:22 /home/yhzhu/LoongEnv/v2/venv -> ../venv-shared
```

## 已安装的包

共享环境中已安装的主要包：
- Flask (Web框架)
- NumPy (数值计算)
- Pinocchio (机器人动力学)
- Hypothesis (属性测试)
- pytest (测试框架)
- 其他依赖包...

查看完整列表：
```bash
source venv/bin/activate
pip list
```

## 更新日期

2026-01-29
