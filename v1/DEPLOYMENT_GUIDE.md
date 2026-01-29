# LoongEnv 部署指南

## 快速打包（推荐方式）

### 方式1：使用部署脚本（最简单）

```bash
# 创建最小化压缩包
bash create_deployment_package.sh
```

这将创建 `LoongEnv_deploy.tar.gz`（约20-30MB），包含所有必需文件。

---

## 方式2：手动打包

### 1. 创建排除列表

```bash
cat > .tarignore << 'EOF'
# Python缓存和虚拟环境
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/

# Node.js
ui/node_modules/
ui/dist/

# IDE和编辑器
.vscode/
.idea/
.kiro/

# Git
.git/

# 临时文件和日志
*.log
*.tmp
simulation_data/
logs/
results/
result/

# 调优报告（保留README，删除具体报告）
tuning_reports/*.png
tuning_reports/*.json
tuning_reports/*.md
!tuning_reports/README.md

# 测试和性能分析
.pytest_cache/
.hypothesis/
*.prof
*.pstats
htmlcov/
.coverage

# 临时开发文件
PerfOpt_temp/
../PerfOpt/
EOF
```

### 2. 创建压缩包

```bash
# 使用tar排除不需要的文件
tar --exclude-from=.tarignore -czf LoongEnv_deploy.tar.gz .
```

---

## 接收方部署步骤

### 1. 解压文件

```bash
tar -xzf LoongEnv_deploy.tar.gz
cd LoongEnv
```

### 2. 安装Python依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
cd ui
npm install
cd ..
```

### 4. 启动系统

```bash
# 方式1：使用启动脚本（推荐）
python tools/start_system.py

# 方式2：手动启动
# 终端1：启动后端
cd ui
python backend_api.py

# 终端2：启动前端
cd ui
npm run dev
```

### 5. 访问系统

- 前端界面：http://localhost:5173
- 后端API：http://localhost:5006

---

## 压缩包内容说明

### 必需文件（约20-30MB）
- `src/` - 核心源代码
- `ui/src/` - 前端源代码
- `ui/public/` - 静态资源（包含STL模型）
- `models/` - 机器人模型文件
- `examples/` - 示例代码
- `tests/` - 测试文件
- `docs/` - 文档
- `requirements.txt` - Python依赖
- `ui/package.json` - Node.js依赖
- `README.md` - 项目说明

### 排除的文件（约760MB）
- `venv/` - Python虚拟环境（需重新创建）
- `ui/node_modules/` - Node.js依赖（需重新安装）
- `ui/dist/` - 前端构建产物
- `__pycache__/` - Python缓存
- `.git/` - Git仓库
- `tuning_reports/*.png` - 调优报告图片
- `.kiro/` - IDE配置

---

## 常见问题

### Q: 为什么不包含 node_modules？
A: node_modules 占用752MB，但可以通过 `npm install` 快速重建。

### Q: 为什么不包含 venv？
A: 虚拟环境依赖系统路径，无法跨机器使用，需要重新创建。

### Q: 如何验证部署成功？
A: 运行测试：
```bash
pytest tests/test_integration_basic.py -v
```

### Q: 缺少依赖怎么办？
A: 检查Python版本（需要3.8+）和系统依赖：
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
brew install python@3.11
```

---

## 最小化技巧

如果需要进一步减小体积：

1. **删除示例图片**（约5MB）：
```bash
rm examples/*.png
rm docs/images/*.png
```

2. **删除测试文件**（约2MB）：
```bash
rm -rf tests/
```

3. **只保留必需的STL模型**（约10MB）：
```bash
# 保留在 ui/public/models/ 和 models/ 中
```

4. **删除文档**（约1MB）：
```bash
rm -rf docs/
```

**注意**：删除这些文件会影响功能完整性，仅在极度需要减小体积时使用。

---

## 生产环境部署

如果是生产环境，建议：

1. **构建前端**：
```bash
cd ui
npm run build
```

2. **使用生产服务器**：
```bash
# 使用 gunicorn 运行后端
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5006 ui.backend_api:app

# 使用 nginx 服务前端静态文件
```

3. **配置环境变量**：
```bash
export FLASK_ENV=production
export PYTHONPATH=/path/to/LoongEnv
```
