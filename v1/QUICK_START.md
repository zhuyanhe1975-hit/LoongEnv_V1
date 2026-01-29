# LoongEnv 快速开始指南

## 🚀 发送方：创建部署包

```bash
# 一键创建最小化压缩包（约20-30MB）
bash create_deployment_package.sh
```

生成文件：`LoongEnv_deploy_YYYYMMDD_HHMMSS.tar.gz`

---

## 📦 接收方：部署系统

### 1. 解压文件
```bash
tar -xzf LoongEnv_deploy_*.tar.gz
cd LoongEnv
```

### 2. 安装Python依赖（约2-3分钟）
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. 安装前端依赖（约3-5分钟）
```bash
cd ui
npm install
cd ..
```

### 4. 启动系统
```bash
python tools/start_system.py
```

### 5. 访问系统
- 前端：http://localhost:5173
- 后端：http://localhost:5006

---

## ✅ 验证部署

```bash
# 运行基础测试
source venv/bin/activate
pytest tests/test_integration_basic.py -v
```

---

## 📋 系统要求

- **Python**: 3.8 或更高
- **Node.js**: 16.0 或更高
- **操作系统**: Linux, macOS, Windows
- **磁盘空间**: 至少 1GB（安装依赖后）

---

## 🔧 常见问题

### 问题1: pip install 失败
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
brew install python@3.11
```

### 问题2: npm install 失败
```bash
# 清除缓存重试
npm cache clean --force
npm install
```

### 问题3: 端口被占用
```bash
# 修改端口（编辑 ui/backend_api.py）
# 将 port=5006 改为其他端口
```

---

## 📚 更多信息

- 完整部署指南：`DEPLOYMENT_GUIDE.md`
- 项目文档：`README.md`
- API文档：`docs/`
