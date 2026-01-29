#!/bin/bash

# LoongEnv 部署包创建脚本
# 创建最小化的可运行压缩包

set -e

echo "======================================"
echo "  LoongEnv 部署包创建工具"
echo "======================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置
PACKAGE_NAME="LoongEnv_deploy"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${PACKAGE_NAME}_${TIMESTAMP}.tar.gz"

echo -e "${YELLOW}步骤 1/4: 检查当前目录...${NC}"
if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 目录检查通过${NC}"
echo ""

echo -e "${YELLOW}步骤 2/4: 创建临时排除列表...${NC}"
cat > .tarignore.tmp << 'EOF'
# Python缓存和虚拟环境
venv/
env/
ENV/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
.eggs/
build/
dist/
*.egg

# Node.js
ui/node_modules/
ui/dist/

# IDE和编辑器
.vscode/
.idea/
.kiro/
*.swp
*.swo
*~

# Git
.git/
.gitignore

# 临时文件和日志
*.log
*.tmp
*.temp
simulation_data/
logs/
results/
result/

# 调优报告（保留README）
tuning_reports/*.png
tuning_reports/*.json
tuning_reports/*.md

# 测试缓存
.pytest_cache/
.hypothesis/
*.prof
*.pstats
htmlcov/
.coverage
.coverage.*
.tox/
.nox/

# 临时开发文件
PerfOpt_temp/
../PerfOpt/

# 打包文件本身
*.tar.gz
*.zip
.tarignore.tmp
EOF

echo -e "${GREEN}✓ 排除列表创建完成${NC}"
echo ""

echo -e "${YELLOW}步骤 3/4: 准备打包...${NC}"
# 复制部署说明到根目录
if [ -f "PACKAGE_README.txt" ]; then
    cp PACKAGE_README.txt 部署说明.txt 2>/dev/null || true
fi
echo ""

echo -e "${YELLOW}步骤 4/4: 创建压缩包...${NC}"
tar \
    --exclude='venv' \
    --exclude='env' \
    --exclude='ENV' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.egg-info' \
    --exclude='ui/node_modules' \
    --exclude='ui/dist' \
    --exclude='.vscode' \
    --exclude='.idea' \
    --exclude='.kiro' \
    --exclude='.git' \
    --exclude='*.log' \
    --exclude='simulation_data' \
    --exclude='logs' \
    --exclude='results' \
    --exclude='result' \
    --exclude='tuning_reports/*.png' \
    --exclude='tuning_reports/*.json' \
    --exclude='tuning_reports/*.md' \
    --exclude='.pytest_cache' \
    --exclude='.hypothesis' \
    --exclude='htmlcov' \
    --exclude='.coverage' \
    --exclude='PerfOpt_temp' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='.tarignore.tmp' \
    -czf "${OUTPUT_FILE}" \
    --transform 's,^\./,LoongEnv/,' \
    .

# 清理临时文件
rm -f .tarignore.tmp

# 获取最终文件大小
FINAL_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)

echo ""
echo -e "${GREEN}======================================"
echo "  打包完成！"
echo "======================================${NC}"
echo ""
echo -e "  文件名: ${GREEN}${OUTPUT_FILE}${NC}"
echo -e "  大小:   ${GREEN}${FINAL_SIZE}${NC}"
echo ""
echo -e "${YELLOW}接收方部署步骤:${NC}"
echo "  1. 解压: tar -xzf ${OUTPUT_FILE}"
echo "  2. 进入: cd LoongEnv"
echo "  3. 安装Python依赖: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
echo "  4. 安装前端依赖: cd ui && npm install && cd .."
echo "  5. 启动系统: python tools/start_system.py"
echo ""
echo -e "${GREEN}详细部署说明请查看: DEPLOYMENT_GUIDE.md${NC}"
echo ""
