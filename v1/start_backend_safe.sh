#!/bin/bash

# 安全启动后端服务脚本
# 在崩溃时自动重启，并记录日志

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/backend_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "启动机器人控制系统后端服务..."
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动计数器
START_COUNT=0
MAX_RESTARTS=5

while true; do
    START_COUNT=$((START_COUNT + 1))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动后端服务 (第 $START_COUNT 次)" | tee -a "$LOG_FILE"
    
    # 运行后端服务
    python3 ui/backend_api.py 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 后端服务退出，退出码: $EXIT_CODE" | tee -a "$LOG_FILE"
    
    # 如果是正常退出（Ctrl+C），则不重启
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo "后端服务正常退出"
        break
    fi
    
    # 如果重启次数过多，停止
    if [ $START_COUNT -ge $MAX_RESTARTS ]; then
        echo "错误: 后端服务已崩溃 $MAX_RESTARTS 次，停止重启" | tee -a "$LOG_FILE"
        echo "请检查日志文件: $LOG_FILE"
        exit 1
    fi
    
    # 等待5秒后重启
    echo "5秒后自动重启..." | tee -a "$LOG_FILE"
    sleep 5
done

echo "后端服务已停止"
