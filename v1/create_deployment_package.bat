@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ======================================
echo   LoongEnv 部署包创建工具 (Windows)
echo ======================================
echo.

REM 检查是否安装了tar（Windows 10 1803+自带）
where tar >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到tar命令
    echo 请使用Windows 10 1803或更高版本，或安装Git for Windows
    pause
    exit /b 1
)

REM 检查当前目录
if not exist "requirements.txt" (
    echo [错误] 请在项目根目录运行此脚本
    pause
    exit /b 1
)

echo [1/3] 检查当前目录...
echo √ 目录检查通过
echo.

REM 生成时间戳
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set OUTPUT_FILE=LoongEnv_deploy_%TIMESTAMP%.tar.gz

echo [2/3] 准备打包...
echo.

echo [3/3] 创建压缩包...
tar ^
    --exclude=venv ^
    --exclude=env ^
    --exclude=ENV ^
    --exclude=__pycache__ ^
    --exclude=*.pyc ^
    --exclude=*.pyo ^
    --exclude=*.egg-info ^
    --exclude=ui/node_modules ^
    --exclude=ui/dist ^
    --exclude=.vscode ^
    --exclude=.idea ^
    --exclude=.kiro ^
    --exclude=.git ^
    --exclude=*.log ^
    --exclude=simulation_data ^
    --exclude=logs ^
    --exclude=results ^
    --exclude=result ^
    --exclude=tuning_reports/*.png ^
    --exclude=tuning_reports/*.json ^
    --exclude=tuning_reports/*.md ^
    --exclude=.pytest_cache ^
    --exclude=.hypothesis ^
    --exclude=htmlcov ^
    --exclude=.coverage ^
    --exclude=PerfOpt_temp ^
    --exclude=*.tar.gz ^
    --exclude=*.zip ^
    -czf %OUTPUT_FILE% .

if %errorlevel% equ 0 (
    echo.
    echo ======================================
    echo   打包完成！
    echo ======================================
    echo.
    for %%A in (%OUTPUT_FILE%) do echo   文件名: %OUTPUT_FILE%
    for %%A in (%OUTPUT_FILE%) do echo   大小:   %%~zA 字节
    echo.
    echo 接收方部署步骤:
    echo   1. 解压: tar -xzf %OUTPUT_FILE%
    echo   2. 进入: cd LoongEnv
    echo   3. 安装Python依赖: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    echo   4. 安装前端依赖: cd ui ^&^& npm install ^&^& cd ..
    echo   5. 启动系统: python tools\start_system.py
    echo.
    echo 详细部署说明请查看: DEPLOYMENT_GUIDE.md
) else (
    echo.
    echo [错误] 打包失败
)

echo.
pause
