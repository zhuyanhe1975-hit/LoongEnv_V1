#!/usr/bin/env python3
"""
机器人控制系统启动脚本

同时启动前端React应用和后端Python API服务
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("检查系统依赖...")
    
    # 检查Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        print(f"Node.js版本: {result.stdout.strip()}")
    except FileNotFoundError:
        print("错误: 未找到Node.js，请先安装Node.js")
        return False
    
    # 检查npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        print(f"npm版本: {result.stdout.strip()}")
    except FileNotFoundError:
        print("错误: 未找到npm")
        return False
    
    # 检查Python依赖 - 使用虚拟环境
    project_root = Path(__file__).parent.parent
    venv_python = project_root / "venv" / "bin" / "python"
    if venv_python.exists():
        python_executable = str(venv_python)
        print(f"使用虚拟环境Python: {python_executable}")
    else:
        python_executable = "python3"
        print(f"使用系统Python: {python_executable}")
    
    try:
        # 测试导入依赖
        test_script = """
try:
    import flask
    import flask_cors
    import numpy
    import matplotlib
    print("Python依赖检查通过")
except ImportError as e:
    print(f"错误: 缺少Python依赖 {e}")
    exit(1)
"""
        result = subprocess.run([python_executable, '-c', test_script], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print("请确保在虚拟环境中安装了所需依赖:")
            print("./venv/bin/pip install flask flask-cors numpy matplotlib scipy")
            return False
        else:
            print(result.stdout.strip())
    except Exception as e:
        print(f"Python依赖检查失败: {e}")
        return False
    
    return True

def install_frontend_dependencies():
    """安装前端依赖"""
    # 脚本在tools/目录，需要回到项目根目录
    project_root = Path(__file__).parent.parent
    ui_dir = project_root / "ui"
    
    if not (ui_dir / "node_modules").exists():
        print("安装前端依赖...")
        try:
            subprocess.run(['npm', 'install'], cwd=ui_dir, check=True)
            print("前端依赖安装完成")
        except subprocess.CalledProcessError:
            print("错误: 前端依赖安装失败")
            return False
    else:
        print("前端依赖已存在")
    
    return True

def start_backend():
    """启动后端服务"""
    print("启动后端API服务...")
    # 脚本在tools/目录，需要回到项目根目录
    project_root = Path(__file__).parent.parent
    backend_script = project_root / "ui" / "backend_api.py"
    
    # 检查是否在虚拟环境中
    venv_python = project_root / "venv" / "bin" / "python"
    if venv_python.exists():
        python_executable = str(venv_python)
        print(f"使用虚拟环境Python: {python_executable}")
    else:
        python_executable = sys.executable
        print(f"使用系统Python: {python_executable}")
    
    try:
        process = subprocess.Popen([
            python_executable, str(backend_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待后端启动
        time.sleep(3)
        
        if process.poll() is None:
            print("后端服务启动成功 (PID: {})".format(process.pid))
            print("后端API地址: http://localhost:5003")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"后端启动失败: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"启动后端服务失败: {e}")
        return None

def start_frontend():
    """启动前端服务"""
    print("启动前端React应用...")
    # 脚本在tools/目录，需要回到项目根目录
    project_root = Path(__file__).parent.parent
    ui_dir = project_root / "ui"
    
    try:
        process = subprocess.Popen([
            'npm', 'run', 'dev'
        ], cwd=ui_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待前端启动
        time.sleep(5)
        
        if process.poll() is None:
            print("前端服务启动成功 (PID: {})".format(process.pid))
            print("前端应用地址: http://localhost:3000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"前端启动失败: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"启动前端服务失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("机器人控制系统启动器")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 安装前端依赖
    if not install_frontend_dependencies():
        sys.exit(1)
    
    processes = []
    
    try:
        # 启动后端
        backend_process = start_backend()
        if backend_process:
            processes.append(backend_process)
        else:
            print("后端启动失败，退出")
            sys.exit(1)
        
        # 启动前端
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
        else:
            print("前端启动失败，但后端仍在运行")
        
        print("\n" + "=" * 60)
        print("系统启动完成！")
        print("=" * 60)
        print("前端地址: http://localhost:3000")
        print("后端API: http://localhost:5003")
        print("\n按 Ctrl+C 停止所有服务")
        print("=" * 60)
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                for process in processes[:]:
                    if process.poll() is not None:
                        print(f"进程 {process.pid} 已退出")
                        processes.remove(process)
                
                if not processes:
                    print("所有进程已退出")
                    break
                    
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止服务...")
    
    finally:
        # 清理进程
        for process in processes:
            try:
                print(f"停止进程 {process.pid}")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"强制终止进程 {process.pid}")
                process.kill()
            except Exception as e:
                print(f"停止进程时出错: {e}")
        
        print("所有服务已停止")

if __name__ == "__main__":
    main()