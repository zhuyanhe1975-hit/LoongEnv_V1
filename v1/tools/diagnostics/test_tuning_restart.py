#!/usr/bin/env python3
"""
测试参数调优重启功能

验证第二次调优是否能正常执行
"""

import requests
import time
import json

API_BASE = "http://localhost:5006/api"

def test_tuning_restart():
    """测试调优重启"""
    
    print("=" * 60)
    print("测试参数调优重启功能")
    print("=" * 60)
    
    # 测试配置
    tuning_config = {
        "method": "differential_evolution",
        "maxIterations": 10,  # 减少迭代次数以加快测试
        "tolerance": 1e-6,
        "populationSize": 5,  # 减少种群大小
        "parameterTypes": ["control_gains"],
        "performanceWeights": {
            "trackingAccuracy": 0.4,
            "settlingTime": 0.2,
            "overshoot": 0.15,
            "energyEfficiency": 0.1,
            "vibrationSuppression": 0.1,
            "safetyMargin": 0.05
        }
    }
    
    # 第一次调优
    print("\n[测试1] 启动第一次参数调优...")
    try:
        response = requests.post(
            f"{API_BASE}/tuning/start",
            json=tuning_config,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ 第一次调优启动成功")
        else:
            print(f"✗ 第一次调优启动失败: {response.status_code}")
            print(f"  错误: {response.json()}")
            return False
            
    except Exception as e:
        print(f"✗ 第一次调优启动异常: {e}")
        return False
    
    # 等待第一次调优完成
    print("\n等待第一次调优完成...")
    max_wait = 60  # 最多等待60秒
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            status_response = requests.get(f"{API_BASE}/tuning/status", timeout=5)
            status = status_response.json()
            
            if not status.get("running"):
                print(f"✓ 第一次调优完成")
                print(f"  进度: {status.get('progress')}%")
                if status.get("results"):
                    print(f"  结果: {status['results'].get('success', False)}")
                break
            else:
                print(f"  进度: {status.get('progress')}%", end='\r')
                
        except Exception as e:
            print(f"✗ 获取状态失败: {e}")
            
        time.sleep(2)
    else:
        print("✗ 第一次调优超时")
        return False
    
    # 等待一小段时间
    print("\n等待2秒...")
    time.sleep(2)
    
    # 第二次调优
    print("\n[测试2] 启动第二次参数调优...")
    try:
        response = requests.post(
            f"{API_BASE}/tuning/start",
            json=tuning_config,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ 第二次调优启动成功")
        else:
            print(f"✗ 第二次调优启动失败: {response.status_code}")
            print(f"  错误: {response.json()}")
            return False
            
    except Exception as e:
        print(f"✗ 第二次调优启动异常: {e}")
        return False
    
    # 等待第二次调优完成
    print("\n等待第二次调优完成...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            status_response = requests.get(f"{API_BASE}/tuning/status", timeout=5)
            status = status_response.json()
            
            if not status.get("running"):
                print(f"✓ 第二次调优完成")
                print(f"  进度: {status.get('progress')}%")
                if status.get("results"):
                    print(f"  结果: {status['results'].get('success', False)}")
                break
            else:
                print(f"  进度: {status.get('progress')}%", end='\r')
                
        except Exception as e:
            print(f"✗ 获取状态失败: {e}")
            
        time.sleep(2)
    else:
        print("✗ 第二次调优超时")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 测试通过：参数调优可以正常重启")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # 检查后端是否运行
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("错误: 后端服务未运行")
            print("请先启动后端: python ui/backend_api.py")
            exit(1)
    except Exception as e:
        print(f"错误: 无法连接到后端服务: {e}")
        print("请先启动后端: python ui/backend_api.py")
        exit(1)
    
    # 运行测试
    success = test_tuning_restart()
    exit(0 if success else 1)
