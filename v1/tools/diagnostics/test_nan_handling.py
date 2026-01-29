#!/usr/bin/env python3
"""
测试NaN处理
"""

import numpy as np
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_nan_json_serialization():
    """测试NaN的JSON序列化"""
    print("=" * 60)
    print("测试NaN的JSON序列化")
    print("=" * 60)
    
    # 测试不同的NaN情况
    test_cases = [
        ("正常浮点数", 3.14159),
        ("整数", 42),
        ("NaN", float('nan')),
        ("正无穷", float('inf')),
        ("负无穷", float('-inf')),
        ("NumPy NaN", np.nan),
        ("NumPy Inf", np.inf),
    ]
    
    print("\n直接JSON序列化测试：\n")
    
    for name, value in test_cases:
        try:
            result = json.dumps({"value": value})
            print(f"✓ {name:15s}: {result}")
        except Exception as e:
            print(f"✗ {name:15s}: 错误 - {type(e).__name__}: {e}")
    
    # 测试safe_convert函数
    print("\n\n使用safe_convert函数测试：\n")
    
    def safe_convert(value):
        """安全转换数值，处理NaN和Inf"""
        if isinstance(value, (np.ndarray, list)):
            return [safe_convert(v) for v in value]
        elif isinstance(value, dict):
            return {k: safe_convert(v) for k, v in value.items()}
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None  # 或者返回一个默认值
            return float(value)
        elif isinstance(value, (int, np.integer)):
            return int(value)
        else:
            return value
    
    for name, value in test_cases:
        try:
            converted = safe_convert(value)
            result = json.dumps({"value": converted})
            print(f"✓ {name:15s}: {result}")
        except Exception as e:
            print(f"✗ {name:15s}: 错误 - {type(e).__name__}: {e}")
    
    # 测试复杂对象
    print("\n\n测试复杂对象：\n")
    
    complex_obj = {
        "normal": 3.14,
        "nan": float('nan'),
        "inf": float('inf'),
        "array": np.array([1.0, np.nan, 3.0]),
        "nested": {
            "value": np.inf,
            "list": [1, 2, float('nan')]
        }
    }
    
    print("原始对象:")
    print(complex_obj)
    
    print("\n转换后:")
    converted_obj = safe_convert(complex_obj)
    print(converted_obj)
    
    print("\nJSON序列化:")
    try:
        json_str = json.dumps(converted_obj, indent=2)
        print(json_str)
        print("\n✓ 复杂对象序列化成功")
        return True
    except Exception as e:
        print(f"\n✗ 复杂对象序列化失败: {e}")
        return False


if __name__ == "__main__":
    success = test_nan_json_serialization()
    sys.exit(0 if success else 1)
