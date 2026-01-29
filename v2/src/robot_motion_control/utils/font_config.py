"""
字体配置工具

用于配置matplotlib的中文字体显示
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
from pathlib import Path


def configure_chinese_font():
    """
    配置matplotlib的中文字体
    
    由于系统中可能没有合适的中文字体，直接返回False使用英文标签
    """
    try:
        # 检查是否有可用的中文字体
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 中文字体候选列表
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei', 
            'Noto Sans CJK SC',
            'Source Han Sans CN',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Heiti SC'
        ]
        
        # 查找可用的中文字体
        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            print(f"使用中文字体: {selected_font}")
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return selected_font
        else:
            print("未找到合适的中文字体，将使用英文标签")
            # 配置英文字体
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return None
            
    except Exception as e:
        print(f"字体配置失败: {e}")
        return None


def get_safe_chinese_labels():
    """
    获取安全的中文标签映射
    
    Returns:
        dict: 中文到英文的标签映射
    """
    return {
        # 通用标签
        '参数调优性能对比': 'Parameter Tuning Performance Comparison',
        '参数类型': 'Parameter Type',
        '性能分数': 'Performance Score',
        '优化历史': 'Optimization History',
        '迭代次数': 'Iteration',
        
        # 性能分析标签
        '算法平均执行时间对比': 'Algorithm Average Execution Time Comparison',
        '执行时间 (ms)': 'Execution Time (ms)',
        '算法吞吐量对比': 'Algorithm Throughput Comparison',
        '吞吐量 (ops/s)': 'Throughput (ops/s)',
        '计算复杂度分析': 'Computational Complexity Analysis',
        '输入规模': 'Input Size',
        '执行时间 (s)': 'Execution Time (s)',
        
        # 参数类型
        '控制增益': 'Control Gains',
        '轨迹参数': 'Trajectory Parameters',
        '振动参数': 'Vibration Parameters',
    }


def apply_chinese_font_or_fallback(use_chinese=True):
    """
    应用中文字体或使用英文备用方案
    
    Args:
        use_chinese (bool): 是否尝试使用中文字体
        
    Returns:
        tuple: (是否成功配置中文字体, 标签映射字典)
    """
    chinese_success = False
    label_map = get_safe_chinese_labels()
    
    if use_chinese:
        font = configure_chinese_font()
        chinese_success = font is not None
    
    if not chinese_success:
        print("使用英文标签作为备用方案")
        # 配置英文字体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_success, label_map


# 自动配置函数
def auto_configure_font():
    """自动配置字体，优先使用中文，失败则使用英文"""
    return apply_chinese_font_or_fallback(use_chinese=True)


if __name__ == "__main__":
    # 测试字体配置
    success, labels = auto_configure_font()
    print(f"字体配置成功: {success}")
    print(f"可用标签映射: {len(labels)} 个")