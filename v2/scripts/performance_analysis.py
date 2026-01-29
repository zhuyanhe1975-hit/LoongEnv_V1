#!/usr/bin/env python3
"""
机器人运动控制系统性能分析脚本

该脚本执行全面的性能基准测试和分析，生成详细的性能报告。
包括：
- 算法计算性能分析
- 内存使用分析
- 并行计算效率分析
- 实时性能分析
- 计算复杂度分析
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import statistics
import psutil
import gc
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入字体配置
try:
    from robot_motion_control.utils.font_config import auto_configure_font
    # 自动配置字体
    font_success, label_map = auto_configure_font()
except ImportError:
    print("警告: 无法导入字体配置模块，使用默认字体")
    font_success = False
    label_map = {}

from robot_motion_control import (
    RobotMotionController, RobotModel, TrajectoryPlanner,
    PathController, VibrationSuppressor, DynamicsEngine
)
from robot_motion_control.core.types import (
    DynamicsParameters, KinodynamicLimits, RobotState,
    TrajectoryPoint, Waypoint, ControlCommand, PayloadInfo
)
from robot_motion_control.core.controller import ControllerConfig
from robot_motion_control.core.parallel_computing import ParallelMode


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    test_name: str
    execution_time_stats: Dict[str, float]  # mean, std, min, max, p95, p99
    throughput: float
    memory_usage: Dict[str, float]  # peak, retained, efficiency
    cpu_usage: float
    success_rate: float
    error_metrics: Dict[str, float]
    timestamp: str


@dataclass
class ComplexityAnalysis:
    """计算复杂度分析结果"""
    algorithm_name: str
    input_sizes: List[int]
    execution_times: List[float]
    complexity_order: str  # O(1), O(n), O(n^2), etc.
    regression_r2: float
    scaling_factor: float


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.results = []
        self.complexity_results = []
        self.process = psutil.Process()
        
    def create_benchmark_robot_model(self) -> RobotModel:
        """创建基准测试机器人模型"""
        n_joints = 6
        
        dynamics_params = DynamicsParameters(
            masses=[25.0, 20.0, 15.0, 10.0, 5.0, 2.0],
            centers_of_mass=[[0.0, 0.0, 0.15]] * n_joints,
            inertias=[[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]] * n_joints,
            friction_coeffs=[0.15] * n_joints,
            gravity=[0.0, 0.0, -9.81]
        )
        
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[3.14] * n_joints,
            min_joint_positions=[-3.14] * n_joints,
            max_joint_velocities=[3.0] * n_joints,
            max_joint_accelerations=[15.0] * n_joints,
            max_joint_jerks=[150.0] * n_joints,
            max_joint_torques=[300.0] * n_joints
        )
        
        return RobotModel(
            name="benchmark_robot",
            n_joints=n_joints,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits
        )
    
    def analyze_trajectory_planning_performance(self, robot_model: RobotModel) -> PerformanceMetrics:
        """分析轨迹规划性能"""
        print("分析轨迹规划性能...")
        
        planner = TrajectoryPlanner(robot_model)
        
        # 不同复杂度的测试用例
        test_cases = [
            {'name': 'simple', 'waypoints': 5, 'iterations': 50},
            {'name': 'medium', 'waypoints': 15, 'iterations': 30},
            {'name': 'complex', 'waypoints': 30, 'iterations': 20},
            {'name': 'large', 'waypoints': 50, 'iterations': 10}
        ]
        
        all_times = []
        memory_peak = 0
        memory_start = self.process.memory_info().rss / 1024 / 1024
        
        for test_case in test_cases:
            waypoints = self._generate_waypoints(test_case['waypoints'], 'curved')
            
            for _ in range(test_case['iterations']):
                start_time = time.time()
                
                # S型插补
                trajectory_s7 = planner.interpolate_s7_trajectory(waypoints)
                
                # TOPP算法
                trajectory_topp = planner.generate_topp_trajectory(
                    waypoints, robot_model.kinodynamic_limits
                )
                
                execution_time = time.time() - start_time
                all_times.append(execution_time)
                
                # 监控内存使用
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_peak = max(memory_peak, current_memory)
        
        # 统计分析
        execution_stats = {
            'mean': statistics.mean(all_times),
            'std': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'min': min(all_times),
            'max': max(all_times),
            'p95': np.percentile(all_times, 95),
            'p99': np.percentile(all_times, 99)
        }
        
        memory_stats = {
            'peak': memory_peak,
            'retained': self.process.memory_info().rss / 1024 / 1024 - memory_start,
            'efficiency': sum(tc['waypoints'] * tc['iterations'] for tc in test_cases) / (memory_peak - memory_start) if memory_peak > memory_start else 0
        }
        
        return PerformanceMetrics(
            test_name="trajectory_planning",
            execution_time_stats=execution_stats,
            throughput=1.0 / execution_stats['mean'],
            memory_usage=memory_stats,
            cpu_usage=0.0,  # 简化
            success_rate=1.0,
            error_metrics={},
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_dynamics_computation_performance(self, robot_model: RobotModel) -> PerformanceMetrics:
        """分析动力学计算性能"""
        print("分析动力学计算性能...")
        
        dynamics_engine = DynamicsEngine(robot_model)
        
        # 测试数据
        q = np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0])
        qd = np.array([0.1, 0.05, 0.02, 0.01, 0.0, 0.0])
        qdd = np.array([0.01, 0.005, 0.002, 0.001, 0.0, 0.0])
        tau = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
        
        # 测试不同动力学计算
        computation_tests = [
            ('inverse_dynamics', lambda: dynamics_engine.inverse_dynamics(q, qd, qdd)),
            ('forward_dynamics', lambda: dynamics_engine.forward_dynamics(q, qd, tau)),
            ('jacobian', lambda: dynamics_engine.jacobian(q)),
            ('gravity_compensation', lambda: dynamics_engine.gravity_compensation(q))
        ]
        
        all_times = []
        memory_start = self.process.memory_info().rss / 1024 / 1024
        
        for test_name, test_func in computation_tests:
            # 预热
            for _ in range(10):
                test_func()
            
            # 性能测试
            test_times = []
            for _ in range(1000):
                start_time = time.time()
                result = test_func()
                execution_time = time.time() - start_time
                test_times.append(execution_time)
                
                # 验证数值稳定性
                if isinstance(result, np.ndarray):
                    assert not np.any(np.isnan(result)), f"{test_name} 产生NaN"
                    assert not np.any(np.isinf(result)), f"{test_name} 产生无穷大"
            
            all_times.extend(test_times)
        
        # 统计分析
        execution_stats = {
            'mean': statistics.mean(all_times),
            'std': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'min': min(all_times),
            'max': max(all_times),
            'p95': np.percentile(all_times, 95),
            'p99': np.percentile(all_times, 99)
        }
        
        memory_end = self.process.memory_info().rss / 1024 / 1024
        memory_stats = {
            'peak': memory_end,
            'retained': memory_end - memory_start,
            'efficiency': len(all_times) / (memory_end - memory_start) if memory_end > memory_start else 0
        }
        
        return PerformanceMetrics(
            test_name="dynamics_computation",
            execution_time_stats=execution_stats,
            throughput=1.0 / execution_stats['mean'],
            memory_usage=memory_stats,
            cpu_usage=0.0,
            success_rate=1.0,
            error_metrics={},
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_path_control_performance(self, robot_model: RobotModel) -> PerformanceMetrics:
        """分析路径控制性能"""
        print("分析路径控制性能...")
        
        controller = PathController(robot_model)
        
        # 创建测试轨迹点
        reference_point = TrajectoryPoint(
            position=np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0]),
            velocity=np.array([0.1, 0.05, 0.02, 0.01, 0.0, 0.0]),
            acceleration=np.array([0.01, 0.005, 0.002, 0.001, 0.0, 0.0]),
            jerk=np.zeros(6),
            time=1.0,
            path_parameter=0.5
        )
        
        current_state = RobotState(
            joint_positions=np.array([0.45, 0.28, 0.18, 0.09, 0.0, 0.0]),
            joint_velocities=np.array([0.08, 0.04, 0.015, 0.008, 0.0, 0.0]),
            joint_accelerations=np.zeros(6),
            joint_torques=np.zeros(6),
            end_effector_transform=np.eye(4),
            timestamp=1.0
        )
        
        # 性能测试
        num_iterations = 5000
        execution_times = []
        tracking_errors = []
        memory_start = self.process.memory_info().rss / 1024 / 1024
        
        for i in range(num_iterations):
            start_time = time.time()
            
            control_command = controller.compute_control(reference_point, current_state)
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # 计算跟踪误差
            if control_command.joint_positions is not None:
                tracking_error = np.linalg.norm(
                    reference_point.position - control_command.joint_positions
                )
                tracking_errors.append(tracking_error)
        
        # 统计分析
        execution_stats = {
            'mean': statistics.mean(execution_times),
            'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'min': min(execution_times),
            'max': max(execution_times),
            'p95': np.percentile(execution_times, 95),
            'p99': np.percentile(execution_times, 99)
        }
        
        memory_end = self.process.memory_info().rss / 1024 / 1024
        memory_stats = {
            'peak': memory_end,
            'retained': memory_end - memory_start,
            'efficiency': num_iterations / (memory_end - memory_start) if memory_end > memory_start else 0
        }
        
        error_metrics = {
            'mean_tracking_error': statistics.mean(tracking_errors) if tracking_errors else 0,
            'max_tracking_error': max(tracking_errors) if tracking_errors else 0,
            'p95_tracking_error': np.percentile(tracking_errors, 95) if tracking_errors else 0
        }
        
        return PerformanceMetrics(
            test_name="path_control",
            execution_time_stats=execution_stats,
            throughput=1.0 / execution_stats['mean'],
            memory_usage=memory_stats,
            cpu_usage=0.0,
            success_rate=1.0,
            error_metrics=error_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_computational_complexity(self, robot_model: RobotModel):
        """分析计算复杂度"""
        print("分析计算复杂度...")
        
        # 分析轨迹规划复杂度
        planner = TrajectoryPlanner(robot_model)
        
        input_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
        execution_times = []
        
        for size in input_sizes:
            waypoints = self._generate_waypoints(size, 'linear')
            
            # 多次测试取平均
            times = []
            for _ in range(5):
                start_time = time.time()
                trajectory = planner.interpolate_s7_trajectory(waypoints)
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            avg_time = statistics.mean(times)
            execution_times.append(avg_time)
            print(f"  路径点数: {size}, 平均时间: {avg_time:.6f}s")
        
        # 拟合复杂度
        complexity = self._fit_complexity(input_sizes, execution_times)
        self.complexity_results.append(ComplexityAnalysis(
            algorithm_name="trajectory_planning",
            input_sizes=input_sizes,
            execution_times=execution_times,
            complexity_order=complexity['order'],
            regression_r2=complexity['r2'],
            scaling_factor=complexity['factor']
        ))
    
    def _fit_complexity(self, sizes: List[int], times: List[float]) -> Dict[str, Any]:
        """拟合计算复杂度"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        sizes_np = np.array(sizes).reshape(-1, 1)
        times_np = np.array(times)
        
        # 尝试不同的复杂度模型
        models = {
            'O(1)': np.ones_like(sizes),
            'O(n)': sizes,
            'O(n^2)': np.array(sizes) ** 2,
            'O(n log n)': np.array(sizes) * np.log(sizes),
            'O(n^3)': np.array(sizes) ** 3
        }
        
        best_r2 = -1
        best_order = 'O(1)'
        best_factor = 0
        
        for order, x_data in models.items():
            try:
                reg = LinearRegression()
                reg.fit(x_data.reshape(-1, 1), times_np)
                y_pred = reg.predict(x_data.reshape(-1, 1))
                r2 = r2_score(times_np, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_order = order
                    best_factor = reg.coef_[0]
            except:
                continue
        
        return {
            'order': best_order,
            'r2': best_r2,
            'factor': best_factor
        }
    
    def _generate_waypoints(self, count: int, pattern: str) -> List[Waypoint]:
        """生成测试路径点"""
        waypoints = []
        
        if pattern == 'linear':
            for i in range(count):
                t = i / (count - 1) if count > 1 else 0
                pos = np.array([t, t*0.5, t*0.3, t*0.2, t*0.1, 0.0])
                waypoints.append(Waypoint(position=pos))
        
        elif pattern == 'curved':
            for i in range(count):
                t = i / (count - 1) * 2 * np.pi if count > 1 else 0
                pos = np.array([
                    0.5 * np.sin(t),
                    0.3 * np.cos(t),
                    0.2 * np.sin(2*t),
                    0.1 * np.cos(2*t),
                    0.05 * np.sin(4*t),
                    0.0
                ])
                waypoints.append(Waypoint(position=pos))
        
        return waypoints
    
    def generate_performance_report(self, output_dir: str = "reports"):
        """生成性能分析报告"""
        print("生成性能分析报告...")
        
        # 创建报告目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成JSON报告
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'python_version': sys.version
            },
            'performance_metrics': [asdict(result) for result in self.results],
            'complexity_analysis': [asdict(result) for result in self.complexity_results]
        }
        
        json_path = os.path.join(output_dir, 'performance_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_markdown_report(report_data, output_dir)
        
        # 生成性能图表
        self._generate_performance_plots(output_dir)
        
        print(f"性能报告已生成到: {output_dir}")
    
    def _generate_markdown_report(self, report_data: Dict, output_dir: str):
        """生成Markdown格式的报告"""
        md_path = os.path.join(output_dir, 'performance_report.md')
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 机器人运动控制系统性能分析报告\n\n")
            f.write(f"生成时间: {report_data['timestamp']}\n\n")
            
            # 系统信息
            f.write("## 系统信息\n\n")
            f.write(f"- CPU核心数: {report_data['system_info']['cpu_count']}\n")
            f.write(f"- 内存总量: {report_data['system_info']['memory_total']:.1f} GB\n")
            f.write(f"- Python版本: {report_data['system_info']['python_version']}\n\n")
            
            # 性能指标
            f.write("## 性能指标\n\n")
            for metric in report_data['performance_metrics']:
                f.write(f"### {metric['test_name']}\n\n")
                f.write(f"- 平均执行时间: {metric['execution_time_stats']['mean']*1000:.3f} ms\n")
                f.write(f"- P95执行时间: {metric['execution_time_stats']['p95']*1000:.3f} ms\n")
                f.write(f"- P99执行时间: {metric['execution_time_stats']['p99']*1000:.3f} ms\n")
                f.write(f"- 吞吐量: {metric['throughput']:.2f} ops/s\n")
                f.write(f"- 内存峰值: {metric['memory_usage']['peak']:.1f} MB\n")
                f.write(f"- 成功率: {metric['success_rate']:.1%}\n\n")
                
                if metric['error_metrics']:
                    f.write("**误差指标:**\n")
                    for key, value in metric['error_metrics'].items():
                        f.write(f"- {key}: {value:.6f}\n")
                    f.write("\n")
            
            # 复杂度分析
            if report_data['complexity_analysis']:
                f.write("## 计算复杂度分析\n\n")
                for analysis in report_data['complexity_analysis']:
                    f.write(f"### {analysis['algorithm_name']}\n\n")
                    f.write(f"- 复杂度阶: {analysis['complexity_order']}\n")
                    f.write(f"- 拟合度 (R²): {analysis['regression_r2']:.4f}\n")
                    f.write(f"- 缩放因子: {analysis['scaling_factor']:.6f}\n\n")
            
            # 性能建议
            f.write("## 性能优化建议\n\n")
            self._generate_optimization_recommendations(f, report_data)
    
    def _generate_optimization_recommendations(self, f, report_data: Dict):
        """生成性能优化建议"""
        f.write("基于性能分析结果，提出以下优化建议：\n\n")
        
        # 分析各个模块的性能
        for metric in report_data['performance_metrics']:
            test_name = metric['test_name']
            mean_time = metric['execution_time_stats']['mean']
            throughput = metric['throughput']
            
            if test_name == 'trajectory_planning':
                if mean_time > 0.1:  # 100ms
                    f.write(f"- **轨迹规划优化**: 当前平均执行时间 {mean_time*1000:.1f}ms，建议优化TOPP算法实现\n")
                if throughput < 50:
                    f.write(f"- **轨迹规划吞吐量**: 当前吞吐量 {throughput:.1f} ops/s，考虑并行化处理\n")
            
            elif test_name == 'path_control':
                if mean_time > 0.001:  # 1ms
                    f.write(f"- **路径控制优化**: 当前平均执行时间 {mean_time*1000:.3f}ms，建议优化控制算法\n")
                if 'mean_tracking_error' in metric['error_metrics']:
                    error = metric['error_metrics']['mean_tracking_error']
                    if error > 0.0001:
                        f.write(f"- **跟踪精度**: 当前平均跟踪误差 {error:.6f}，建议调整控制参数\n")
            
            elif test_name == 'dynamics_computation':
                if mean_time > 0.0001:  # 0.1ms
                    f.write(f"- **动力学计算**: 当前平均执行时间 {mean_time*1000:.3f}ms，考虑使用更高效的动力学库\n")
        
        # 复杂度相关建议
        for analysis in report_data.get('complexity_analysis', []):
            if analysis['complexity_order'] in ['O(n^2)', 'O(n^3)']:
                f.write(f"- **算法复杂度**: {analysis['algorithm_name']} 的复杂度为 {analysis['complexity_order']}，建议优化算法实现\n")
        
        f.write("\n")
    
    def _generate_performance_plots(self, output_dir: str):
        """生成性能图表"""
        try:
            import matplotlib.pyplot as plt
            
            # 执行时间对比图
            test_names = [result.test_name for result in self.results]
            mean_times = [result.execution_time_stats['mean'] * 1000 for result in self.results]  # ms
            
            plt.figure(figsize=(10, 6))
            plt.bar(test_names, mean_times)
            title = '算法平均执行时间对比' if font_success else 'Algorithm Average Execution Time Comparison'
            ylabel = '执行时间 (ms)' if font_success else 'Execution Time (ms)'
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'))
            plt.close()
            
            # 吞吐量对比图
            throughputs = [result.throughput for result in self.results]
            
            plt.figure(figsize=(10, 6))
            plt.bar(test_names, throughputs)
            title = '算法吞吐量对比' if font_success else 'Algorithm Throughput Comparison'
            ylabel = '吞吐量 (ops/s)' if font_success else 'Throughput (ops/s)'
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))
            plt.close()
            
            # 复杂度分析图
            if self.complexity_results:
                for analysis in self.complexity_results:
                    plt.figure(figsize=(10, 6))
                    plt.plot(analysis.input_sizes, analysis.execution_times, 'bo-')
                    title = f'{analysis.algorithm_name} 计算复杂度分析 ({analysis.complexity_order})' if font_success else f'{analysis.algorithm_name} Computational Complexity Analysis ({analysis.complexity_order})'
                    xlabel = '输入规模' if font_success else 'Input Size'
                    ylabel = '执行时间 (s)' if font_success else 'Execution Time (s)'
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{analysis.algorithm_name}_complexity.png'))
                    plt.close()
            
            print("性能图表已生成")
            
        except ImportError:
            print("matplotlib未安装，跳过图表生成")
    
    def run_full_analysis(self):
        """运行完整的性能分析"""
        print("开始机器人运动控制系统性能分析...")
        print("=" * 60)
        
        # 创建基准测试模型
        robot_model = self.create_benchmark_robot_model()
        
        # 执行各项性能分析
        try:
            # 轨迹规划性能
            trajectory_metrics = self.analyze_trajectory_planning_performance(robot_model)
            self.results.append(trajectory_metrics)
            
            # 动力学计算性能
            dynamics_metrics = self.analyze_dynamics_computation_performance(robot_model)
            self.results.append(dynamics_metrics)
            
            # 路径控制性能
            control_metrics = self.analyze_path_control_performance(robot_model)
            self.results.append(control_metrics)
            
            # 计算复杂度分析
            self.analyze_computational_complexity(robot_model)
            
            # 生成报告
            self.generate_performance_report()
            
            print("=" * 60)
            print("性能分析完成！")
            
            # 打印摘要
            self._print_summary()
            
        except Exception as e:
            print(f"性能分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_summary(self):
        """打印性能分析摘要"""
        print("\n性能分析摘要:")
        print("-" * 40)
        
        for result in self.results:
            print(f"{result.test_name}:")
            print(f"  平均执行时间: {result.execution_time_stats['mean']*1000:.3f} ms")
            print(f"  P99执行时间: {result.execution_time_stats['p99']*1000:.3f} ms")
            print(f"  吞吐量: {result.throughput:.2f} ops/s")
            print(f"  内存峰值: {result.memory_usage['peak']:.1f} MB")
            print()


def main():
    """主函数"""
    analyzer = PerformanceAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()