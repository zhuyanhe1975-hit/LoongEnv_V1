#!/usr/bin/env python3
"""
机器人控制系统后端API服务

提供REST API接口，连接React UI和Python后端功能。
支持参数调优、轨迹规划、路径控制等核心功能。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import json
import time
import threading
from typing import Dict, List, Any, Optional
import traceback

# 导入机器人控制模块
try:
    from src.robot_motion_control.core.models import RobotModel
    from src.robot_motion_control.core.controller import RobotMotionController, ControllerConfig
    from src.robot_motion_control.core.types import RobotState, TrajectoryPoint, Waypoint
    from src.robot_motion_control.algorithms.parameter_tuning import (
        ParameterTuner, OptimizationConfig, PerformanceWeights, 
        ParameterType, OptimizationMethod, TuningReportGenerator
    )
    from src.robot_motion_control.algorithms.trajectory_planning import TrajectoryPlanner
    from src.robot_motion_control.algorithms.path_control import PathController, ControlMode
    from src.robot_motion_control.algorithms.dynamics import DynamicsEngine
    from src.robot_motion_control.simulation.environment import SimulationEnvironment
except ImportError as e:
    print(f"警告: 无法导入机器人控制模块: {e}")
    print("将使用模拟模式运行")

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
robot_model = None
controller = None
parameter_tuner = None
simulation_env = None
current_trajectory = []
tuning_status = {"running": False, "progress": 0, "results": None}

def initialize_robot_system():
    """初始化机器人系统"""
    global robot_model, controller, parameter_tuner, simulation_env
    
    try:
        # 导入必要的类型
        from src.robot_motion_control.core.types import DynamicsParameters, KinodynamicLimits
        
        # 创建ER15-1400的动力学参数
        dynamics_params = DynamicsParameters(
            masses=[54.52, 11.11, 25.03, 10.81, 4.48, 0.28],  # 从URDF提取的质量
            centers_of_mass=[
                [0.09835, -0.02908, -0.0995],
                [0.25263, -0.00448, 0.15471],
                [0.03913, -0.02495, 0.03337],
                [-0.00132, -0.0012, -0.30035],
                [0.0004, -0.03052, 0.01328],
                [0, 0, 0]
            ],
            inertias=[
                [[1.16916852, 0.0865367, -0.47354118],
                 [0.0865367, 1.39934751, 0.11859959],
                 [-0.47354118, 0.11859959, 1.00920236]],
                [[0.04507715, -0.00764148, -0.01800527],
                 [-0.00764148, 0.58269106, 0.00057833],
                 [-0.01800527, 0.00057833, 0.60235638]],
                [[0.33717585, 0.06955124, 0.00142677],
                 [0.06955124, 0.38576036, -0.00313441],
                 [0.00142677, -0.00313441, 0.24095087]],
                [[0.28066314, -0.00003381, 0.00084678],
                 [-0.00003381, 0.27142738, 0.00437676],
                 [0.00084678, 0.00437676, 0.04425281]],
                [[0.01710138, -0.00002606, 0.00000867],
                 [-0.00002606, 0.01098115, -0.00175535],
                 [0.00000867, -0.00175535, 0.01408541]],
                [[0.0001346961, 0.0000076, -0.00000827],
                 [0.0000076, 0.0001645611, 0.000118982],
                 [-0.00000827, 0.000118982, 0.001539171]]
            ],
            friction_coeffs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        
        # 创建运动学动力学限制
        kinodynamic_limits = KinodynamicLimits(
            max_joint_positions=[2.967, 1.5708, 3.0543, 3.316, 2.2689, 6.2832],
            min_joint_positions=[-2.967, -2.7925, -1.4835, -3.316, -2.2689, -6.2832],
            max_joint_velocities=[3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
            max_joint_accelerations=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            max_joint_jerks=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            max_joint_torques=[100.0, 100.0, 100.0, 50.0, 50.0, 25.0]
        )
        
        # 创建机器人模型（ER15-1400）
        robot_model = RobotModel(
            name="ER15-1400",
            n_joints=6,
            dynamics_params=dynamics_params,
            kinodynamic_limits=kinodynamic_limits,
            urdf_path=str(project_root / "models" / "ER15-1400.urdf")
        )
        
        # 创建控制器
        config = ControllerConfig(
            control_frequency=1000.0,
            enable_parallel_computing=True,
            enable_vibration_suppression=True
        )
        controller = RobotMotionController(robot_model, config)
        
        # 创建参数调优器
        parameter_tuner = ParameterTuner(robot_model)
        
        # 创建仿真环境
        simulation_env = SimulationEnvironment(robot_model)
        
        print("机器人系统初始化成功")
        return True
        
    except Exception as e:
        print(f"机器人系统初始化失败: {e}")
        traceback.print_exc()
        return False

# API路由定义

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "backend_available": robot_model is not None
    })

@app.route('/api/robot/status', methods=['GET'])
def get_robot_status():
    """获取机器人状态"""
    try:
        if not robot_model:
            return jsonify({"error": "机器人系统未初始化"}), 500
        
        # 获取当前状态（模拟）
        current_state = simulation_env.get_current_state() if simulation_env else None
        
        if not current_state:
            # 创建默认状态
            current_state = RobotState(
                joint_positions=np.zeros(6),
                joint_velocities=np.zeros(6),
                joint_accelerations=np.zeros(6),
                joint_torques=np.zeros(6),
                end_effector_transform=np.eye(4),  # 4x4单位矩阵
                timestamp=time.time()
            )
        
        return jsonify({
            "isConnected": True,
            "jointPositions": current_state.joint_positions.tolist(),
            "jointVelocities": current_state.joint_velocities.tolist(),
            "jointTorques": current_state.joint_torques.tolist(),
            "timestamp": current_state.timestamp,
            "operationMode": "automatic",
            "safetyStatus": "safe",
            "systemLoad": {
                "cpu": 45.0,
                "memory": 68.0,
                "temperature": 42.0
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/robot/specs', methods=['GET'])
def get_robot_specs():
    """获取机器人规格"""
    try:
        if not robot_model:
            return jsonify({"error": "机器人系统未初始化"}), 500
        
        return jsonify({
            "name": "ER15-1400",
            "manufacturer": "Elite Robot",
            "dof": 6,
            "payload": 15,
            "reach": 1500,
            "repeatability": 0.1,
            "totalMass": 206,
            "jointLimits": [
                {"name": f"joint_{i+1}", "lower": -3.14, "upper": 3.14, "range": 6.28}
                for i in range(6)
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tuning/start', methods=['POST'])
def start_parameter_tuning():
    """启动参数调优"""
    global tuning_status
    
    try:
        if not parameter_tuner:
            return jsonify({"error": "参数调优器未初始化"}), 500
        
        if tuning_status["running"]:
            return jsonify({"error": "参数调优正在进行中"}), 400
        
        data = request.get_json()
        
        # 解析调优配置
        config = OptimizationConfig(
            method=OptimizationMethod(data.get("method", "differential_evolution")),
            max_iterations=data.get("maxIterations", 50),
            tolerance=data.get("tolerance", 1e-6),
            population_size=data.get("populationSize", 15),
            verbose=True
        )
        
        # 解析性能权重
        weights_data = data.get("performanceWeights", {})
        weights = PerformanceWeights(
            tracking_accuracy=weights_data.get("trackingAccuracy", 0.4),
            settling_time=weights_data.get("settlingTime", 0.2),
            overshoot=weights_data.get("overshoot", 0.15),
            energy_efficiency=weights_data.get("energyEfficiency", 0.1),
            vibration_suppression=weights_data.get("vibrationSuppression", 0.1),
            safety_margin=weights_data.get("safetyMargin", 0.05)
        )
        
        # 更新调优器配置
        parameter_tuner.config = config
        parameter_tuner.performance_weights = weights
        
        # 在后台线程中运行调优
        def run_tuning():
            global tuning_status
            
            try:
                tuning_status["running"] = True
                tuning_status["progress"] = 0
                
                # 创建测试轨迹
                test_waypoints = [
                    Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    Waypoint(position=np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])),
                    Waypoint(position=np.array([1.0, -0.6, 0.4, 0.0, 0.0, 0.0])),
                    Waypoint(position=np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0])),
                    Waypoint(position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                ]
                
                # 生成参考轨迹
                trajectory_planner = TrajectoryPlanner(robot_model)
                reference_trajectory = trajectory_planner.interpolate_s7_trajectory(test_waypoints)
                
                # 创建测试场景
                test_scenarios = [
                    {"initial_state": RobotState(
                        joint_positions=np.zeros(6),
                        joint_velocities=np.zeros(6),
                        joint_accelerations=np.zeros(6),
                        joint_torques=np.zeros(6),
                        end_effector_transform=np.eye(4),
                        timestamp=0.0
                    )}
                ]
                
                tuning_status["progress"] = 10
                
                # 执行综合调优
                parameter_types = data.get("parameterTypes", ["control_gains", "trajectory_params"])
                param_types_enum = [ParameterType(pt) for pt in parameter_types]
                
                results = parameter_tuner.comprehensive_tuning(
                    reference_trajectory, test_scenarios, param_types_enum
                )
                
                tuning_status["progress"] = 90
                
                # 生成报告
                report_generator = TuningReportGenerator()
                report = report_generator.generate_report(
                    results, robot_model, config, weights, {}
                )
                
                tuning_status["progress"] = 100
                tuning_status["results"] = {
                    "success": True,
                    "overallImprovement": report.overall_performance_improvement,
                    "results": {
                        param_type.value: {
                            "success": result.success,
                            "bestPerformance": result.best_performance,
                            "computationTime": result.computation_time,
                            "optimalParameters": {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in result.optimal_parameters.items()
                            }
                        }
                        for param_type, result in results.items()
                    },
                    "recommendations": report.recommendations,
                    "plotPaths": report.plots_paths
                }
                
            except Exception as e:
                tuning_status["results"] = {
                    "success": False,
                    "error": str(e)
                }
                traceback.print_exc()
            finally:
                tuning_status["running"] = False
        
        # 启动后台线程
        threading.Thread(target=run_tuning, daemon=True).start()
        
        return jsonify({"message": "参数调优已启动", "status": "started"})
        
    except Exception as e:
        tuning_status["running"] = False
        return jsonify({"error": str(e)}), 500

@app.route('/api/tuning/status', methods=['GET'])
def get_tuning_status():
    """获取调优状态"""
    return jsonify(tuning_status)

@app.route('/api/tuning/stop', methods=['POST'])
def stop_parameter_tuning():
    """停止参数调优"""
    global tuning_status
    
    tuning_status["running"] = False
    tuning_status["progress"] = 0
    
    return jsonify({"message": "参数调优已停止"})

@app.route('/api/trajectory/plan', methods=['POST'])
def plan_trajectory():
    """规划轨迹"""
    global current_trajectory
    
    try:
        if not controller:
            return jsonify({"error": "控制器未初始化"}), 500
        
        data = request.get_json()
        waypoints_data = data.get("waypoints", [])
        
        # 转换路径点
        waypoints = []
        for wp_data in waypoints_data:
            waypoint = Waypoint(
                position=np.array(wp_data["position"]),
                velocity=np.array(wp_data.get("velocity", [0]*6)),
                time=wp_data.get("time", 0.0)
            )
            waypoints.append(waypoint)
        
        # 规划轨迹
        trajectory = controller.plan_trajectory(
            waypoints, 
            optimize_time=data.get("optimizeTime", True)
        )
        
        current_trajectory = trajectory
        
        # 转换为JSON格式
        trajectory_data = []
        for point in trajectory:
            trajectory_data.append({
                "position": point.position.tolist(),
                "velocity": point.velocity.tolist(),
                "acceleration": point.acceleration.tolist(),
                "time": point.time,
                "pathParameter": point.path_parameter
            })
        
        return jsonify({
            "success": True,
            "trajectory": trajectory_data,
            "totalTime": trajectory[-1].time if trajectory else 0.0,
            "totalPoints": len(trajectory)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/trajectory/current', methods=['GET'])
def get_current_trajectory():
    """获取当前轨迹"""
    global current_trajectory
    
    try:
        if not current_trajectory:
            return jsonify({"trajectory": [], "totalTime": 0.0, "totalPoints": 0})
        
        trajectory_data = []
        for point in current_trajectory:
            trajectory_data.append({
                "position": point.position.tolist(),
                "velocity": point.velocity.tolist(),
                "acceleration": point.acceleration.tolist(),
                "time": point.time,
                "pathParameter": point.path_parameter
            })
        
        return jsonify({
            "trajectory": trajectory_data,
            "totalTime": current_trajectory[-1].time,
            "totalPoints": len(current_trajectory)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/control/gains', methods=['GET', 'POST'])
def control_gains():
    """获取或设置控制增益"""
    try:
        if not controller:
            return jsonify({"error": "控制器未初始化"}), 500
        
        path_controller = controller.path_controller
        
        if request.method == 'GET':
            return jsonify({
                "kp": path_controller.kp.tolist(),
                "ki": path_controller.ki.tolist(),
                "kd": path_controller.kd.tolist(),
                "controlMode": path_controller.control_mode.value
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            if "kp" in data:
                path_controller.kp = np.array(data["kp"])
            if "ki" in data:
                path_controller.ki = np.array(data["ki"])
            if "kd" in data:
                path_controller.kd = np.array(data["kd"])
            if "controlMode" in data:
                path_controller.control_mode = ControlMode(data["controlMode"])
            
            return jsonify({"message": "控制增益已更新"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/performance', methods=['GET'])
def get_performance_metrics():
    """获取性能指标"""
    try:
        if not controller:
            return jsonify({"error": "控制器未初始化"}), 500
        
        metrics = controller.get_performance_metrics()
        
        return jsonify({
            "computationTime": metrics.computation_time,
            "memoryUsage": metrics.memory_usage,
            "trackingError": metrics.tracking_error,
            "vibrationAmplitude": metrics.vibration_amplitude,
            "successRate": metrics.success_rate
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/<filename>', methods=['GET'])
def get_report_file(filename):
    """获取报告文件"""
    try:
        reports_dir = project_root / "tuning_reports"
        file_path = reports_dir / filename
        
        if file_path.exists():
            return send_file(str(file_path))
        else:
            return jsonify({"error": "文件不存在"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """启动仿真"""
    try:
        if not simulation_env:
            return jsonify({"error": "仿真环境未初始化"}), 500
        
        data = request.get_json()
        duration = data.get("duration", 10.0)
        
        # 启动仿真
        simulation_env.start_simulation(duration)
        
        return jsonify({"message": "仿真已启动", "duration": duration})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """停止仿真"""
    try:
        if not simulation_env:
            return jsonify({"error": "仿真环境未初始化"}), 500
        
        simulation_env.stop_simulation()
        
        return jsonify({"message": "仿真已停止"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "API端点不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "内部服务器错误"}), 500

if __name__ == '__main__':
    print("启动机器人控制系统后端API服务...")
    
    # 初始化机器人系统
    if initialize_robot_system():
        print("机器人系统初始化成功")
    else:
        print("机器人系统初始化失败，将以模拟模式运行")
    
    # 启动Flask服务器
    app.run(
        host='0.0.0.0',
        port=5003,
        debug=True,
        threaded=True
    )