"""
安全监控算法模块

实现算法异常检测、安全限制监控和碰撞检测等安全功能。
"""

from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from ..core.models import RobotModel
from ..core.types import RobotState, Vector, AlgorithmError


@dataclass
class SafetyStatus:
    """安全状态"""
    is_safe: bool
    violations: list
    warnings: list
    emergency_stop_required: bool = False


class SafetyMonitor:
    """
    安全监控器
    
    实现多层安全监控，包括算法异常检测、限制检查和碰撞检测。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化安全监控器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        self.n_joints = robot_model.n_joints
        
        # 安全阈值
        self.max_position_error = 0.1  # rad
        self.max_velocity_error = 1.0  # rad/s
        self.max_acceleration = np.array(robot_model.kinodynamic_limits.max_joint_accelerations)
        self.max_jerk = np.array(robot_model.kinodynamic_limits.max_joint_jerks)
        
        # 历史状态用于检测异常
        self.state_history = []
        self.max_history_length = 10
    
    def check_safety(self, current_state: RobotState) -> SafetyStatus:
        """
        检查安全状态
        
        Args:
            current_state: 当前机器人状态
        
        Returns:
            安全状态
        """
        violations = []
        warnings = []
        
        # 添加到历史记录
        self.state_history.append(current_state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
        
        # 检查位置限制
        position_violations = self._check_position_limits(current_state)
        violations.extend(position_violations)
        
        # 检查速度限制
        velocity_violations = self._check_velocity_limits(current_state)
        violations.extend(velocity_violations)
        
        # 检查加速度限制
        acceleration_violations = self._check_acceleration_limits(current_state)
        violations.extend(acceleration_violations)
        
        # 检查数值稳定性
        numerical_warnings = self._check_numerical_stability(current_state)
        warnings.extend(numerical_warnings)
        
        # 检查碰撞风险
        collision_warnings = self._check_collision_risk(current_state)
        warnings.extend(collision_warnings)
        
        # 判断是否需要紧急停止
        emergency_stop_required = any("CRITICAL" in v for v in violations)
        
        return SafetyStatus(
            is_safe=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            emergency_stop_required=emergency_stop_required
        )
    
    def _check_position_limits(self, state: RobotState) -> list:
        """检查位置限制"""
        violations = []
        
        min_pos = np.array(self.robot_model.kinodynamic_limits.min_joint_positions)
        max_pos = np.array(self.robot_model.kinodynamic_limits.max_joint_positions)
        
        for i, pos in enumerate(state.joint_positions):
            if pos < min_pos[i] or pos > max_pos[i]:
                violations.append(f"关节{i}位置超限: {pos:.3f} not in [{min_pos[i]:.3f}, {max_pos[i]:.3f}]")
        
        return violations
    
    def _check_velocity_limits(self, state: RobotState) -> list:
        """检查速度限制"""
        violations = []
        
        max_vel = np.array(self.robot_model.kinodynamic_limits.max_joint_velocities)
        
        for i, vel in enumerate(state.joint_velocities):
            if abs(vel) > max_vel[i]:
                violations.append(f"关节{i}速度超限: {vel:.3f} > {max_vel[i]:.3f}")
        
        return violations
    
    def _check_acceleration_limits(self, state: RobotState) -> list:
        """检查加速度限制"""
        violations = []
        
        for i, acc in enumerate(state.joint_accelerations):
            if abs(acc) > self.max_acceleration[i]:
                violations.append(f"关节{i}加速度超限: {acc:.3f} > {self.max_acceleration[i]:.3f}")
        
        return violations
    
    def _check_numerical_stability(self, state: RobotState) -> list:
        """检查数值稳定性"""
        warnings = []
        
        # 检查NaN或无穷大
        if np.any(np.isnan(state.joint_positions)) or np.any(np.isinf(state.joint_positions)):
            warnings.append("位置数据包含NaN或无穷大值")
        
        if np.any(np.isnan(state.joint_velocities)) or np.any(np.isinf(state.joint_velocities)):
            warnings.append("速度数据包含NaN或无穷大值")
        
        # 检查异常大的数值
        if np.any(np.abs(state.joint_positions) > 100):
            warnings.append("位置数值异常大")
        
        if np.any(np.abs(state.joint_velocities) > 100):
            warnings.append("速度数值异常大")
        
        return warnings
    
    def _check_collision_risk(self, state: RobotState) -> list:
        """检查碰撞风险"""
        warnings = []
        
        try:
            # 使用完整的碰撞检测算法
            from .collision_detection import CollisionMonitor
            
            if not hasattr(self, '_collision_monitor'):
                self._collision_monitor = CollisionMonitor(self.robot_model)
            
            # 创建虚拟控制指令用于碰撞检测
            from ..core.types import ControlCommand
            dummy_command = ControlCommand(
                joint_positions=state.joint_positions,
                joint_velocities=state.joint_velocities,
                control_mode="position",
                timestamp=state.timestamp
            )
            
            # 检测碰撞
            collisions, _ = self._collision_monitor.update(state, dummy_command)
            
            # 转换为警告信息
            for collision in collisions:
                severity_level = "严重" if collision.severity > 0.8 else "中等" if collision.severity > 0.5 else "轻微"
                warnings.append(
                    f"检测到{severity_level}碰撞风险: {collision.collision_pair.geometry1.name} "
                    f"与 {collision.collision_pair.geometry2.name} 距离 {collision.distance:.3f}m "
                    f"(严重程度: {collision.severity:.2f})"
                )
            
        except Exception as e:
            # 回退到简化的碰撞检测
            for i in range(len(state.joint_positions) - 1):
                pos_diff = abs(state.joint_positions[i+1] - state.joint_positions[i])
                if pos_diff < 0.1:  # 简化阈值
                    warnings.append(f"关节{i}和{i+1}距离过近，可能存在碰撞风险")
        
        return warnings


class AlgorithmRecoveryManager:
    """
    算法恢复管理器
    
    处理算法异常和数值问题，提供自动恢复机制。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化恢复管理器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        self.recovery_strategies = {
            "numerical_instability": self._recover_numerical_instability,
            "convergence_failure": self._recover_convergence_failure,
            "safety_violation": self._recover_safety_violation
        }
    
    def detect_algorithm_error(self, computation_state: Dict[str, Any]) -> Optional[str]:
        """
        检测算法错误
        
        Args:
            computation_state: 计算状态
        
        Returns:
            错误类型（如果有）
        """
        # 检查数值稳定性
        if "result" in computation_state:
            result = computation_state["result"]
            if isinstance(result, np.ndarray):
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    return "numerical_instability"
        
        # 检查收敛性
        if "iterations" in computation_state and "max_iterations" in computation_state:
            if computation_state["iterations"] >= computation_state["max_iterations"]:
                return "convergence_failure"
        
        return None
    
    def _recover_numerical_instability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """恢复数值不稳定"""
        # 简化恢复策略：重置为安全值
        if "result" in state:
            result = state["result"]
            if isinstance(result, np.ndarray):
                # 将NaN和无穷大替换为零
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                state["result"] = result
        
        return state
    
    def _recover_convergence_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """恢复收敛失败"""
        # 简化恢复策略：使用上一次的结果
        if "last_valid_result" in state:
            state["result"] = state["last_valid_result"]
        
        return state
    
    def _recover_safety_violation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """恢复安全违规"""
        # 简化恢复策略：停止运动
        state["emergency_stop"] = True
        
        return state