"""
路径控制算法模块

实现高精度路径跟踪控制算法，包括前馈控制和反馈控制。
支持多种控制策略：PID、计算力矩控制、滑模控制等。

主要功能：
- 高精度路径跟踪（≤0.1mm偏差）
- 动力学前馈控制
- 自适应反馈控制
- 多种控制模式支持
- 实时参数调整
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import time

from ..core.models import RobotModel
from ..core.types import RobotState, TrajectoryPoint, ControlCommand, Vector, AlgorithmError


class ControlMode(Enum):
    """控制模式枚举"""
    PID = "pid"
    COMPUTED_TORQUE = "computed_torque"
    SLIDING_MODE = "sliding_mode"
    ADAPTIVE = "adaptive"


class PathController:
    """
    高精度路径控制器
    
    实现多种控制算法的路径跟踪控制器，支持前馈和反馈控制的组合。
    针对需求1.1, 1.2, 1.4进行优化，确保高精度跟踪性能。
    """
    
    def __init__(
        self, 
        robot_model: RobotModel,
        control_mode: ControlMode = ControlMode.COMPUTED_TORQUE,
        enable_feedforward: bool = True,
        enable_adaptation: bool = True
    ):
        """
        初始化路径控制器
        
        Args:
            robot_model: 机器人模型
            control_mode: 控制模式
            enable_feedforward: 是否启用前馈控制
            enable_adaptation: 是否启用自适应控制
        """
        self.robot_model = robot_model
        self.n_joints = robot_model.n_joints
        self.control_mode = control_mode
        self.enable_feedforward = enable_feedforward
        self.enable_adaptation = enable_adaptation
        
        # 动力学引擎（延迟初始化）
        self._dynamics_engine = None
        
        # PID控制参数
        self.kp = np.ones(self.n_joints) * 200.0   # 比例增益（提高精度）
        self.ki = np.ones(self.n_joints) * 20.0    # 积分增益
        self.kd = np.ones(self.n_joints) * 15.0    # 微分增益
        
        # 计算力矩控制参数
        self.kp_ct = np.ones(self.n_joints) * 500.0  # 计算力矩比例增益
        self.kd_ct = np.ones(self.n_joints) * 50.0   # 计算力矩微分增益
        
        # 滑模控制参数
        self.lambda_sm = np.ones(self.n_joints) * 10.0  # 滑模面参数
        self.k_sm = np.ones(self.n_joints) * 100.0      # 滑模增益
        self.phi_sm = np.ones(self.n_joints) * 0.1      # 边界层厚度
        
        # 自适应控制参数
        self.gamma_adapt = np.ones(self.n_joints) * 0.1  # 自适应增益
        self.param_estimates = np.ones(self.n_joints)     # 参数估计
        
        # 状态变量
        self.integral_error = np.zeros(self.n_joints)
        self.last_error = np.zeros(self.n_joints)
        self.last_time = 0.0
        
        # 性能监控
        self.tracking_errors = []
        self.control_efforts = []
        self.computation_times = []
        
        # 前馈补偿参数
        self.feedforward_gain = 1.0
        self.gravity_compensation_gain = 1.0
        self.friction_compensation_gain = 0.8
    
    @property
    def dynamics_engine(self):
        """延迟初始化动力学引擎"""
        if self._dynamics_engine is None:
            from .dynamics import DynamicsEngine
            self._dynamics_engine = DynamicsEngine(self.robot_model)
        return self._dynamics_engine
    
    def compute_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: Optional[float] = None
    ) -> ControlCommand:
        """
        计算控制指令
        
        Args:
            reference: 参考轨迹点
            current_state: 当前机器人状态
            dt: 时间步长（可选，自动计算）
        
        Returns:
            控制指令
        """
        start_time = time.time()
        
        try:
            # 计算时间步长
            if dt is None:
                dt = current_state.timestamp - self.last_time if self.last_time > 0 else 0.001
                self.last_time = current_state.timestamp
            
            # 前馈控制
            feedforward_command = np.zeros(self.n_joints)
            if self.enable_feedforward:
                feedforward_command = self.feedforward_control(reference)
            
            # 反馈控制
            feedback_command = self.feedback_control(reference, current_state, dt)
            
            # 组合控制指令
            if self.control_mode in [ControlMode.COMPUTED_TORQUE, ControlMode.SLIDING_MODE, ControlMode.ADAPTIVE]:
                # 计算力矩控制模式：输出力矩
                total_torque = feedforward_command + feedback_command
                
                control_command = ControlCommand(
                    joint_torques=total_torque,
                    control_mode="torque",
                    timestamp=current_state.timestamp
                )
            else:
                # 位置控制模式：输出位置
                total_position = reference.position + feedback_command
                
                control_command = ControlCommand(
                    joint_positions=total_position,
                    joint_velocities=reference.velocity.copy() if hasattr(reference, 'velocity') else None,
                    control_mode="position",
                    timestamp=current_state.timestamp
                )
            
            # 记录性能数据
            self._record_performance_data(reference, current_state, start_time)
            
            return control_command
            
        except Exception as e:
            raise AlgorithmError(f"路径控制计算失败: {e}")
    
    def feedforward_control(self, reference: TrajectoryPoint) -> Vector:
        """
        前馈控制算法
        
        基于动力学模型的前馈补偿，包括：
        - 重力补偿
        - 惯性补偿  
        - 摩擦补偿
        
        Args:
            reference: 参考轨迹点
        
        Returns:
            前馈控制输出（力矩或位置修正）
        """
        try:
            if self.control_mode == ControlMode.COMPUTED_TORQUE:
                # 计算力矩控制的前馈
                return self._compute_dynamics_feedforward(reference)
            else:
                # 位置控制的前馈（返回位置修正）
                return self._compute_position_feedforward(reference)
                
        except Exception as e:
            # 前馈失败时返回零补偿
            return np.zeros(self.n_joints)
    
    def _compute_dynamics_feedforward(self, reference: TrajectoryPoint) -> Vector:
        """
        基于动力学的前馈控制
        
        Args:
            reference: 参考轨迹点
        
        Returns:
            前馈力矩
        """
        # 逆动力学计算
        tau_ff = self.dynamics_engine.inverse_dynamics(
            reference.position,
            reference.velocity,
            reference.acceleration
        )
        
        # 重力补偿
        gravity_comp = self.dynamics_engine.gravity_compensation(reference.position)
        tau_ff += self.gravity_compensation_gain * gravity_comp
        
        # 摩擦补偿
        friction_comp = self.dynamics_engine.compute_friction_torque(reference.velocity)
        tau_ff += self.friction_compensation_gain * friction_comp
        
        return self.feedforward_gain * tau_ff
    
    def _compute_position_feedforward(self, reference: TrajectoryPoint) -> Vector:
        """
        位置控制的前馈补偿
        
        Args:
            reference: 参考轨迹点
        
        Returns:
            位置修正量
        """
        # 简化的位置前馈：基于速度和加速度的预测
        dt = 0.001  # 假设1ms控制周期
        
        # 预测下一时刻的位置偏差
        velocity_compensation = reference.velocity * dt
        acceleration_compensation = 0.5 * reference.acceleration * dt * dt
        
        return self.feedforward_gain * (velocity_compensation + acceleration_compensation)
    
    def feedback_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: float
    ) -> Vector:
        """
        反馈控制算法
        
        根据控制模式选择不同的反馈控制策略
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
        
        Returns:
            反馈控制输出
        """
        if self.control_mode == ControlMode.PID:
            return self._pid_control(reference, current_state, dt)
        elif self.control_mode == ControlMode.COMPUTED_TORQUE:
            return self._computed_torque_control(reference, current_state, dt)
        elif self.control_mode == ControlMode.SLIDING_MODE:
            return self._sliding_mode_control(reference, current_state, dt)
        elif self.control_mode == ControlMode.ADAPTIVE:
            return self._adaptive_control(reference, current_state, dt)
        else:
            # 默认使用PID控制
            return self._pid_control(reference, current_state, dt)
    
    def _pid_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: float
    ) -> Vector:
        """
        PID反馈控制
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
        
        Returns:
            PID控制输出
        """
        # 计算位置误差
        position_error = reference.position - current_state.joint_positions
        
        # 更新积分误差（带积分饱和限制）
        self.integral_error += position_error * dt
        integral_limit = 0.1  # 积分限制
        self.integral_error = np.clip(self.integral_error, -integral_limit, integral_limit)
        
        # 计算微分误差
        if dt > 0:
            derivative_error = (position_error - self.last_error) / dt
        else:
            derivative_error = np.zeros(self.n_joints)
        
        self.last_error = position_error.copy()
        
        # PID控制律
        control_output = (
            self.kp * position_error +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )
        
        return control_output
    
    def _computed_torque_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: float
    ) -> Vector:
        """
        计算力矩控制（逆动力学控制）
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
        
        Returns:
            计算力矩控制输出
        """
        # 位置和速度误差
        position_error = reference.position - current_state.joint_positions
        velocity_error = reference.velocity - current_state.joint_velocities
        
        # 期望加速度（PD控制律）
        desired_acceleration = (
            reference.acceleration + 
            self.kp_ct * position_error + 
            self.kd_ct * velocity_error
        )
        
        # 逆动力学计算所需力矩
        tau_feedback = self.dynamics_engine.inverse_dynamics(
            current_state.joint_positions,
            current_state.joint_velocities,
            desired_acceleration
        )
        
        return tau_feedback
    
    def _sliding_mode_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: float
    ) -> Vector:
        """
        滑模控制
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
        
        Returns:
            滑模控制输出
        """
        # 位置和速度误差
        position_error = reference.position - current_state.joint_positions
        velocity_error = reference.velocity - current_state.joint_velocities
        
        # 滑模面
        sliding_surface = velocity_error + self.lambda_sm * position_error
        
        # 滑模控制律（带边界层）
        control_output = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            if abs(sliding_surface[i]) > self.phi_sm[i]:
                # 在边界层外使用符号函数
                control_output[i] = -self.k_sm[i] * np.sign(sliding_surface[i])
            else:
                # 在边界层内使用线性函数（避免抖振）
                control_output[i] = -self.k_sm[i] * sliding_surface[i] / self.phi_sm[i]
        
        # 加上期望加速度项
        desired_acceleration = reference.acceleration + control_output
        
        # 逆动力学计算
        tau_sliding = self.dynamics_engine.inverse_dynamics(
            current_state.joint_positions,
            current_state.joint_velocities,
            desired_acceleration
        )
        
        return tau_sliding
    
    def _adaptive_control(
        self, 
        reference: TrajectoryPoint, 
        current_state: RobotState,
        dt: float
    ) -> Vector:
        """
        自适应控制
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            dt: 时间步长
        
        Returns:
            自适应控制输出
        """
        # 位置和速度误差
        position_error = reference.position - current_state.joint_positions
        velocity_error = reference.velocity - current_state.joint_velocities
        
        # 跟踪误差
        tracking_error = velocity_error + self.lambda_sm * position_error
        
        # 参数自适应律
        if dt > 0:
            self.param_estimates += self.gamma_adapt * tracking_error * dt
        
        # 自适应控制律
        desired_acceleration = (
            reference.acceleration + 
            self.kp_ct * position_error + 
            self.kd_ct * velocity_error +
            self.param_estimates * tracking_error
        )
        
        # 逆动力学计算
        tau_adaptive = self.dynamics_engine.inverse_dynamics(
            current_state.joint_positions,
            current_state.joint_velocities,
            desired_acceleration
        )
        
        return tau_adaptive
    
    def set_control_gains(
        self,
        kp: Optional[Vector] = None,
        ki: Optional[Vector] = None,
        kd: Optional[Vector] = None,
        control_mode: Optional[ControlMode] = None
    ) -> None:
        """
        设置控制增益
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            control_mode: 控制模式
        """
        if kp is not None:
            self.kp = np.array(kp)
            self.kp_ct = np.array(kp) * 2.5  # 计算力矩控制使用更高增益
        
        if ki is not None:
            self.ki = np.array(ki)
        
        if kd is not None:
            self.kd = np.array(kd)
            self.kd_ct = np.array(kd) * 3.3  # 计算力矩控制使用更高增益
        
        if control_mode is not None:
            self.control_mode = control_mode
    
    def auto_tune_gains(
        self,
        reference_trajectory: list,
        simulation_states: list,
        optimization_method: str = "gradient_descent"
    ) -> Dict[str, Vector]:
        """
        自动调参
        
        Args:
            reference_trajectory: 参考轨迹
            simulation_states: 仿真状态序列
            optimization_method: 优化方法
        
        Returns:
            优化后的增益参数
        """
        # 简化的自动调参实现
        # 实际应用中可以使用更复杂的优化算法
        
        best_gains = {
            'kp': self.kp.copy(),
            'ki': self.ki.copy(),
            'kd': self.kd.copy()
        }
        
        best_performance = float('inf')
        
        # 网格搜索（简化版）
        kp_range = np.linspace(50, 500, 5)
        ki_range = np.linspace(5, 50, 5)
        kd_range = np.linspace(2, 30, 5)
        
        for kp_val in kp_range:
            for ki_val in ki_range:
                for kd_val in kd_range:
                    # 设置测试增益
                    test_kp = np.ones(self.n_joints) * kp_val
                    test_ki = np.ones(self.n_joints) * ki_val
                    test_kd = np.ones(self.n_joints) * kd_val
                    
                    # 评估性能（简化）
                    performance = self._evaluate_control_performance(
                        test_kp, test_ki, test_kd,
                        reference_trajectory, simulation_states
                    )
                    
                    if performance < best_performance:
                        best_performance = performance
                        best_gains['kp'] = test_kp
                        best_gains['ki'] = test_ki
                        best_gains['kd'] = test_kd
        
        # 应用最佳增益
        self.set_control_gains(
            kp=best_gains['kp'],
            ki=best_gains['ki'],
            kd=best_gains['kd']
        )
        
        return best_gains
    
    def _evaluate_control_performance(
        self,
        kp: Vector,
        ki: Vector,
        kd: Vector,
        reference_trajectory: list,
        simulation_states: list
    ) -> float:
        """
        评估控制性能
        
        Args:
            kp, ki, kd: 控制增益
            reference_trajectory: 参考轨迹
            simulation_states: 仿真状态
        
        Returns:
            性能指标（越小越好）
        """
        # 简化的性能评估：计算平均跟踪误差
        total_error = 0.0
        count = 0
        
        for ref_point, sim_state in zip(reference_trajectory, simulation_states):
            if hasattr(ref_point, 'position') and hasattr(sim_state, 'joint_positions'):
                error = np.linalg.norm(ref_point.position - sim_state.joint_positions)
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else float('inf')
    
    def get_tracking_performance(self) -> Dict[str, float]:
        """
        获取跟踪性能指标
        
        Returns:
            性能指标字典
        """
        if not self.tracking_errors:
            return {
                'mean_tracking_error': 0.0,
                'max_tracking_error': 0.0,
                'rms_tracking_error': 0.0,
                'mean_computation_time': 0.0
            }
        
        errors = np.array(self.tracking_errors)
        times = np.array(self.computation_times)
        
        return {
            'mean_tracking_error': np.mean(errors),
            'max_tracking_error': np.max(errors),
            'rms_tracking_error': np.sqrt(np.mean(errors**2)),
            'mean_computation_time': np.mean(times),
            'tracking_error_std': np.std(errors)
        }
    
    def reset_controller_state(self) -> None:
        """重置控制器状态"""
        self.integral_error = np.zeros(self.n_joints)
        self.last_error = np.zeros(self.n_joints)
        self.last_time = 0.0
        self.param_estimates = np.ones(self.n_joints)
        
        # 清空性能记录
        self.tracking_errors.clear()
        self.control_efforts.clear()
        self.computation_times.clear()
    
    def _record_performance_data(
        self,
        reference: TrajectoryPoint,
        current_state: RobotState,
        start_time: float
    ) -> None:
        """
        记录性能数据
        
        Args:
            reference: 参考轨迹点
            current_state: 当前状态
            start_time: 计算开始时间
        """
        # 跟踪误差
        tracking_error = np.linalg.norm(
            reference.position - current_state.joint_positions
        )
        self.tracking_errors.append(tracking_error)
        
        # 计算时间
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        # 限制记录长度
        max_records = 1000
        if len(self.tracking_errors) > max_records:
            self.tracking_errors.pop(0)
            self.computation_times.pop(0)
    
    def enable_feedforward_control(self, enable: bool = True) -> None:
        """
        启用/禁用前馈控制
        
        Args:
            enable: 是否启用前馈控制
        """
        self.enable_feedforward = enable
    
    def set_feedforward_gains(
        self,
        feedforward_gain: float = 1.0,
        gravity_gain: float = 1.0,
        friction_gain: float = 0.8
    ) -> None:
        """
        设置前馈控制增益
        
        Args:
            feedforward_gain: 总前馈增益
            gravity_gain: 重力补偿增益
            friction_gain: 摩擦补偿增益
        """
        self.feedforward_gain = feedforward_gain
        self.gravity_compensation_gain = gravity_gain
        self.friction_compensation_gain = friction_gain