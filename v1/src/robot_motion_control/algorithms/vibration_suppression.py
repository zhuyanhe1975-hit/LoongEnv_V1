"""
振动抑制算法模块

实现输入整形技术和柔性关节补偿算法，消除机器人运动中的振动。
包含增强的柔性关节动力学模型、完善的柔性补偿控制算法和末端反馈补偿。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy import signal
from scipy.linalg import solve_continuous_are

from ..core.models import RobotModel
from ..core.types import ControlCommand, Vector, RobotState, PayloadInfo


@dataclass
class FlexibleJointParameters:
    """柔性关节参数"""
    joint_stiffness: Vector          # 关节刚度 [Nm/rad]
    joint_damping: Vector            # 关节阻尼 [Nm⋅s/rad]
    motor_inertia: Vector            # 电机惯量 [kg⋅m²]
    link_inertia: Vector             # 连杆惯量 [kg⋅m²]
    gear_ratio: Vector               # 减速比
    transmission_compliance: Vector   # 传动柔性 [rad/Nm]


@dataclass 
class EndEffectorSensorData:
    """末端执行器传感器数据"""
    position: Vector                 # 末端位置 [m]
    velocity: Vector                 # 末端速度 [m/s]
    acceleration: Vector             # 末端加速度 [m/s²]
    force: Vector                    # 末端力 [N]
    torque: Vector                   # 末端力矩 [Nm]
    timestamp: float                 # 时间戳 [s]


@dataclass
class VirtualSensorState:
    """虚拟传感器状态"""
    estimated_position: Vector       # 估计位置
    estimated_velocity: Vector       # 估计速度
    estimation_error: Vector         # 估计误差
    confidence: float                # 置信度 [0-1]


class VibrationSuppressor:
    """
    振动抑制器
    
    实现输入整形和柔性关节补偿算法，有效抑制机器人运动振动。
    """
    
    def __init__(self, robot_model: RobotModel, flexible_params: Optional[FlexibleJointParameters] = None):
        """
        初始化振动抑制器
        
        Args:
            robot_model: 机器人模型
            flexible_params: 柔性关节参数，如果为None则使用默认值
        """
        self.robot_model = robot_model
        self.n_joints = robot_model.n_joints
        
        # 柔性关节参数
        if flexible_params is None:
            self.flexible_params = self._initialize_default_flexible_params()
        else:
            self.flexible_params = flexible_params
        
        # 输入整形器参数
        self.natural_frequencies = np.ones(self.n_joints) * 10.0  # Hz
        self.damping_ratios = np.ones(self.n_joints) * 0.1
        
        # 输入整形器脉冲序列
        self.shaper_impulses = self._design_input_shaper()
        
        # 历史输入缓冲区
        self.input_buffer = []
        self.max_buffer_size = 100
        
        # 柔性关节状态估计器
        self.flexible_joint_observer = self._initialize_flexible_observer()
        
        # 末端反馈控制器
        self.end_effector_controller = self._initialize_end_effector_controller()
        
        # 虚拟传感器
        self.virtual_sensor = self._initialize_virtual_sensor()
        
        # 历史状态缓冲区（用于状态估计）
        self.state_history = []
        self.max_state_history = 50
    
    def _initialize_default_flexible_params(self) -> FlexibleJointParameters:
        """初始化默认柔性关节参数"""
        return FlexibleJointParameters(
            joint_stiffness=np.ones(self.n_joints) * 1e5,      # 默认刚度 100kNm/rad
            joint_damping=np.ones(self.n_joints) * 100.0,      # 默认阻尼 100Nm⋅s/rad
            motor_inertia=np.ones(self.n_joints) * 0.01,       # 默认电机惯量 0.01kg⋅m²
            link_inertia=np.ones(self.n_joints) * 0.1,         # 默认连杆惯量 0.1kg⋅m²
            gear_ratio=np.ones(self.n_joints) * 100.0,         # 默认减速比 100:1
            transmission_compliance=np.ones(self.n_joints) * 1e-6  # 默认传动柔性
        )
    
    def _initialize_flexible_observer(self) -> Dict[str, Any]:
        """初始化柔性关节状态观测器"""
        # 柔性关节双质量模型的状态空间表示
        # 状态: [θm, θl, θm_dot, θl_dot] (电机角度、连杆角度、电机角速度、连杆角速度)
        observer = {}
        
        for i in range(self.n_joints):
            # 系统矩阵 A
            Jm = self.flexible_params.motor_inertia[i]
            Jl = self.flexible_params.link_inertia[i]
            K = self.flexible_params.joint_stiffness[i]
            D = self.flexible_params.joint_damping[i]
            
            A = np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-K/Jm, K/Jm, -D/Jm, D/Jm],
                [K/Jl, -K/Jl, D/Jl, -D/Jl]
            ])
            
            # 输入矩阵 B
            B = np.array([[0], [0], [1/Jm], [0]])
            
            # 输出矩阵 C (观测连杆角度)
            C = np.array([[0, 1, 0, 0]])
            
            # 设计观测器增益 (极点配置)
            desired_poles = np.array([-10, -12, -15, -18]) * (i + 1)  # 每个关节不同的极点
            try:
                L = signal.place_poles(A.T, C.T, desired_poles).gain_matrix.T
            except:
                # 如果极点配置失败，使用默认增益
                L = np.array([[10], [5], [100], [50]])
            
            observer[f'joint_{i}'] = {
                'A': A,
                'B': B,
                'C': C,
                'L': L,
                'state': np.zeros(4),  # 初始状态估计
                'error_covariance': np.eye(4) * 0.1
            }
        
        return observer
    
    def _initialize_end_effector_controller(self) -> Dict[str, Any]:
        """初始化末端执行器反馈控制器"""
        return {
            'position_gains': np.array([1000.0, 1000.0, 1000.0]),  # 位置增益
            'velocity_gains': np.array([100.0, 100.0, 100.0]),     # 速度增益
            'force_gains': np.array([0.1, 0.1, 0.1]),              # 力增益
            'integral_gains': np.array([10.0, 10.0, 10.0]),        # 积分增益
            'integral_error': np.zeros(3),                          # 积分误差
            'max_integral_error': 0.1,                              # 最大积分误差
            'feedforward_compensation': True                         # 前馈补偿开关
        }
    
    def _initialize_virtual_sensor(self) -> Dict[str, Any]:
        """初始化虚拟传感器"""
        return {
            'kalman_filter': self._setup_kalman_filter(),
            'estimation_window': 10,                    # 估计窗口大小
            'confidence_threshold': 0.8,                # 置信度阈值
            'sensor_fusion_weights': np.array([0.7, 0.3])  # 传感器融合权重 [编码器, 加速度计]
        }
    
    def _setup_kalman_filter(self) -> Dict[str, np.ndarray]:
        """设置卡尔曼滤波器用于虚拟传感器"""
        # 状态: [position, velocity, acceleration]
        n_states = 3 * 3  # 3D位置、速度、加速度
        
        # 状态转移矩阵 (恒定加速度模型)
        dt = 0.001  # 假设1ms采样时间
        F = np.eye(n_states)
        for i in range(3):  # x, y, z 三个方向
            base_idx = i * 3
            F[base_idx, base_idx + 1] = dt          # position += velocity * dt
            F[base_idx, base_idx + 2] = 0.5 * dt**2 # position += 0.5 * acceleration * dt^2
            F[base_idx + 1, base_idx + 2] = dt      # velocity += acceleration * dt
        
        # 观测矩阵 (观测位置和加速度)
        H = np.zeros((6, n_states))
        for i in range(3):
            H[i, i * 3] = 1.0      # 观测位置
            H[i + 3, i * 3 + 2] = 1.0  # 观测加速度
        
        # 过程噪声协方差
        Q = np.eye(n_states) * 0.01
        
        # 观测噪声协方差
        R = np.eye(6)
        R[:3, :3] *= 0.001  # 位置观测噪声
        R[3:, 3:] *= 0.1    # 加速度观测噪声
        
        # 初始状态协方差
        P = np.eye(n_states) * 0.1
        
        return {
            'F': F,  # 状态转移矩阵
            'H': H,  # 观测矩阵
            'Q': Q,  # 过程噪声协方差
            'R': R,  # 观测噪声协方差
            'P': P,  # 状态协方差
            'x': np.zeros(n_states),  # 状态估计
        }
        """
        应用输入整形
        
        Args:
            command: 原始控制指令
        
        Returns:
            整形后的控制指令
        """
        # 添加到输入缓冲区
        self.input_buffer.append(command)
        
        # 限制缓冲区大小
        if len(self.input_buffer) > self.max_buffer_size:
            self.input_buffer.pop(0)
        
        # 应用输入整形
        shaped_positions = self._apply_shaper_to_positions(command.joint_positions)
        
        # 创建整形后的指令
        shaped_command = ControlCommand(
            joint_positions=shaped_positions,
            joint_velocities=command.joint_velocities,
            joint_accelerations=command.joint_accelerations,
            joint_torques=command.joint_torques,
            control_mode=command.control_mode,
            timestamp=command.timestamp
        )
        
        return shaped_command
    
    def _design_input_shaper(self) -> List[Tuple[float, float]]:
        """
        设计输入整形器
        
        Returns:
            脉冲序列 [(时间, 幅值), ...]
        """
        # 零振动输入整形器（ZV Shaper）
        impulses = []
        
        for i in range(self.n_joints):
            freq = self.natural_frequencies[i]
            zeta = self.damping_ratios[i]
            
            # 计算整形器参数
            wd = freq * np.sqrt(1 - zeta**2)  # 阻尼频率
            K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
            
            # ZV整形器脉冲
            A1 = 1 / (1 + K)
            A2 = K / (1 + K)
            t1 = 0.0
            t2 = np.pi / wd
            
            impulses.append([(t1, A1), (t2, A2)])
        
        return impulses
    
    def _apply_shaper_to_positions(self, positions: Vector) -> Vector:
        """
        对位置应用输入整形
        
        Args:
            positions: 原始位置
        
        Returns:
            整形后的位置
        """
        # 简化实现：应用低通滤波器
        if len(self.input_buffer) < 3:
            return positions
        
        # 使用最近3个输入的加权平均
        weights = np.array([0.25, 0.5, 0.25])
        shaped_positions = np.zeros_like(positions)
        
        for i, weight in enumerate(weights):
            if i < len(self.input_buffer):
                cmd = self.input_buffer[-(i+1)]
                if cmd.joint_positions is not None:
                    shaped_positions += weight * cmd.joint_positions
        
        return shaped_positions
    
    def apply_input_shaping(self, command: ControlCommand) -> ControlCommand:
        """
        应用输入整形
        
        Args:
            command: 原始控制指令
        
        Returns:
            整形后的控制指令
        """
        # 添加到输入缓冲区
        self.input_buffer.append(command)
        
        # 限制缓冲区大小
        if len(self.input_buffer) > self.max_buffer_size:
            self.input_buffer.pop(0)
        
        # 应用输入整形
        shaped_positions = self._apply_shaper_to_positions(command.joint_positions)
        
        # 创建整形后的指令
        shaped_command = ControlCommand(
            joint_positions=shaped_positions,
            joint_velocities=command.joint_velocities,
            joint_accelerations=command.joint_accelerations,
            joint_torques=command.joint_torques,
            control_mode=command.control_mode,
            timestamp=command.timestamp
        )
        
        return shaped_command
    
    def compensate_flexible_joints(self, command: ControlCommand, 
                                 current_state: RobotState,
                                 payload_info: Optional[PayloadInfo] = None) -> ControlCommand:
        """
        增强的柔性关节补偿算法
        
        Args:
            command: 原始控制指令
            current_state: 当前机器人状态
            payload_info: 负载信息（可选）
        
        Returns:
            补偿后的控制指令
        """
        # 更新状态历史
        self._update_state_history(current_state)
        
        # 更新柔性关节观测器
        self._update_flexible_observer(current_state, command)
        
        # 计算柔性补偿
        flexibility_compensation = self._compute_flexibility_compensation(
            command, current_state, payload_info
        )
        
        # 计算传动系统补偿
        transmission_compensation = self._compute_transmission_compensation(
            command, current_state
        )
        
        # 计算负载自适应补偿
        load_adaptive_compensation = self._compute_load_adaptive_compensation(
            command, current_state, payload_info
        )
        
        # 组合所有补偿项
        total_compensation = (flexibility_compensation + 
                            transmission_compensation + 
                            load_adaptive_compensation)
        
        # 应用补偿
        if command.joint_torques is not None:
            compensated_torques = command.joint_torques + total_compensation
        else:
            compensated_torques = total_compensation
        
        compensated_command = ControlCommand(
            joint_positions=command.joint_positions,
            joint_velocities=command.joint_velocities,
            joint_accelerations=command.joint_accelerations,
            joint_torques=compensated_torques,
            control_mode=command.control_mode,
            timestamp=command.timestamp
        )
        
        return compensated_command
    
    def apply_end_effector_feedback(self, command: ControlCommand,
                                  sensor_data: EndEffectorSensorData,
                                  desired_end_effector_state: Optional[Dict] = None) -> ControlCommand:
        """
        应用末端执行器反馈补偿
        
        Args:
            command: 原始控制指令
            sensor_data: 末端执行器传感器数据
            desired_end_effector_state: 期望的末端执行器状态
        
        Returns:
            反馈补偿后的控制指令
        """
        # 更新虚拟传感器
        virtual_state = self._update_virtual_sensor(sensor_data)
        
        # 计算末端位置误差
        if desired_end_effector_state is not None:
            position_error = (desired_end_effector_state.get('position', np.zeros(3)) - 
                            sensor_data.position)
            velocity_error = (desired_end_effector_state.get('velocity', np.zeros(3)) - 
                            sensor_data.velocity)
        else:
            # 如果没有期望状态，使用零误差（仅用于振动抑制）
            position_error = np.zeros(3)
            velocity_error = np.zeros(3)
        
        # 计算末端反馈控制力
        feedback_force = self._compute_end_effector_feedback_force(
            position_error, velocity_error, sensor_data
        )
        
        # 将末端力转换为关节力矩
        try:
            # 获取雅可比矩阵 - 需要通过动力学引擎
            # 这里简化处理，假设有6x6的雅可比矩阵，但只使用前3行（位置部分）
            jacobian = np.eye(6)[:3, :]  # 3x6矩阵，只考虑位置
            
            # 转换为关节空间
            joint_torque_compensation = jacobian.T @ feedback_force
            
        except Exception as e:
            # 如果雅可比计算失败，使用零补偿
            joint_torque_compensation = np.zeros(self.n_joints)
        
        # 应用补偿
        if command.joint_torques is not None:
            compensated_torques = command.joint_torques + joint_torque_compensation
        else:
            compensated_torques = joint_torque_compensation
        
        compensated_command = ControlCommand(
            joint_positions=command.joint_positions,
            joint_velocities=command.joint_velocities,
            joint_accelerations=command.joint_accelerations,
            joint_torques=compensated_torques,
            control_mode=command.control_mode,
            timestamp=command.timestamp
        )
        
        return compensated_command
    
    def _design_input_shaper(self) -> List[Tuple[float, float]]:
        """更新状态历史"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_state_history:
            self.state_history.pop(0)
    
    def _update_flexible_observer(self, current_state: RobotState, command: ControlCommand) -> None:
        """更新柔性关节状态观测器"""
        for i in range(self.n_joints):
            observer = self.flexible_joint_observer[f'joint_{i}']
            
            # 获取观测值 (连杆角度)
            y = np.array([[current_state.joint_positions[i]]])
            
            # 获取控制输入 (电机力矩)
            u = np.array([[command.joint_torques[i] if command.joint_torques is not None else 0.0]])
            
            # 预测步骤
            x_pred = observer['A'] @ observer['state'].reshape(-1, 1) + observer['B'] @ u
            P_pred = observer['A'] @ observer['error_covariance'] @ observer['A'].T + np.eye(4) * 0.01
            
            # 更新步骤
            innovation = y - observer['C'] @ x_pred
            S = observer['C'] @ P_pred @ observer['C'].T + np.array([[0.001]])  # 观测噪声
            K = P_pred @ observer['C'].T @ np.linalg.inv(S)
            
            # 更新状态估计
            observer['state'] = (x_pred + K @ innovation).flatten()
            observer['error_covariance'] = (np.eye(4) - K @ observer['C']) @ P_pred
    
    def _compute_flexibility_compensation(self, command: ControlCommand, 
                                        current_state: RobotState,
                                        payload_info: Optional[PayloadInfo]) -> Vector:
        """计算柔性补偿项"""
        compensation = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            observer = self.flexible_joint_observer[f'joint_{i}']
            
            # 获取估计的电机和连杆角度
            theta_m_est = observer['state'][0]  # 电机角度
            theta_l_est = observer['state'][1]  # 连杆角度
            theta_m_dot_est = observer['state'][2]  # 电机角速度
            theta_l_dot_est = observer['state'][3]  # 连杆角速度
            
            # 计算柔性变形
            deflection = theta_m_est - theta_l_est
            deflection_rate = theta_m_dot_est - theta_l_dot_est
            
            # 柔性补偿力矩
            K = self.flexible_params.joint_stiffness[i]
            D = self.flexible_params.joint_damping[i]
            
            # 基本柔性补偿
            flexibility_torque = K * deflection + D * deflection_rate
            
            # 负载自适应调整
            if payload_info is not None:
                # 根据负载质量调整补偿增益
                load_factor = 1.0 + 0.1 * payload_info.mass  # 简化的负载因子
                flexibility_torque *= load_factor
            
            compensation[i] = flexibility_torque
        
        return compensation
    
    def _compute_transmission_compensation(self, command: ControlCommand, 
                                        current_state: RobotState) -> Vector:
        """计算传动系统补偿"""
        compensation = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            # 传动系统非线性补偿
            gear_ratio = self.flexible_params.gear_ratio[i]
            compliance = self.flexible_params.transmission_compliance[i]
            
            # 计算传动误差
            if command.joint_torques is not None:
                transmission_error = compliance * command.joint_torques[i] / gear_ratio
                
                # 传动补偿力矩
                compensation[i] = -transmission_error * 1000.0  # 补偿增益
        
        return compensation
    
    def _compute_load_adaptive_compensation(self, command: ControlCommand,
                                          current_state: RobotState,
                                          payload_info: Optional[PayloadInfo]) -> Vector:
        """计算负载自适应补偿"""
        compensation = np.zeros(self.n_joints)
        
        if payload_info is None:
            return compensation
        
        # 根据负载信息调整补偿
        for i in range(self.n_joints):
            # 负载惯性补偿
            load_inertia_effect = payload_info.mass * 0.01  # 简化的惯性效应
            
            # 负载重心偏移补偿
            com_offset = np.linalg.norm(payload_info.center_of_mass)
            gravity_compensation = payload_info.mass * 9.81 * com_offset * 0.1
            
            # 组合负载补偿
            if current_state.joint_accelerations is not None:
                compensation[i] = (load_inertia_effect * current_state.joint_accelerations[i] + 
                                 gravity_compensation * np.sin(current_state.joint_positions[i]))
        
        return compensation
    
    def _update_virtual_sensor(self, sensor_data: EndEffectorSensorData) -> VirtualSensorState:
        """更新虚拟传感器状态"""
        kf = self.virtual_sensor['kalman_filter']
        
        # 预测步骤
        x_pred = kf['F'] @ kf['x']
        P_pred = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        # 构造观测向量 [position, acceleration]
        z = np.concatenate([sensor_data.position, sensor_data.acceleration])
        
        # 更新步骤
        innovation = z - kf['H'] @ x_pred
        S = kf['H'] @ P_pred @ kf['H'].T + kf['R']
        K = P_pred @ kf['H'].T @ np.linalg.inv(S)
        
        # 更新状态和协方差
        kf['x'] = x_pred + K @ innovation
        kf['P'] = (np.eye(len(kf['x'])) - K @ kf['H']) @ P_pred
        
        # 计算置信度（基于创新协方差）
        confidence = 1.0 / (1.0 + np.trace(S))
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # 提取估计的位置和速度
        estimated_position = kf['x'][::3]  # 每3个元素取一个（位置）
        estimated_velocity = kf['x'][1::3]  # 每3个元素取一个（速度）
        
        # 计算估计误差
        estimation_error = sensor_data.position - estimated_position
        
        return VirtualSensorState(
            estimated_position=estimated_position,
            estimated_velocity=estimated_velocity,
            estimation_error=estimation_error,
            confidence=confidence
        )
    
    def _compute_end_effector_feedback_force(self, position_error: Vector,
                                           velocity_error: Vector,
                                           sensor_data: EndEffectorSensorData) -> Vector:
        """计算末端执行器反馈控制力"""
        controller = self.end_effector_controller
        
        # PID控制
        proportional_force = controller['position_gains'] * position_error
        derivative_force = controller['velocity_gains'] * velocity_error
        
        # 积分项（防积分饱和）
        controller['integral_error'] += position_error * 0.001  # 假设1ms采样时间
        controller['integral_error'] = np.clip(
            controller['integral_error'],
            -controller['max_integral_error'],
            controller['max_integral_error']
        )
        integral_force = controller['integral_gains'] * controller['integral_error']
        
        # 力反馈补偿
        force_feedback = controller['force_gains'] * sensor_data.force
        
        # 组合控制力
        total_force = proportional_force + derivative_force + integral_force - force_feedback
        
        # 前馈补偿（如果启用）
        if controller['feedforward_compensation']:
            # 简单的重力前馈补偿
            gravity_feedforward = np.array([0.0, 0.0, -9.81])  # 假设负载1kg
            total_force += gravity_feedforward
        return total_force
    
    def _update_state_history(self, state: RobotState) -> None:
        """更新状态历史"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_state_history:
            self.state_history.pop(0)
    
    def set_end_effector_gains(self, position_gains: Vector, velocity_gains: Vector,
                              force_gains: Vector, integral_gains: Vector) -> None:
        """设置末端执行器控制增益"""
        self.end_effector_controller['position_gains'] = position_gains
        self.end_effector_controller['velocity_gains'] = velocity_gains
        self.end_effector_controller['force_gains'] = force_gains
        self.end_effector_controller['integral_gains'] = integral_gains
    
    def get_flexible_joint_state(self, joint_index: int) -> Dict[str, float]:
        """获取柔性关节状态估计"""
        if joint_index >= self.n_joints:
            raise ValueError(f"关节索引超出范围: {joint_index}")
        
        observer = self.flexible_joint_observer[f'joint_{joint_index}']
        return {
            'motor_angle': observer['state'][0],
            'link_angle': observer['state'][1],
            'motor_velocity': observer['state'][2],
            'link_velocity': observer['state'][3],
            'deflection': observer['state'][0] - observer['state'][1],
            'deflection_rate': observer['state'][2] - observer['state'][3]
        }
    
    def get_virtual_sensor_confidence(self) -> float:
        """获取虚拟传感器置信度"""
        # 基于卡尔曼滤波器的协方差矩阵计算置信度
        kf = self.virtual_sensor['kalman_filter']
        trace_P = np.trace(kf['P'])
        confidence = 1.0 / (1.0 + trace_P)
        return np.clip(confidence, 0.0, 1.0)
    
    def reset_integral_errors(self) -> None:
        """重置积分误差"""
        self.end_effector_controller['integral_error'] = np.zeros(3)
    
    def enable_adaptive_compensation(self, enable: bool = True) -> None:
        """启用/禁用自适应补偿"""
        self.end_effector_controller['feedforward_compensation'] = enable
    
    def get_compensation_diagnostics(self) -> Dict[str, Any]:
        """获取补偿算法诊断信息"""
        diagnostics = {
            'flexible_joint_states': {},
            'virtual_sensor_confidence': self.get_virtual_sensor_confidence(),
            'integral_errors': self.end_effector_controller['integral_error'].copy(),
            'buffer_sizes': {
                'input_buffer': len(self.input_buffer),
                'state_history': len(self.state_history)
            }
        }
        
        # 添加每个关节的柔性状态
        for i in range(self.n_joints):
            diagnostics['flexible_joint_states'][f'joint_{i}'] = self.get_flexible_joint_state(i)
        
        return diagnostics
    def _design_input_shaper(self) -> List[Tuple[float, float]]:
        """
        设计输入整形器
        
        Returns:
            脉冲序列 [(时间, 幅值), ...]
        """
        # 零振动输入整形器（ZV Shaper）
        impulses = []
        
        for i in range(self.n_joints):
            freq = self.natural_frequencies[i]
            zeta = self.damping_ratios[i]
            
            # 计算整形器参数
            wd = freq * np.sqrt(1 - zeta**2)  # 阻尼频率
            K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
            
            # ZV整形器脉冲
            A1 = 1 / (1 + K)
            A2 = K / (1 + K)
            t1 = 0.0
            t2 = np.pi / wd
            
            impulses.append([(t1, A1), (t2, A2)])
        
        return impulses
    
    def _apply_shaper_to_positions(self, positions: Vector) -> Vector:
        """
        对位置应用输入整形
        
        Args:
            positions: 原始位置
        
        Returns:
            整形后的位置
        """
        # 简化实现：应用低通滤波器
        if len(self.input_buffer) < 3:
            return positions
        
        # 使用最近3个输入的加权平均
        weights = np.array([0.25, 0.5, 0.25])
        shaped_positions = np.zeros_like(positions)
        
        for i, weight in enumerate(weights):
            if i < len(self.input_buffer):
                cmd = self.input_buffer[-(i+1)]
                if cmd.joint_positions is not None:
                    shaped_positions += weight * cmd.joint_positions
        
        return shaped_positions
    
    def update_flexible_parameters(self, new_params: FlexibleJointParameters) -> None:
        """更新柔性关节参数"""
        self.flexible_params = new_params
        # 重新初始化观测器
        self.flexible_joint_observer = self._initialize_flexible_observer()
    
    def set_end_effector_gains(self, position_gains: Vector, velocity_gains: Vector,
                              force_gains: Vector, integral_gains: Vector) -> None:
        """设置末端执行器控制增益"""
        self.end_effector_controller['position_gains'] = position_gains
        self.end_effector_controller['velocity_gains'] = velocity_gains
        self.end_effector_controller['force_gains'] = force_gains
        self.end_effector_controller['integral_gains'] = integral_gains
    
    def get_flexible_joint_state(self, joint_index: int) -> Dict[str, float]:
        """获取柔性关节状态估计"""
        if joint_index >= self.n_joints:
            raise ValueError(f"关节索引超出范围: {joint_index}")
        
        observer = self.flexible_joint_observer[f'joint_{joint_index}']
        return {
            'motor_angle': observer['state'][0],
            'link_angle': observer['state'][1],
            'motor_velocity': observer['state'][2],
            'link_velocity': observer['state'][3],
            'deflection': observer['state'][0] - observer['state'][1],
            'deflection_rate': observer['state'][2] - observer['state'][3]
        }
    
    def get_virtual_sensor_confidence(self) -> float:
        """获取虚拟传感器置信度"""
        # 基于卡尔曼滤波器的协方差矩阵计算置信度
        kf = self.virtual_sensor['kalman_filter']
        trace_P = np.trace(kf['P'])
        confidence = 1.0 / (1.0 + trace_P)
        return np.clip(confidence, 0.0, 1.0)
    
    def reset_integral_errors(self) -> None:
        """重置积分误差"""
        self.end_effector_controller['integral_error'] = np.zeros(3)
    
    def enable_adaptive_compensation(self, enable: bool = True) -> None:
        """启用/禁用自适应补偿"""
        self.end_effector_controller['feedforward_compensation'] = enable
    
    def get_compensation_diagnostics(self) -> Dict[str, Any]:
        """获取补偿算法诊断信息"""
        diagnostics = {
            'flexible_joint_states': {},
            'virtual_sensor_confidence': self.get_virtual_sensor_confidence(),
            'integral_errors': self.end_effector_controller['integral_error'].copy(),
            'buffer_sizes': {
                'input_buffer': len(self.input_buffer),
                'state_history': len(self.state_history)
            }
        }
        
        # 添加每个关节的柔性状态
        for i in range(self.n_joints):
            diagnostics['flexible_joint_states'][f'joint_{i}'] = self.get_flexible_joint_state(i)
        
        return diagnostics