"""
轨迹规划算法模块

实现时间最优路径参数化（TOPP）算法和S型插补算法。
提供高效的轨迹生成和优化功能。

主要功能：
- 时间最优路径参数化（TOPP）算法
- 七段式S型插补算法
- 自适应包络线调整
- 动力学约束处理
- 负载自适应优化
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import warnings

from ..core.models import RobotModel
from ..core.types import (
    Path, Trajectory, TrajectoryPoint, KinodynamicLimits, 
    Waypoint, PayloadInfo, Vector
)


@dataclass
class S7SegmentParameters:
    """七段式S型速度曲线参数"""
    T1: float  # 加加速度段时间
    T2: float  # 匀加速度段时间
    T3: float  # 减加速度段时间
    T4: float  # 匀速段时间
    T5: float  # 加减速度段时间
    T6: float  # 匀减速度段时间
    T7: float  # 减减速度段时间
    
    j_max: float  # 最大加加速度
    a_max: float  # 最大加速度
    v_max: float  # 最大速度
    
    @property
    def total_time(self) -> float:
        """总时间"""
        return self.T1 + self.T2 + self.T3 + self.T4 + self.T5 + self.T6 + self.T7
    
    def validate(self) -> bool:
        """验证参数有效性"""
        times = [self.T1, self.T2, self.T3, self.T4, self.T5, self.T6, self.T7]
        limits = [self.j_max, self.a_max, self.v_max]
        
        # 所有时间必须非负
        if any(t < 0 for t in times):
            return False
        
        # 所有限制必须为正
        if any(limit <= 0 for limit in limits):
            return False
        
        return True


class TrajectoryPlanner:
    """
    轨迹规划器
    
    实现TOPP算法和S型插补，生成时间最优和平滑的机器人轨迹。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化轨迹规划器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        self.n_joints = robot_model.n_joints
    
    def generate_topp_trajectory(
        self, 
        path: Path, 
        limits: KinodynamicLimits,
        payload: Optional[PayloadInfo] = None,
        adaptive_envelope: bool = True
    ) -> Trajectory:
        """
        生成时间最优轨迹（TOPP算法）
        
        实现基于动力学约束的时间最优路径参数化算法，
        支持自适应包络线调整和负载自适应优化。
        
        Args:
            path: 输入路径
            limits: 运动学动力学限制
            payload: 负载信息（可选）
            adaptive_envelope: 是否启用自适应包络线调整
        
        Returns:
            时间最优轨迹
        """
        if not path:
            return []
        
        if len(path) == 1:
            # 单点路径，返回静止轨迹
            return [TrajectoryPoint(
                position=path[0].position,
                velocity=np.zeros(self.n_joints),
                acceleration=np.zeros(self.n_joints),
                jerk=np.zeros(self.n_joints),
                time=0.0,
                path_parameter=0.0
            )]
        
        try:
            # 更新负载信息
            if payload:
                self.robot_model.update_payload(payload)
            
            # 1. 路径预处理和参数化
            parameterized_path = self._parameterize_path(path)
            
            # 2. 计算速度限制包络线
            velocity_limits = self._compute_velocity_limits(parameterized_path, limits)
            
            # 3. 自适应包络线调整
            if adaptive_envelope:
                velocity_limits = self._adaptive_envelope_adjustment(
                    parameterized_path, velocity_limits, limits
                )
            
            # 4. 执行TOPP算法
            trajectory = self._execute_topp_algorithm(
                parameterized_path, velocity_limits, limits
            )
            
            # 5. 轨迹后处理和验证
            trajectory = self._post_process_trajectory(trajectory, limits)
            
            return trajectory
            
        except Exception as e:
            warnings.warn(f"TOPP算法失败，使用备用轨迹生成: {e}")
            return self._generate_fallback_trajectory(path, limits)
    
    def interpolate_s7_trajectory(
        self, 
        path: Path,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        max_jerk: Optional[float] = None
    ) -> Trajectory:
        """
        七段式S型插补
        
        Args:
            path: 输入路径
            max_velocity: 最大速度限制（可选）
            max_acceleration: 最大加速度限制（可选）
            max_jerk: 最大加加速度限制（可选）
        
        Returns:
            S型插补轨迹
        """
        if not path:
            return []
        
        if len(path) == 1:
            # 单点路径，返回静止轨迹
            return [TrajectoryPoint(
                position=path[0].position,
                velocity=np.zeros(self.n_joints),
                acceleration=np.zeros(self.n_joints),
                jerk=np.zeros(self.n_joints),
                time=0.0,
                path_parameter=0.0
            )]
        
        # 使用机器人限制或用户指定的限制
        limits = self.robot_model.kinodynamic_limits
        
        # 确保用户指定的限制不超过机器人限制
        if max_velocity is not None:
            v_max = min(max_velocity, min(limits.max_joint_velocities))
        else:
            v_max = min(limits.max_joint_velocities)
            
        if max_acceleration is not None:
            a_max = min(max_acceleration, min(limits.max_joint_accelerations))
        else:
            a_max = min(limits.max_joint_accelerations)
            
        if max_jerk is not None:
            j_max = min(max_jerk, min(limits.max_joint_jerks))
        else:
            j_max = min(limits.max_joint_jerks)
        
        trajectory = []
        current_time = 0.0
        
        # 对每个路径段进行S型插补
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # 计算路径段轨迹
            segment_trajectory = self._interpolate_s7_segment(
                start_point, end_point, v_max, a_max, j_max
            )
            
            # 调整时间偏移以确保连续性
            if segment_trajectory:
                for point in segment_trajectory:
                    point.time += current_time
                
                # 添加到总轨迹
                if i == 0:
                    trajectory.extend(segment_trajectory)
                else:
                    # 跳过起始点以避免重复，但确保连续性
                    if len(trajectory) > 0 and len(segment_trajectory) > 0:
                        # 确保连接点的连续性
                        last_point = trajectory[-1]
                        first_point = segment_trajectory[0]
                        
                        # 检查位置连续性
                        if not np.allclose(last_point.position, first_point.position, atol=1e-6):
                            # 调整第一个点的位置以确保连续性
                            segment_trajectory[0].position = last_point.position.copy()
                    
                    trajectory.extend(segment_trajectory[1:])  # 跳过起始点
                
                # 更新当前时间
                if segment_trajectory:
                    current_time = segment_trajectory[-1].time
        
        # 更新路径参数
        self._update_path_parameters(trajectory)
        
        return trajectory
    
    def _interpolate_s7_segment(
        self,
        start_point: Waypoint,
        end_point: Waypoint,
        v_max: float,
        a_max: float,
        j_max: float
    ) -> Trajectory:
        """
        对单个路径段进行七段式S型插补
        
        Args:
            start_point: 起始点
            end_point: 结束点
            v_max: 最大速度
            a_max: 最大加速度
            j_max: 最大加加速度
        
        Returns:
            路径段轨迹
        """
        # 计算位移
        displacement = end_point.position - start_point.position
        distance = np.linalg.norm(displacement)
        
        if distance < 1e-6:
            # 距离太小，返回静止轨迹
            return [TrajectoryPoint(
                position=start_point.position,
                velocity=np.zeros(self.n_joints),
                acceleration=np.zeros(self.n_joints),
                jerk=np.zeros(self.n_joints),
                time=0.0,
                path_parameter=0.0
            )]
        
        # 计算七段式S型参数
        s7_params = self._calculate_s7_parameters(distance, v_max, a_max, j_max)
        
        if not s7_params.validate():
            warnings.warn("S型参数无效，使用简化轨迹")
            return self._generate_simple_trajectory(start_point, end_point)
        
        # 生成轨迹点
        trajectory = []
        dt = min(0.001, s7_params.total_time / 1000)  # 更密集的采样以提高平滑性
        t = 0.0
        
        # 添加起始点
        point = TrajectoryPoint(
            position=start_point.position.copy(),
            velocity=np.zeros(self.n_joints),
            acceleration=np.zeros(self.n_joints),
            jerk=np.zeros(self.n_joints),
            time=0.0,
            path_parameter=0.0
        )
        trajectory.append(point)
        
        # 生成中间点
        t = dt
        while t < s7_params.total_time:
            # 计算当前时刻的运动状态
            s, v, a, j = self._evaluate_s7_profile(t, s7_params)
            
            # 将标量运动状态映射到关节空间
            direction = displacement / distance
            position = start_point.position + (s / distance) * displacement
            velocity = v * direction
            acceleration = a * direction
            jerk_vec = j * direction
            
            # 确保路径参数在有效范围内
            path_param = min(max(s / distance, 0.0), 1.0) if distance > 0 else 0.0
            
            point = TrajectoryPoint(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk_vec,
                time=t,
                path_parameter=path_param
            )
            
            trajectory.append(point)
            t += dt
        
        # 确保最后一点的边界条件精确匹配
        if trajectory:
            last_point = TrajectoryPoint(
                position=end_point.position.copy(),
                velocity=np.zeros(self.n_joints),
                acceleration=np.zeros(self.n_joints),
                jerk=np.zeros(self.n_joints),
                time=s7_params.total_time,
                path_parameter=1.0
            )
            trajectory.append(last_point)
        
        return trajectory
    
    def _calculate_s7_parameters(
        self,
        distance: float,
        v_max: float,
        a_max: float,
        j_max: float
    ) -> S7SegmentParameters:
        """
        计算七段式S型速度曲线参数
        
        Args:
            distance: 总位移
            v_max: 最大速度
            a_max: 最大加速度
            j_max: 最大加加速度
        
        Returns:
            S型参数
        """
        # 确保参数合理性
        if distance <= 0 or v_max <= 0 or a_max <= 0 or j_max <= 0:
            return S7SegmentParameters(
                T1=0.1, T2=0.0, T3=0.1, T4=0.0, T5=0.1, T6=0.0, T7=0.1,
                j_max=j_max, a_max=a_max, v_max=v_max
            )
        
        # 计算理论最小时间
        t_acc_to_max = a_max / j_max  # 达到最大加速度所需时间
        
        # 检查是否能达到最大速度
        # 如果只用加加速度段，能达到的最大速度
        v_max_jerk_only = j_max * t_acc_to_max**2
        
        if v_max <= v_max_jerk_only:
            # 不需要匀加速度段，使用三角形速度曲线
            T1 = T3 = T5 = T7 = np.sqrt(v_max / j_max)
            T2 = T4 = T6 = 0.0
            
            # 检查位移是否匹配
            s_calculated = 4 * (j_max * T1**3 / 3)  # 四个加加速度段的总位移
            
            if s_calculated > distance:
                # 需要进一步降低速度
                # 求解: 4 * (j_max * T^3 / 3) = distance
                T1 = T3 = T5 = T7 = np.power(3 * distance / (4 * j_max), 1/3)
                v_max = j_max * T1**2
        else:
            # 需要匀加速度段
            T1 = T3 = T5 = T7 = t_acc_to_max
            
            # 计算匀加速度段时间
            v_at_end_of_T1 = j_max * T1**2
            if v_max > v_at_end_of_T1:
                T2 = T6 = (v_max - v_at_end_of_T1) / a_max
            else:
                T2 = T6 = 0.0
                # 重新计算T1以达到所需速度
                T1 = T3 = T5 = T7 = np.sqrt(v_max / j_max)
            
            # 计算加速和减速阶段的位移
            s_acc = (j_max * T1**3 / 6 + 
                    j_max * T1**2 * T2 / 2 + 
                    a_max * T2**2 / 2 +
                    v_max * T3 - 
                    j_max * T3**3 / 6)
            s_dec = s_acc  # 对称
            
            # 计算匀速段
            s_const = distance - s_acc - s_dec
            
            if s_const >= 0:
                T4 = s_const / v_max if v_max > 0 else 0.0
            else:
                # 无法达到最大速度，需要降低速度
                T4 = 0.0
                # 使用二分法求解合适的速度
                v_low, v_high = 0.0, v_max
                for _ in range(20):  # 最多20次迭代
                    v_test = (v_low + v_high) / 2
                    T1_test = T3_test = T5_test = T7_test = np.sqrt(v_test / j_max)
                    T2_test = T6_test = 0.0
                    
                    s_test = 4 * (j_max * T1_test**3 / 3)
                    
                    if abs(s_test - distance) < 1e-6:
                        break
                    elif s_test > distance:
                        v_high = v_test
                    else:
                        v_low = v_test
                
                v_max = (v_low + v_high) / 2
                T1 = T3 = T5 = T7 = np.sqrt(v_max / j_max)
                T2 = T6 = 0.0
        
        return S7SegmentParameters(
            T1=T1, T2=T2, T3=T3, T4=T4, T5=T5, T6=T6, T7=T7,
            j_max=j_max, a_max=a_max, v_max=v_max
        )
    
    def _evaluate_s7_profile(
        self,
        t: float,
        params: S7SegmentParameters
    ) -> Tuple[float, float, float, float]:
        """
        计算七段式S型曲线在时刻t的位置、速度、加速度和加加速度
        
        Args:
            t: 时间
            params: S型参数
        
        Returns:
            (位置, 速度, 加速度, 加加速度)
        """
        if t <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        if t >= params.total_time:
            # 计算总位移 - 使用精确公式
            total_displacement = self._calculate_total_displacement(params)
            return total_displacement, 0.0, 0.0, 0.0
        
        # 计算累积时间点
        t1 = params.T1
        t2 = t1 + params.T2
        t3 = t2 + params.T3
        t4 = t3 + params.T4
        t5 = t4 + params.T5
        t6 = t5 + params.T6
        t7 = t6 + params.T7
        
        # 预计算各段结束时的状态以确保连续性
        states = self._precompute_segment_states(params)
        
        if t <= t1:
            # 第一段：加加速度
            return self._evaluate_segment_1(t, params)
        elif t <= t2:
            # 第二段：匀加速度
            return self._evaluate_segment_2(t, t1, params, states[0])
        elif t <= t3:
            # 第三段：减加速度
            return self._evaluate_segment_3(t, t2, params, states[1])
        elif t <= t4:
            # 第四段：匀速
            return self._evaluate_segment_4(t, t3, params, states[2])
        elif t <= t5:
            # 第五段：加减速度（开始减速）
            return self._evaluate_segment_5(t, t4, params, states[3])
        elif t <= t6:
            # 第六段：匀减速度
            return self._evaluate_segment_6(t, t5, params, states[4])
        else:
            # 第七段：减减速度
            return self._evaluate_segment_7(t, t6, params, states[5])
    
    def _calculate_total_displacement(self, params: S7SegmentParameters) -> float:
        """计算S7曲线的总位移"""
        # 使用对称性和精确公式
        s_acc = (params.j_max * params.T1**3 / 6 + 
                params.j_max * params.T1**2 * params.T2 / 2 + 
                params.a_max * params.T2**2 / 2 +
                params.v_max * params.T3 - 
                params.j_max * params.T3**3 / 6)
        
        s_const = params.v_max * params.T4
        s_dec = s_acc  # 对称
        
        return s_acc + s_const + s_dec
    
    def _precompute_segment_states(self, params: S7SegmentParameters) -> List[Tuple[float, float, float]]:
        """预计算各段结束时的状态 (s, v, a)"""
        states = []
        
        # 段1结束
        s1 = params.j_max * params.T1**3 / 6
        v1 = params.j_max * params.T1**2 / 2
        a1 = params.j_max * params.T1
        states.append((s1, v1, a1))
        
        # 段2结束
        s2 = s1 + v1 * params.T2 + 0.5 * a1 * params.T2**2
        v2 = v1 + a1 * params.T2
        a2 = a1
        states.append((s2, v2, a2))
        
        # 段3结束
        s3 = s2 + v2 * params.T3 + 0.5 * a2 * params.T3**2 - params.j_max * params.T3**3 / 6
        v3 = v2 + a2 * params.T3 - 0.5 * params.j_max * params.T3**2
        a3 = a2 - params.j_max * params.T3
        states.append((s3, v3, a3))
        
        # 段4结束
        s4 = s3 + v3 * params.T4
        v4 = v3
        a4 = 0.0
        states.append((s4, v4, a4))
        
        # 段5结束
        s5 = s4 + v4 * params.T5 - params.j_max * params.T5**3 / 6
        v5 = v4 - 0.5 * params.j_max * params.T5**2
        a5 = -params.j_max * params.T5
        states.append((s5, v5, a5))
        
        # 段6结束
        s6 = s5 + v5 * params.T6 + 0.5 * a5 * params.T6**2
        v6 = v5 + a5 * params.T6
        a6 = a5
        states.append((s6, v6, a6))
        
        return states
    
    def _evaluate_segment_1(self, t: float, params: S7SegmentParameters) -> Tuple[float, float, float, float]:
        """第一段：加加速度"""
        j = params.j_max
        a = params.j_max * t
        v = 0.5 * params.j_max * t**2
        s = params.j_max * t**3 / 6
        return s, v, a, j
    
    def _evaluate_segment_2(self, t: float, t1: float, params: S7SegmentParameters, 
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第二段：匀加速度"""
        dt = t - t1
        s0, v0, a0 = prev_state
        
        j = 0.0
        a = a0
        v = v0 + a0 * dt
        s = s0 + v0 * dt + 0.5 * a0 * dt**2
        return s, v, a, j
    
    def _evaluate_segment_3(self, t: float, t2: float, params: S7SegmentParameters,
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第三段：减加速度"""
        dt = t - t2
        s0, v0, a0 = prev_state
        
        j = -params.j_max
        a = a0 - params.j_max * dt
        v = v0 + a0 * dt - 0.5 * params.j_max * dt**2
        s = s0 + v0 * dt + 0.5 * a0 * dt**2 - params.j_max * dt**3 / 6
        return s, v, a, j
    
    def _evaluate_segment_4(self, t: float, t3: float, params: S7SegmentParameters,
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第四段：匀速"""
        dt = t - t3
        s0, v0, a0 = prev_state
        
        j = 0.0
        a = 0.0
        v = v0  # 应该等于v_max
        s = s0 + v0 * dt
        return s, v, a, j
    
    def _evaluate_segment_5(self, t: float, t4: float, params: S7SegmentParameters,
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第五段：加减速度（开始减速）"""
        dt = t - t4
        s0, v0, a0 = prev_state
        
        j = -params.j_max
        a = -params.j_max * dt
        v = v0 - 0.5 * params.j_max * dt**2
        s = s0 + v0 * dt - params.j_max * dt**3 / 6
        return s, v, a, j
    
    def _evaluate_segment_6(self, t: float, t5: float, params: S7SegmentParameters,
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第六段：匀减速度"""
        dt = t - t5
        s0, v0, a0 = prev_state
        
        j = 0.0
        a = a0  # 应该等于-a_max
        v = v0 + a0 * dt
        s = s0 + v0 * dt + 0.5 * a0 * dt**2
        return s, v, a, j
    
    def _evaluate_segment_7(self, t: float, t6: float, params: S7SegmentParameters,
                           prev_state: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """第七段：减减速度"""
        dt = t - t6
        s0, v0, a0 = prev_state
        
        j = params.j_max
        a = a0 + params.j_max * dt
        v = v0 + a0 * dt + 0.5 * params.j_max * dt**2
        s = s0 + v0 * dt + 0.5 * a0 * dt**2 + params.j_max * dt**3 / 6
        return s, v, a, j
    
    def _generate_simple_trajectory(
        self,
        start_point: Waypoint,
        end_point: Waypoint,
        duration: float = 1.0
    ) -> Trajectory:
        """
        生成简单的线性轨迹（备用方案）
        
        Args:
            start_point: 起始点
            end_point: 结束点
            duration: 轨迹持续时间
        
        Returns:
            简单轨迹
        """
        trajectory = []
        dt = 0.01  # 10ms采样间隔
        n_points = int(duration / dt) + 1
        
        for i in range(n_points):
            t = i * dt
            s = t / duration if duration > 0 else 0.0
            
            position = (1 - s) * start_point.position + s * end_point.position
            velocity = (end_point.position - start_point.position) / duration if duration > 0 else np.zeros_like(start_point.position)
            
            point = TrajectoryPoint(
                position=position,
                velocity=velocity,
                acceleration=np.zeros_like(start_point.position),
                jerk=np.zeros_like(start_point.position),
                time=t,
                path_parameter=s
            )
            
            trajectory.append(point)
        
        return trajectory
    
    def _update_path_parameters(self, trajectory: Trajectory) -> None:
        """
        更新轨迹的路径参数
        
        Args:
            trajectory: 轨迹
        """
        if not trajectory:
            return
        
        # 计算累积距离
        cumulative_distance = 0.0
        distances = [0.0]  # 起始点距离为0
        
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1].position
            curr_pos = trajectory[i].position
            segment_distance = np.linalg.norm(curr_pos - prev_pos)
            cumulative_distance += segment_distance
            distances.append(cumulative_distance)
        
        # 归一化路径参数
        total_distance = cumulative_distance
        if total_distance > 0:
            for i, point in enumerate(trajectory):
                point.path_parameter = distances[i] / total_distance
        else:
            # 如果总距离为0，所有点的路径参数都为0
            for point in trajectory:
                point.path_parameter = 0.0
    
    def validate_trajectory_smoothness(
        self,
        trajectory: Trajectory,
        position_tolerance: float = 1e-3,
        velocity_tolerance: float = 1e-2,
        acceleration_tolerance: float = 1e-1
    ) -> Tuple[bool, List[str]]:
        """
        验证轨迹平滑性
        
        Args:
            trajectory: 待验证的轨迹
            position_tolerance: 位置连续性容差
            velocity_tolerance: 速度连续性容差
            acceleration_tolerance: 加速度连续性容差
        
        Returns:
            (是否平滑, 错误信息列表)
        """
        if len(trajectory) < 2:
            return True, []
        
        errors = []
        
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            dt = curr_point.time - prev_point.time
            if dt <= 0:
                errors.append(f"时间点{i}：时间不递增")
                continue
            
            # 检查位置连续性
            pos_diff = np.linalg.norm(curr_point.position - prev_point.position)
            expected_pos_diff = np.linalg.norm(prev_point.velocity) * dt
            if abs(pos_diff - expected_pos_diff) > position_tolerance:
                errors.append(f"时间点{i}：位置不连续，差异{abs(pos_diff - expected_pos_diff):.6f}")
            
            # 检查速度连续性
            vel_diff = np.linalg.norm(curr_point.velocity - prev_point.velocity)
            expected_vel_diff = np.linalg.norm(prev_point.acceleration) * dt
            if abs(vel_diff - expected_vel_diff) > velocity_tolerance:
                errors.append(f"时间点{i}：速度不连续，差异{abs(vel_diff - expected_vel_diff):.6f}")
            
            # 检查加速度连续性
            acc_diff = np.linalg.norm(curr_point.acceleration - prev_point.acceleration)
            expected_acc_diff = np.linalg.norm(prev_point.jerk) * dt
            if abs(acc_diff - expected_acc_diff) > acceleration_tolerance:
                errors.append(f"时间点{i}：加速度不连续，差异{abs(acc_diff - expected_acc_diff):.6f}")
        
        return len(errors) == 0, errors
    
    def compute_trajectory_metrics(self, trajectory: Trajectory) -> dict:
        """
        计算轨迹质量指标
        
        Args:
            trajectory: 轨迹
        
        Returns:
            轨迹指标字典
        """
        if not trajectory:
            return {}
        
        positions = np.array([p.position for p in trajectory])
        velocities = np.array([p.velocity for p in trajectory])
        accelerations = np.array([p.acceleration for p in trajectory])
        jerks = np.array([p.jerk for p in trajectory])
        
        metrics = {
            'total_time': trajectory[-1].time,
            'total_distance': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
            'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
            'max_acceleration': np.max(np.linalg.norm(accelerations, axis=1)),
            'max_jerk': np.max(np.linalg.norm(jerks, axis=1)),
            'avg_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
            'velocity_smoothness': np.std(np.linalg.norm(velocities, axis=1)),
            'acceleration_smoothness': np.std(np.linalg.norm(accelerations, axis=1))
        }
        
        return metrics
    
    def _parameterize_path(self, path: Path) -> List[Tuple[float, Vector, Vector]]:
        """
        路径参数化：将路径转换为参数化形式
        
        Args:
            path: 输入路径
        
        Returns:
            参数化路径 [(s, position, tangent), ...]
        """
        parameterized = []
        
        # 计算累积弧长
        cumulative_length = 0.0
        lengths = [0.0]
        
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(path[i].position - path[i-1].position)
            cumulative_length += segment_length
            lengths.append(cumulative_length)
        
        # 归一化路径参数
        total_length = cumulative_length
        if total_length < 1e-6:
            # 路径长度太小，返回单点
            return [(0.0, path[0].position, np.zeros(self.n_joints))]
        
        # 生成参数化路径点
        for i, waypoint in enumerate(path):
            s = lengths[i] / total_length
            
            # 计算切向量
            if i == 0:
                # 起始点：使用向前差分
                if len(path) > 1:
                    tangent = path[1].position - path[0].position
                else:
                    tangent = np.zeros(self.n_joints)
            elif i == len(path) - 1:
                # 终点：使用向后差分
                tangent = path[i].position - path[i-1].position
            else:
                # 中间点：使用中心差分
                tangent = (path[i+1].position - path[i-1].position) / 2.0
            
            # 归一化切向量
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                tangent = tangent / tangent_norm
            
            parameterized.append((s, waypoint.position, tangent))
        
        return parameterized
    
    def _compute_velocity_limits(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        limits: KinodynamicLimits
    ) -> List[float]:
        """
        计算速度限制包络线
        
        基于运动学和动力学约束计算每个路径点的最大允许速度。
        
        Args:
            parameterized_path: 参数化路径
            limits: 运动学动力学限制
        
        Returns:
            速度限制数组
        """
        velocity_limits = []
        
        for s, position, tangent in parameterized_path:
            # 1. 运动学速度限制
            kinematic_limit = self._compute_kinematic_velocity_limit(tangent, limits)
            
            # 2. 动力学速度限制（基于力矩约束）
            dynamic_limit = self._compute_dynamic_velocity_limit(position, tangent, limits)
            
            # 3. 曲率限制（防止过快转弯）
            curvature_limit = self._compute_curvature_velocity_limit(
                parameterized_path, s, limits
            )
            
            # 取最严格的限制
            max_velocity = min(kinematic_limit, dynamic_limit, curvature_limit)
            velocity_limits.append(max_velocity)
        
        return velocity_limits
    
    def _compute_kinematic_velocity_limit(
        self, 
        tangent: Vector, 
        limits: KinodynamicLimits
    ) -> float:
        """
        计算运动学速度限制
        
        Args:
            tangent: 路径切向量
            limits: 运动学限制
        
        Returns:
            最大允许速度
        """
        max_joint_velocities = np.array(limits.max_joint_velocities)
        
        # 避免除零
        tangent_abs = np.abs(tangent)
        valid_joints = tangent_abs > 1e-6
        
        if not np.any(valid_joints):
            return float('inf')  # 无运动，无限制
        
        # 计算每个关节的速度限制
        joint_limits = np.full(self.n_joints, float('inf'))
        joint_limits[valid_joints] = max_joint_velocities[valid_joints] / tangent_abs[valid_joints]
        
        return np.min(joint_limits)
    
    def _compute_dynamic_velocity_limit(
        self, 
        position: Vector, 
        tangent: Vector, 
        limits: KinodynamicLimits
    ) -> float:
        """
        计算动力学速度限制
        
        基于关节力矩限制和动力学模型计算速度限制。
        
        Args:
            position: 关节位置
            tangent: 路径切向量
            limits: 动力学限制
        
        Returns:
            最大允许速度
        """
        try:
            # 获取动力学引擎
            from .dynamics import DynamicsEngine
            dynamics = DynamicsEngine(self.robot_model)
            
            # 计算重力补偿
            gravity_torque = dynamics.gravity_compensation(position)
            
            # 可用力矩 = 最大力矩 - 重力补偿
            max_torques = np.array(limits.max_joint_torques)
            available_torque = max_torques - np.abs(gravity_torque)
            
            # 确保可用力矩为正
            available_torque = np.maximum(available_torque, 0.1 * max_torques)
            
            # 简化的动力学速度限制计算
            # 假设速度相关的力矩主要来自摩擦和科里奥利效应
            
            # 摩擦力矩估算
            friction_coeffs = np.array(dynamics.friction_coeffs)
            
            # 计算每个关节的速度限制
            tangent_abs = np.abs(tangent)
            valid_joints = tangent_abs > 1e-6
            
            if not np.any(valid_joints):
                return float('inf')
            
            joint_velocity_limits = np.full(self.n_joints, float('inf'))
            
            for i in range(self.n_joints):
                if valid_joints[i]:
                    # 简化模型：可用力矩 = 摩擦系数 * 速度 * 切向量
                    if friction_coeffs[i] > 1e-6:
                        max_vel = available_torque[i] / (friction_coeffs[i] * tangent_abs[i])
                        joint_velocity_limits[i] = max_vel
            
            return np.min(joint_velocity_limits)
            
        except Exception:
            # 如果动力学计算失败，使用保守的速度限制
            return min(limits.max_joint_velocities) * 0.5
    
    def _compute_curvature_velocity_limit(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        current_s: float, 
        limits: KinodynamicLimits
    ) -> float:
        """
        计算基于曲率的速度限制
        
        Args:
            parameterized_path: 参数化路径
            current_s: 当前路径参数
            limits: 运动学限制
        
        Returns:
            基于曲率的最大速度
        """
        # 找到当前点在路径中的位置
        current_idx = 0
        for i, (s, _, _) in enumerate(parameterized_path):
            if s >= current_s:
                current_idx = i
                break
        
        # 计算局部曲率
        curvature = self._estimate_path_curvature(parameterized_path, current_idx)
        
        if curvature < 1e-6:
            return float('inf')  # 直线段，无曲率限制
        
        # 基于加速度限制计算曲率速度限制
        # v^2 * κ ≤ a_max  =>  v ≤ sqrt(a_max / κ)
        max_acceleration = min(limits.max_joint_accelerations)
        curvature_velocity_limit = np.sqrt(max_acceleration / curvature)
        
        return curvature_velocity_limit
    
    def _estimate_path_curvature(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        idx: int
    ) -> float:
        """
        估算路径曲率
        
        Args:
            parameterized_path: 参数化路径
            idx: 当前点索引
        
        Returns:
            曲率值
        """
        if len(parameterized_path) < 3 or idx == 0 or idx >= len(parameterized_path) - 1:
            return 0.0  # 端点或路径太短，曲率为0
        
        # 使用三点法估算曲率
        prev_pos = parameterized_path[idx - 1][1]
        curr_pos = parameterized_path[idx][1]
        next_pos = parameterized_path[idx + 1][1]
        
        # 计算向量
        v1 = curr_pos - prev_pos
        v2 = next_pos - curr_pos
        
        # 计算曲率 κ = |v1 × v2| / |v1|^3
        # 对于高维空间，使用角度变化率近似
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0
        
        # 计算角度变化
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数值稳定性
        angle_change = np.arccos(cos_angle)
        
        # 曲率近似为角度变化率
        segment_length = (v1_norm + v2_norm) / 2.0
        curvature = angle_change / segment_length if segment_length > 1e-6 else 0.0
        
        return curvature
    
    def _adaptive_envelope_adjustment(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        velocity_limits: List[float], 
        limits: KinodynamicLimits
    ) -> List[float]:
        """
        自适应包络线调整
        
        根据系统状态和负载信息动态调整速度限制包络线。
        
        Args:
            parameterized_path: 参数化路径
            velocity_limits: 初始速度限制
            limits: 运动学动力学限制
        
        Returns:
            调整后的速度限制
        """
        adjusted_limits = velocity_limits.copy()
        
        # 1. 负载自适应调整
        if hasattr(self.robot_model, 'current_payload') and self.robot_model.current_payload:
            payload = self.robot_model.current_payload
            
            # 根据负载质量调整速度限制
            payload_factor = self._compute_payload_adjustment_factor(payload)
            adjusted_limits = [limit * payload_factor for limit in adjusted_limits]
        
        # 2. 路径复杂度自适应调整
        complexity_factors = self._compute_path_complexity_factors(parameterized_path)
        
        for i in range(len(adjusted_limits)):
            adjusted_limits[i] *= complexity_factors[i]
        
        # 3. 平滑处理，避免速度限制突变
        adjusted_limits = self._smooth_velocity_limits(adjusted_limits)
        
        # 4. 确保限制在合理范围内
        max_reasonable_velocity = min(limits.max_joint_velocities) * 0.8
        adjusted_limits = [min(limit, max_reasonable_velocity) for limit in adjusted_limits]
        
        return adjusted_limits
    
    def _compute_payload_adjustment_factor(self, payload: PayloadInfo) -> float:
        """
        计算负载调整因子
        
        Args:
            payload: 负载信息
        
        Returns:
            调整因子 (0.0 - 1.0)
        """
        # 基于负载质量的调整
        base_mass = 1.0  # kg，基准负载质量
        mass_factor = base_mass / (base_mass + payload.mass)
        
        # 基于识别置信度的调整
        confidence_factor = payload.identification_confidence
        
        # 综合调整因子
        adjustment_factor = mass_factor * confidence_factor
        
        # 确保在合理范围内
        return np.clip(adjustment_factor, 0.3, 1.0)
    
    def _compute_path_complexity_factors(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]]
    ) -> List[float]:
        """
        计算路径复杂度调整因子
        
        Args:
            parameterized_path: 参数化路径
        
        Returns:
            每个点的复杂度调整因子
        """
        factors = []
        
        for i in range(len(parameterized_path)):
            # 计算局部曲率
            curvature = self._estimate_path_curvature(parameterized_path, i)
            
            # 基于曲率的调整因子
            # 高曲率区域降低速度限制
            curvature_factor = 1.0 / (1.0 + 10.0 * curvature)
            
            # 计算方向变化率
            direction_change_factor = self._compute_direction_change_factor(
                parameterized_path, i
            )
            
            # 综合复杂度因子
            complexity_factor = min(curvature_factor, direction_change_factor)
            factors.append(complexity_factor)
        
        return factors
    
    def _compute_direction_change_factor(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        idx: int
    ) -> float:
        """
        计算方向变化调整因子
        
        Args:
            parameterized_path: 参数化路径
            idx: 当前点索引
        
        Returns:
            方向变化调整因子
        """
        if idx == 0 or idx >= len(parameterized_path) - 1:
            return 1.0
        
        prev_tangent = parameterized_path[idx - 1][2]
        curr_tangent = parameterized_path[idx][2]
        
        # 计算切向量的变化
        tangent_change = np.linalg.norm(curr_tangent - prev_tangent)
        
        # 基于变化率的调整因子
        change_factor = 1.0 / (1.0 + 5.0 * tangent_change)
        
        return change_factor
    
    def _smooth_velocity_limits(self, velocity_limits: List[float]) -> List[float]:
        """
        平滑速度限制，避免突变
        
        Args:
            velocity_limits: 原始速度限制
        
        Returns:
            平滑后的速度限制
        """
        if len(velocity_limits) < 3:
            return velocity_limits
        
        smoothed = velocity_limits.copy()
        
        # 使用简单的移动平均滤波
        window_size = min(5, len(velocity_limits) // 3)
        
        for i in range(len(velocity_limits)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(velocity_limits), i + window_size // 2 + 1)
            
            window_values = velocity_limits[start_idx:end_idx]
            smoothed[i] = np.mean(window_values)
        
        return smoothed
    
    def _execute_topp_algorithm(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        velocity_limits: List[float], 
        limits: KinodynamicLimits
    ) -> Trajectory:
        """
        执行TOPP算法核心计算
        
        使用动态规划方法求解时间最优路径参数化问题。
        
        Args:
            parameterized_path: 参数化路径
            velocity_limits: 速度限制包络线
            limits: 运动学动力学限制
        
        Returns:
            时间最优轨迹
        """
        n_points = len(parameterized_path)
        if n_points < 2:
            return []
        
        # 1. 前向积分：计算最大可达速度
        max_velocities = self._forward_integration(
            parameterized_path, velocity_limits, limits
        )
        
        # 2. 后向积分：确保能够停止
        feasible_velocities = self._backward_integration(
            parameterized_path, max_velocities, limits
        )
        
        # 3. 生成时间最优轨迹
        trajectory = self._generate_time_optimal_trajectory(
            parameterized_path, feasible_velocities, limits
        )
        
        # 4. 密化轨迹以提高平滑性
        if len(trajectory) > 1:
            trajectory = self._densify_trajectory(trajectory)
        
        return trajectory
    
    def _forward_integration(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        velocity_limits: List[float], 
        limits: KinodynamicLimits
    ) -> List[float]:
        """
        前向积分：从起点开始计算最大可达速度
        
        Args:
            parameterized_path: 参数化路径
            velocity_limits: 速度限制
            limits: 运动学动力学限制
        
        Returns:
            最大可达速度数组
        """
        n_points = len(parameterized_path)
        max_velocities = [0.0] * n_points
        
        # 起点速度为0
        max_velocities[0] = 0.0
        
        # 最大加速度
        max_acceleration = min(limits.max_joint_accelerations)
        
        for i in range(1, n_points):
            # 计算路径段长度
            prev_s = parameterized_path[i-1][0]
            curr_s = parameterized_path[i][0]
            ds = curr_s - prev_s
            
            if ds < 1e-6:
                max_velocities[i] = max_velocities[i-1]
                continue
            
            # 基于加速度限制的最大速度增长
            # v^2 = v0^2 + 2*a*ds
            prev_velocity = max_velocities[i-1]
            max_velocity_from_acceleration = np.sqrt(
                prev_velocity**2 + 2 * max_acceleration * ds
            )
            
            # 取速度限制和加速度限制的最小值
            max_velocities[i] = min(
                velocity_limits[i], 
                max_velocity_from_acceleration
            )
        
        return max_velocities
    
    def _backward_integration(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        max_velocities: List[float], 
        limits: KinodynamicLimits
    ) -> List[float]:
        """
        后向积分：从终点开始确保能够停止
        
        Args:
            parameterized_path: 参数化路径
            max_velocities: 前向积分得到的最大速度
            limits: 运动学动力学限制
        
        Returns:
            可行速度数组
        """
        n_points = len(parameterized_path)
        feasible_velocities = max_velocities.copy()
        
        # 终点速度为0
        feasible_velocities[-1] = 0.0
        
        # 最大减速度（负加速度）
        max_deceleration = min(limits.max_joint_accelerations)
        
        for i in range(n_points - 2, -1, -1):
            # 计算路径段长度
            curr_s = parameterized_path[i][0]
            next_s = parameterized_path[i+1][0]
            ds = next_s - curr_s
            
            if ds < 1e-6:
                continue
            
            # 基于减速度限制的最大速度
            # v^2 = v_next^2 + 2*a*ds
            next_velocity = feasible_velocities[i+1]
            max_velocity_from_deceleration = np.sqrt(
                next_velocity**2 + 2 * max_deceleration * ds
            )
            
            # 取前向积分结果和后向积分结果的最小值
            feasible_velocities[i] = min(
                feasible_velocities[i], 
                max_velocity_from_deceleration
            )
        
        return feasible_velocities
    
    def _generate_time_optimal_trajectory(
        self, 
        parameterized_path: List[Tuple[float, Vector, Vector]], 
        feasible_velocities: List[float], 
        limits: KinodynamicLimits
    ) -> Trajectory:
        """
        生成时间最优轨迹
        
        Args:
            parameterized_path: 参数化路径
            feasible_velocities: 可行速度
            limits: 运动学动力学限制
        
        Returns:
            时间最优轨迹
        """
        if not parameterized_path:
            return []
            
        trajectory = []
        current_time = 0.0
        
        # 首先计算所有时间步长
        time_steps = []
        for i in range(len(parameterized_path)):
            if i == 0:
                time_steps.append(0.0)
            else:
                dt = self._compute_time_step(
                    parameterized_path[i-1], parameterized_path[i], 
                    feasible_velocities[i-1], feasible_velocities[i]
                )
                time_steps.append(dt)
        
        # 生成轨迹点，确保平滑性
        for i in range(len(parameterized_path)):
            s, position, tangent = parameterized_path[i]
            velocity_magnitude = feasible_velocities[i]
            
            # 计算关节空间速度
            velocity = velocity_magnitude * tangent
            
            # 计算加速度 - 使用更平滑的方法
            if i == 0:
                acceleration = np.zeros(self.n_joints)
            elif i == len(parameterized_path) - 1:
                # 最后一点，确保速度为零
                prev_velocity = feasible_velocities[i-1] * parameterized_path[i-1][2]
                dt = time_steps[i]
                if dt > 1e-6:
                    acceleration = (np.zeros(self.n_joints) - prev_velocity) / dt
                else:
                    acceleration = np.zeros(self.n_joints)
            else:
                # 中间点，使用中心差分
                prev_velocity = feasible_velocities[i-1] * parameterized_path[i-1][2]
                next_velocity = feasible_velocities[i+1] * parameterized_path[i+1][2]
                dt_prev = time_steps[i]
                dt_next = time_steps[i+1] if i+1 < len(time_steps) else dt_prev
                
                if dt_prev + dt_next > 1e-6:
                    # 使用加权平均来计算加速度
                    acceleration = (next_velocity - prev_velocity) / (dt_prev + dt_next)
                else:
                    acceleration = np.zeros(self.n_joints)
            
            # 限制加速度幅值
            acc_magnitude = np.linalg.norm(acceleration)
            max_acc = min(limits.max_joint_accelerations)
            if acc_magnitude > max_acc:
                acceleration = acceleration * (max_acc / acc_magnitude)
            
            # 计算加加速度（简化为零，避免数值不稳定）
            jerk = np.zeros(self.n_joints)
            
            # 更新时间
            if i > 0:
                current_time += time_steps[i]
            
            # 创建轨迹点
            point = TrajectoryPoint(
                position=position.copy(),
                velocity=velocity.copy(),
                acceleration=acceleration.copy(),
                jerk=jerk.copy(),
                time=current_time,
                path_parameter=s
            )
            
            trajectory.append(point)
        
        return trajectory
    
    def _densify_trajectory(self, sparse_trajectory: Trajectory) -> Trajectory:
        """
        密化稀疏轨迹，在轨迹点之间插入更多点以提高平滑性
        
        Args:
            sparse_trajectory: 稀疏轨迹
        
        Returns:
            密化后的轨迹
        """
        if len(sparse_trajectory) < 2:
            return sparse_trajectory
        
        dense_trajectory = []
        
        for i in range(len(sparse_trajectory) - 1):
            current_point = sparse_trajectory[i]
            next_point = sparse_trajectory[i + 1]
            
            # 添加当前点
            dense_trajectory.append(current_point)
            
            # 计算时间间隔
            dt_total = next_point.time - current_point.time
            
            if dt_total > 0.01:  # 只有当时间间隔足够大时才插值
                # 计算需要插入的点数
                n_interpolate = max(1, int(dt_total / 0.005))  # 每5ms一个点
                
                for j in range(1, n_interpolate):
                    alpha = j / n_interpolate
                    t_interp = current_point.time + alpha * dt_total
                    
                    # 线性插值位置
                    pos_interp = (1 - alpha) * current_point.position + alpha * next_point.position
                    
                    # 线性插值速度
                    vel_interp = (1 - alpha) * current_point.velocity + alpha * next_point.velocity
                    
                    # 线性插值加速度
                    acc_interp = (1 - alpha) * current_point.acceleration + alpha * next_point.acceleration
                    
                    # 线性插值路径参数
                    path_param_interp = (1 - alpha) * current_point.path_parameter + alpha * next_point.path_parameter
                    
                    # 创建插值点
                    interp_point = TrajectoryPoint(
                        position=pos_interp,
                        velocity=vel_interp,
                        acceleration=acc_interp,
                        jerk=np.zeros(self.n_joints),
                        time=t_interp,
                        path_parameter=path_param_interp
                    )
                    
                    dense_trajectory.append(interp_point)
        
        # 添加最后一个点
        dense_trajectory.append(sparse_trajectory[-1])
        
        return dense_trajectory
    
    def _compute_time_step(
        self, 
        prev_point: Tuple[float, Vector, Vector], 
        curr_point: Tuple[float, Vector, Vector], 
        prev_velocity: float, 
        curr_velocity: float
    ) -> float:
        """
        计算时间步长
        
        Args:
            prev_point: 前一个路径点
            curr_point: 当前路径点
            prev_velocity: 前一点速度
            curr_velocity: 当前点速度
        
        Returns:
            时间步长
        """
        # 计算路径段长度
        ds = curr_point[0] - prev_point[0]
        
        if ds < 1e-6:
            return 1e-3  # 返回最小时间步长
        
        # 使用梯形积分计算时间（更精确）
        if prev_velocity < 1e-6 and curr_velocity < 1e-6:
            # 两点都是静止，使用最小速度
            return ds / 1e-3
        elif prev_velocity < 1e-6:
            # 从静止开始
            return 2.0 * ds / curr_velocity
        elif curr_velocity < 1e-6:
            # 到静止结束
            return 2.0 * ds / prev_velocity
        else:
            # 正常情况，使用梯形公式
            avg_velocity = (prev_velocity + curr_velocity) / 2.0
            return ds / avg_velocity
    
    def _post_process_trajectory(
        self, 
        trajectory: Trajectory, 
        limits: KinodynamicLimits
    ) -> Trajectory:
        """
        轨迹后处理和验证
        
        Args:
            trajectory: 原始轨迹
            limits: 运动学动力学限制
        
        Returns:
            后处理后的轨迹
        """
        if not trajectory:
            return trajectory
        
        # 1. 验证轨迹约束
        violations = self._check_trajectory_constraints(trajectory, limits)
        
        if violations:
            warnings.warn(f"轨迹约束违反: {violations}")
            # 可以选择修正或重新生成轨迹
        
        # 2. 平滑处理
        trajectory = self._smooth_trajectory(trajectory)
        
        # 3. 重新计算时间戳确保一致性
        trajectory = self._recalculate_timestamps(trajectory)
        
        return trajectory
    
    def _check_trajectory_constraints(
        self, 
        trajectory: Trajectory, 
        limits: KinodynamicLimits
    ) -> List[str]:
        """
        检查轨迹约束违反
        
        Args:
            trajectory: 轨迹
            limits: 约束限制
        
        Returns:
            违反约束的描述列表
        """
        violations = []
        
        max_velocities = np.array(limits.max_joint_velocities)
        max_accelerations = np.array(limits.max_joint_accelerations)
        
        for i, point in enumerate(trajectory):
            # 检查速度约束
            velocity_violations = np.abs(point.velocity) > max_velocities
            if np.any(velocity_violations):
                violations.append(f"点{i}: 速度约束违反")
            
            # 检查加速度约束
            acceleration_violations = np.abs(point.acceleration) > max_accelerations
            if np.any(acceleration_violations):
                violations.append(f"点{i}: 加速度约束违反")
        
        return violations
    
    def _smooth_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """
        轨迹平滑处理
        
        Args:
            trajectory: 原始轨迹
        
        Returns:
            平滑后的轨迹
        """
        if len(trajectory) < 3:
            return trajectory
        
        # 对速度和加速度进行平滑
        smoothed_trajectory = []
        
        for i, point in enumerate(trajectory):
            if i == 0 or i == len(trajectory) - 1:
                # 保持端点不变
                smoothed_trajectory.append(point)
            else:
                # 使用相邻点的平均值进行平滑
                prev_point = trajectory[i-1]
                next_point = trajectory[i+1]
                
                smoothed_velocity = (prev_point.velocity + point.velocity + next_point.velocity) / 3.0
                smoothed_acceleration = (prev_point.acceleration + point.acceleration + next_point.acceleration) / 3.0
                
                smoothed_point = TrajectoryPoint(
                    position=point.position,
                    velocity=smoothed_velocity,
                    acceleration=smoothed_acceleration,
                    jerk=point.jerk,
                    time=point.time,
                    path_parameter=point.path_parameter
                )
                
                smoothed_trajectory.append(smoothed_point)
        
        return smoothed_trajectory
    
    def _recalculate_timestamps(self, trajectory: Trajectory) -> Trajectory:
        """
        重新计算时间戳确保一致性
        
        Args:
            trajectory: 轨迹
        
        Returns:
            时间戳一致的轨迹
        """
        if len(trajectory) < 2:
            return trajectory
        
        # 重新计算时间戳
        trajectory[0].time = 0.0
        
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # 计算位移
            displacement = np.linalg.norm(curr_point.position - prev_point.position)
            
            # 计算平均速度
            avg_velocity = (np.linalg.norm(prev_point.velocity) + np.linalg.norm(curr_point.velocity)) / 2.0
            
            # 计算时间步长
            if avg_velocity > 1e-6:
                dt = displacement / avg_velocity
            else:
                dt = 0.001  # 默认时间步长
            
            curr_point.time = prev_point.time + dt
        
        return trajectory
    
    def _generate_fallback_trajectory(
        self, 
        path: Path, 
        limits: KinodynamicLimits
    ) -> Trajectory:
        """
        生成备用轨迹（当TOPP算法失败时）
        
        Args:
            path: 输入路径
            limits: 运动学动力学限制
        
        Returns:
            备用轨迹
        """
        # 使用S型插补作为备用方案
        return self.interpolate_s7_trajectory(
            path,
            max_velocity=min(limits.max_joint_velocities) * 0.5,
            max_acceleration=min(limits.max_joint_accelerations) * 0.5,
            max_jerk=min(limits.max_joint_jerks) * 0.5
        )