"""
碰撞检测算法模块

实现基于距离的碰撞检测和避让策略，包括：
- 几何体碰撞检测
- 距离场计算
- 碰撞避让路径规划
- 实时碰撞监控
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time

from ..core.models import RobotModel
from ..core.types import RobotState, Vector, Pose, ControlCommand


class CollisionType(Enum):
    """碰撞类型"""
    SELF_COLLISION = "self_collision"      # 自碰撞
    ENVIRONMENT = "environment"            # 环境碰撞
    OBSTACLE = "obstacle"                  # 障碍物碰撞


@dataclass
class CollisionGeometry:
    """碰撞几何体"""
    name: str
    geometry_type: str  # "sphere", "cylinder", "box", "mesh"
    parameters: Dict[str, float]  # 几何参数
    pose: Pose  # 相对于连杆的位姿
    link_index: int  # 所属连杆索引


@dataclass
class CollisionPair:
    """碰撞对"""
    geometry1: CollisionGeometry
    geometry2: CollisionGeometry
    min_distance: float = 0.01  # 最小安全距离 [m]
    is_adjacent: bool = False   # 是否为相邻连杆


@dataclass
class CollisionInfo:
    """碰撞信息"""
    collision_type: CollisionType
    distance: float
    closest_points: Tuple[Vector, Vector]  # 最近点对
    collision_pair: CollisionPair
    severity: float  # 严重程度 [0-1]
    timestamp: float


@dataclass
class AvoidanceCommand:
    """避让指令"""
    joint_velocities: Vector
    joint_accelerations: Vector
    avoidance_force: Vector  # 避让力
    priority: float  # 优先级 [0-1]


class DistanceCalculator:
    """
    距离计算器
    
    实现各种几何体之间的距离计算算法。
    """
    
    @staticmethod
    def sphere_sphere_distance(
        center1: Vector, radius1: float,
        center2: Vector, radius2: float
    ) -> Tuple[float, Vector, Vector]:
        """
        计算两个球体之间的距离
        
        Args:
            center1, center2: 球心位置
            radius1, radius2: 半径
        
        Returns:
            (距离, 球1上最近点, 球2上最近点)
        """
        center_diff = center2 - center1
        center_distance = np.linalg.norm(center_diff)
        
        if center_distance < 1e-6:
            # 球心重合的特殊情况
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = center_diff / center_distance
        
        point1 = center1 + radius1 * direction
        point2 = center2 - radius2 * direction
        
        distance = max(0.0, center_distance - radius1 - radius2)
        
        return distance, point1, point2
    
    @staticmethod
    def sphere_cylinder_distance(
        sphere_center: Vector, sphere_radius: float,
        cylinder_start: Vector, cylinder_end: Vector, cylinder_radius: float
    ) -> Tuple[float, Vector, Vector]:
        """
        计算球体与圆柱体之间的距离
        
        Args:
            sphere_center: 球心
            sphere_radius: 球半径
            cylinder_start, cylinder_end: 圆柱体两端点
            cylinder_radius: 圆柱体半径
        
        Returns:
            (距离, 球上最近点, 圆柱体上最近点)
        """
        # 计算球心到圆柱体轴线的最近点
        axis = cylinder_end - cylinder_start
        axis_length = np.linalg.norm(axis)
        
        if axis_length < 1e-6:
            # 退化为点的情况
            return DistanceCalculator.sphere_sphere_distance(
                sphere_center, sphere_radius,
                cylinder_start, cylinder_radius
            )
        
        axis_unit = axis / axis_length
        to_sphere = sphere_center - cylinder_start
        
        # 投影到轴线上
        projection_length = np.dot(to_sphere, axis_unit)
        projection_length = np.clip(projection_length, 0.0, axis_length)
        
        # 轴线上最近点
        axis_point = cylinder_start + projection_length * axis_unit
        
        # 计算距离
        to_axis = sphere_center - axis_point
        distance_to_axis = np.linalg.norm(to_axis)
        
        if distance_to_axis < 1e-6:
            # 球心在轴线上
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = to_axis / distance_to_axis
        
        # 最近点
        cylinder_surface = axis_point + cylinder_radius * direction
        sphere_surface = sphere_center - sphere_radius * direction
        
        distance = max(0.0, distance_to_axis - sphere_radius - cylinder_radius)
        
        return distance, sphere_surface, cylinder_surface
    
    @staticmethod
    def point_box_distance(
        point: Vector,
        box_center: Vector, box_dimensions: Vector
    ) -> Tuple[float, Vector]:
        """
        计算点到盒子的距离
        
        Args:
            point: 点坐标
            box_center: 盒子中心
            box_dimensions: 盒子尺寸 [长, 宽, 高]
        
        Returns:
            (距离, 盒子表面最近点)
        """
        # 转换到盒子坐标系
        relative_point = point - box_center
        half_dims = box_dimensions / 2.0
        
        # 计算到各个面的距离
        clamped_point = np.clip(relative_point, -half_dims, half_dims)
        
        # 最近点
        closest_point = box_center + clamped_point
        
        # 距离
        distance = np.linalg.norm(relative_point - clamped_point)
        
        return distance, closest_point


class CollisionDetector:
    """
    碰撞检测器
    
    实现基于距离的碰撞检测算法。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化碰撞检测器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        
        # 检测参数
        self.min_safe_distance = 0.05  # 最小安全距离 [m]
        self.warning_distance = 0.10   # 警告距离 [m]
        self.critical_distance = 0.02  # 临界距离 [m]
        
        # 初始化几何体和碰撞对
        self.collision_geometries = self._initialize_collision_geometries()
        self.collision_pairs = self._initialize_collision_pairs()
        self.distance_calculator = DistanceCalculator()
        
        # 性能优化
        self.last_check_time = 0.0
        self.check_interval = 0.001  # 检查间隔 [s]
        self.distance_cache = {}
    
    def _initialize_collision_geometries(self) -> List[CollisionGeometry]:
        """初始化碰撞几何体"""
        geometries = []
        
        # 为每个连杆创建简化的碰撞几何体
        for i in range(self.robot_model.n_joints):
            # 连杆主体（圆柱体）
            link_geometry = CollisionGeometry(
                name=f"link_{i}",
                geometry_type="cylinder",
                parameters={
                    "radius": 0.05,  # 5cm 半径
                    "length": 0.20   # 20cm 长度
                },
                pose=Pose(
                    position=np.array([0.0, 0.0, 0.1]),
                    orientation=np.eye(3)
                ),
                link_index=i
            )
            geometries.append(link_geometry)
            
            # 关节（球体）
            joint_geometry = CollisionGeometry(
                name=f"joint_{i}",
                geometry_type="sphere",
                parameters={"radius": 0.06},  # 6cm 半径
                pose=Pose(
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.eye(3)
                ),
                link_index=i
            )
            geometries.append(joint_geometry)
        
        return geometries
    
    def _initialize_collision_pairs(self) -> List[CollisionPair]:
        """初始化碰撞对"""
        pairs = []
        
        # 生成所有可能的碰撞对
        for i, geom1 in enumerate(self.collision_geometries):
            for j, geom2 in enumerate(self.collision_geometries):
                if i >= j:
                    continue
                
                # 检查是否为相邻连杆
                is_adjacent = abs(geom1.link_index - geom2.link_index) <= 1
                
                # 相邻连杆使用更小的安全距离
                min_distance = 0.01 if is_adjacent else self.min_safe_distance
                
                pair = CollisionPair(
                    geometry1=geom1,
                    geometry2=geom2,
                    min_distance=min_distance,
                    is_adjacent=is_adjacent
                )
                pairs.append(pair)
        
        return pairs
    
    def check_collisions(self, robot_state: RobotState) -> List[CollisionInfo]:
        """
        检查碰撞
        
        Args:
            robot_state: 机器人状态
        
        Returns:
            碰撞信息列表
        """
        current_time = time.time()
        
        # 性能优化：限制检查频率
        if current_time - self.last_check_time < self.check_interval:
            return []
        
        self.last_check_time = current_time
        
        collisions = []
        
        # 计算正向运动学
        link_poses = self._compute_link_poses(robot_state)
        
        # 检查所有碰撞对
        for pair in self.collision_pairs:
            collision_info = self._check_collision_pair(pair, link_poses, current_time)
            if collision_info:
                collisions.append(collision_info)
        
        return collisions
    
    def _compute_link_poses(self, robot_state: RobotState) -> List[Pose]:
        """
        计算连杆位姿
        
        Args:
            robot_state: 机器人状态
        
        Returns:
            连杆位姿列表
        """
        # 使用动力学引擎计算正向运动学
        try:
            from ..algorithms.dynamics import DynamicsEngine
            dynamics = DynamicsEngine(self.robot_model)
            
            # 计算每个连杆的位姿
            link_poses = []
            for i in range(self.robot_model.n_joints):
                # 简化实现：使用关节角度计算近似位姿
                position = np.array([
                    i * 0.2 * np.cos(robot_state.joint_positions[i]),
                    i * 0.2 * np.sin(robot_state.joint_positions[i]),
                    i * 0.15
                ])
                
                # 简化的旋转矩阵
                angle = robot_state.joint_positions[i]
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                
                pose = Pose(position=position, orientation=rotation)
                link_poses.append(pose)
            
            return link_poses
            
        except Exception:
            # 回退到简化计算
            return [Pose(np.zeros(3), np.eye(3)) for _ in range(self.robot_model.n_joints)]
    
    def _check_collision_pair(
        self,
        pair: CollisionPair,
        link_poses: List[Pose],
        timestamp: float
    ) -> Optional[CollisionInfo]:
        """
        检查碰撞对
        
        Args:
            pair: 碰撞对
            link_poses: 连杆位姿
            timestamp: 时间戳
        
        Returns:
            碰撞信息（如果有碰撞）
        """
        # 获取几何体的世界坐标
        geom1_pose = self._get_geometry_world_pose(pair.geometry1, link_poses)
        geom2_pose = self._get_geometry_world_pose(pair.geometry2, link_poses)
        
        # 计算距离
        distance, point1, point2 = self._calculate_geometry_distance(
            pair.geometry1, geom1_pose,
            pair.geometry2, geom2_pose
        )
        
        # 检查是否存在碰撞风险
        if distance < self.warning_distance:
            severity = self._calculate_severity(distance, pair.min_distance)
            
            return CollisionInfo(
                collision_type=CollisionType.SELF_COLLISION,
                distance=distance,
                closest_points=(point1, point2),
                collision_pair=pair,
                severity=severity,
                timestamp=timestamp
            )
        
        return None
    
    def _get_geometry_world_pose(
        self,
        geometry: CollisionGeometry,
        link_poses: List[Pose]
    ) -> Pose:
        """
        获取几何体的世界坐标位姿
        
        Args:
            geometry: 碰撞几何体
            link_poses: 连杆位姿
        
        Returns:
            世界坐标位姿
        """
        link_pose = link_poses[geometry.link_index]
        
        # 变换到世界坐标系
        world_position = (
            link_pose.position +
            link_pose.orientation @ geometry.pose.position
        )
        
        world_orientation = link_pose.orientation @ geometry.pose.orientation
        
        return Pose(position=world_position, orientation=world_orientation)
    
    def _calculate_geometry_distance(
        self,
        geom1: CollisionGeometry, pose1: Pose,
        geom2: CollisionGeometry, pose2: Pose
    ) -> Tuple[float, Vector, Vector]:
        """
        计算两个几何体之间的距离
        
        Args:
            geom1, geom2: 几何体
            pose1, pose2: 位姿
        
        Returns:
            (距离, 几何体1上最近点, 几何体2上最近点)
        """
        if geom1.geometry_type == "sphere" and geom2.geometry_type == "sphere":
            return self.distance_calculator.sphere_sphere_distance(
                pose1.position, geom1.parameters["radius"],
                pose2.position, geom2.parameters["radius"]
            )
        
        elif geom1.geometry_type == "sphere" and geom2.geometry_type == "cylinder":
            # 计算圆柱体的端点
            length = geom2.parameters["length"]
            start = pose2.position - pose2.orientation @ np.array([0, 0, length/2])
            end = pose2.position + pose2.orientation @ np.array([0, 0, length/2])
            
            return self.distance_calculator.sphere_cylinder_distance(
                pose1.position, geom1.parameters["radius"],
                start, end, geom2.parameters["radius"]
            )
        
        elif geom1.geometry_type == "cylinder" and geom2.geometry_type == "sphere":
            # 交换参数
            length = geom1.parameters["length"]
            start = pose1.position - pose1.orientation @ np.array([0, 0, length/2])
            end = pose1.position + pose1.orientation @ np.array([0, 0, length/2])
            
            distance, point2, point1 = self.distance_calculator.sphere_cylinder_distance(
                pose2.position, geom2.parameters["radius"],
                start, end, geom1.parameters["radius"]
            )
            return distance, point1, point2
        
        else:
            # 简化处理：使用包围球
            radius1 = self._get_bounding_sphere_radius(geom1)
            radius2 = self._get_bounding_sphere_radius(geom2)
            
            return self.distance_calculator.sphere_sphere_distance(
                pose1.position, radius1,
                pose2.position, radius2
            )
    
    def _get_bounding_sphere_radius(self, geometry: CollisionGeometry) -> float:
        """获取包围球半径"""
        if geometry.geometry_type == "sphere":
            return geometry.parameters["radius"]
        elif geometry.geometry_type == "cylinder":
            radius = geometry.parameters["radius"]
            length = geometry.parameters["length"]
            return np.sqrt(radius**2 + (length/2)**2)
        elif geometry.geometry_type == "box":
            dims = geometry.parameters.get("dimensions", [0.1, 0.1, 0.1])
            return np.linalg.norm(dims) / 2
        else:
            return 0.1  # 默认值
    
    def _calculate_severity(self, distance: float, min_distance: float) -> float:
        """
        计算碰撞严重程度
        
        Args:
            distance: 当前距离
            min_distance: 最小安全距离
        
        Returns:
            严重程度 [0-1]
        """
        if distance <= min_distance:
            return 1.0  # 最严重
        elif distance >= self.warning_distance:
            return 0.0  # 无风险
        else:
            # 线性插值
            return 1.0 - (distance - min_distance) / (self.warning_distance - min_distance)


class CollisionAvoidance:
    """
    碰撞避让控制器
    
    实现基于人工势场的碰撞避让算法。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化碰撞避让控制器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        
        # 避让参数
        self.repulsive_gain = 1.0      # 排斥力增益
        self.attractive_gain = 0.5     # 吸引力增益
        self.influence_distance = 0.2  # 影响距离 [m]
        self.max_avoidance_velocity = 0.5  # 最大避让速度 [rad/s]
        
        # 滤波器参数
        self.velocity_filter_alpha = 0.8  # 速度滤波系数
        self.last_avoidance_velocity = np.zeros(robot_model.n_joints)
    
    def compute_avoidance_command(
        self,
        collisions: List[CollisionInfo],
        current_state: RobotState,
        desired_command: ControlCommand
    ) -> AvoidanceCommand:
        """
        计算避让指令
        
        Args:
            collisions: 碰撞信息列表
            current_state: 当前状态
            desired_command: 期望控制指令
        
        Returns:
            避让指令
        """
        if not collisions:
            # 无碰撞，返回零避让指令
            return AvoidanceCommand(
                joint_velocities=np.zeros(self.robot_model.n_joints),
                joint_accelerations=np.zeros(self.robot_model.n_joints),
                avoidance_force=np.zeros(self.robot_model.n_joints),
                priority=0.0
            )
        
        # 计算总的避让力
        total_avoidance_force = np.zeros(self.robot_model.n_joints)
        max_severity = 0.0
        
        for collision in collisions:
            avoidance_force = self._compute_repulsive_force(collision, current_state)
            total_avoidance_force += avoidance_force
            max_severity = max(max_severity, collision.severity)
        
        # 计算避让速度
        avoidance_velocity = self._force_to_velocity(
            total_avoidance_force, current_state
        )
        
        # 应用速度滤波
        filtered_velocity = (
            self.velocity_filter_alpha * self.last_avoidance_velocity +
            (1 - self.velocity_filter_alpha) * avoidance_velocity
        )
        self.last_avoidance_velocity = filtered_velocity
        
        # 限制避让速度
        max_vel = np.full(self.robot_model.n_joints, self.max_avoidance_velocity)
        limited_velocity = np.clip(filtered_velocity, -max_vel, max_vel)
        
        # 计算避让加速度
        dt = 0.001  # 假设控制周期
        avoidance_acceleration = (limited_velocity - current_state.joint_velocities) / dt
        
        return AvoidanceCommand(
            joint_velocities=limited_velocity,
            joint_accelerations=avoidance_acceleration,
            avoidance_force=total_avoidance_force,
            priority=max_severity
        )
    
    def _compute_repulsive_force(
        self,
        collision: CollisionInfo,
        current_state: RobotState
    ) -> Vector:
        """
        计算排斥力
        
        Args:
            collision: 碰撞信息
            current_state: 当前状态
        
        Returns:
            关节空间的排斥力
        """
        # 计算笛卡尔空间的排斥力
        distance = collision.distance
        severity = collision.severity
        
        # 排斥力方向（从碰撞点指向远离方向）
        point1, point2 = collision.closest_points
        direction = point1 - point2
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            # 避免除零
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / direction_norm
        
        # 排斥力大小（距离越近，力越大）
        if distance < 1e-6:
            force_magnitude = self.repulsive_gain * 1000  # 很大的力
        else:
            force_magnitude = self.repulsive_gain * severity / distance
        
        cartesian_force = force_magnitude * direction
        
        # 转换到关节空间
        joint_force = self._cartesian_to_joint_force(
            cartesian_force, point1, current_state
        )
        
        return joint_force
    
    def _cartesian_to_joint_force(
        self,
        cartesian_force: Vector,
        contact_point: Vector,
        current_state: RobotState
    ) -> Vector:
        """
        将笛卡尔空间力转换为关节空间力
        
        Args:
            cartesian_force: 笛卡尔空间力
            contact_point: 接触点
            current_state: 当前状态
        
        Returns:
            关节空间力
        """
        try:
            # 使用雅可比矩阵转换
            from ..algorithms.dynamics import DynamicsEngine
            dynamics = DynamicsEngine(self.robot_model)
            
            # 计算雅可比矩阵（简化实现）
            jacobian = self._compute_simplified_jacobian(current_state, contact_point)
            
            # 转换力
            joint_force = jacobian.T @ cartesian_force
            
            return joint_force
            
        except Exception:
            # 回退到简化计算
            return np.random.normal(0, 0.1, self.robot_model.n_joints)
    
    def _compute_simplified_jacobian(
        self,
        current_state: RobotState,
        point: Vector
    ) -> np.ndarray:
        """
        计算简化的雅可比矩阵
        
        Args:
            current_state: 当前状态
            point: 目标点
        
        Returns:
            雅可比矩阵 (3 x n_joints)
        """
        n_joints = self.robot_model.n_joints
        jacobian = np.zeros((3, n_joints))
        
        # 简化的雅可比计算
        for i in range(n_joints):
            # 假设每个关节对末端位置的影响
            jacobian[:, i] = np.array([
                -0.2 * np.sin(current_state.joint_positions[i]),
                0.2 * np.cos(current_state.joint_positions[i]),
                0.1
            ])
        
        return jacobian
    
    def _force_to_velocity(
        self,
        force: Vector,
        current_state: RobotState
    ) -> Vector:
        """
        将力转换为速度
        
        Args:
            force: 关节空间力
            current_state: 当前状态
        
        Returns:
            关节速度
        """
        # 简化的力到速度转换
        # 使用阻尼系数
        damping = 10.0
        velocity = force / damping
        
        return velocity


class CollisionMonitor:
    """
    碰撞监控器
    
    集成碰撞检测和避让功能，提供统一的碰撞监控接口。
    """
    
    def __init__(self, robot_model: RobotModel):
        """
        初始化碰撞监控器
        
        Args:
            robot_model: 机器人模型
        """
        self.robot_model = robot_model
        self.collision_detector = CollisionDetector(robot_model)
        self.collision_avoidance = CollisionAvoidance(robot_model)
        
        # 监控状态
        self.is_enabled = True
        self.collision_history = []
        self.max_history_length = 100
        
        # 统计信息
        self.total_collisions_detected = 0
        self.total_avoidance_actions = 0
    
    def update(
        self,
        current_state: RobotState,
        desired_command: ControlCommand
    ) -> Tuple[List[CollisionInfo], Optional[AvoidanceCommand]]:
        """
        更新碰撞监控
        
        Args:
            current_state: 当前状态
            desired_command: 期望控制指令
        
        Returns:
            (碰撞信息列表, 避让指令)
        """
        if not self.is_enabled:
            return [], None
        
        # 检测碰撞
        collisions = self.collision_detector.check_collisions(current_state)
        
        # 更新统计
        if collisions:
            self.total_collisions_detected += len(collisions)
        
        # 记录历史
        self.collision_history.extend(collisions)
        if len(self.collision_history) > self.max_history_length:
            self.collision_history = self.collision_history[-self.max_history_length:]
        
        # 计算避让指令
        avoidance_command = None
        if collisions:
            avoidance_command = self.collision_avoidance.compute_avoidance_command(
                collisions, current_state, desired_command
            )
            self.total_avoidance_actions += 1
        
        return collisions, avoidance_command
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """
        获取碰撞统计信息
        
        Returns:
            统计信息字典
        """
        recent_collisions = [
            c for c in self.collision_history
            if time.time() - c.timestamp < 10.0  # 最近10秒
        ]
        
        return {
            "total_collisions_detected": self.total_collisions_detected,
            "total_avoidance_actions": self.total_avoidance_actions,
            "recent_collisions_count": len(recent_collisions),
            "collision_history_length": len(self.collision_history),
            "is_enabled": self.is_enabled,
            "average_collision_severity": (
                np.mean([c.severity for c in recent_collisions])
                if recent_collisions else 0.0
            )
        }
    
    def enable_monitoring(self) -> None:
        """启用碰撞监控"""
        self.is_enabled = True
    
    def disable_monitoring(self) -> None:
        """禁用碰撞监控"""
        self.is_enabled = False
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_collisions_detected = 0
        self.total_avoidance_actions = 0
        self.collision_history.clear()