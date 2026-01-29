"""
MJCF文件解析器

解析MuJoCo MJCF格式的机器人模型文件，提取机器人的几何、动力学参数。
专门针对ER15-1400机械臂模型进行优化。
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

from .types import DynamicsParameters, KinodynamicLimits


class MJCFParser:
    """MJCF文件解析器"""
    
    def __init__(self, mjcf_path: str):
        """
        初始化MJCF解析器
        
        Args:
            mjcf_path: MJCF文件路径
        """
        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF文件不存在: {mjcf_path}")
        
        # 解析XML文件
        self.tree = ET.parse(self.mjcf_path)
        self.root = self.tree.getroot()
        
        # 提取基本信息
        self.model_name = self.root.get('model', 'unknown')
        
        # 解析的数据
        self._joints = []
        self._bodies = []
        self._inertials = []
        
        self._parse_model()
    
    def _parse_model(self) -> None:
        """解析模型结构"""
        # 递归解析worldbody
        worldbody = self.root.find('worldbody')
        if worldbody is not None:
            self._parse_body(worldbody, parent_name='world')
    
    def _parse_body(self, body_element: ET.Element, parent_name: str) -> None:
        """
        递归解析body元素
        
        Args:
            body_element: body XML元素
            parent_name: 父body名称
        """
        body_name = body_element.get('name', f'body_{len(self._bodies)}')
        
        # 解析位置和姿态
        pos = self._parse_vector(body_element.get('pos', '0 0 0'))
        quat = self._parse_vector(body_element.get('quat', '1 0 0 0'))
        
        body_info = {
            'name': body_name,
            'parent': parent_name,
            'pos': pos,
            'quat': quat
        }
        self._bodies.append(body_info)
        
        # 解析关节
        for joint in body_element.findall('joint'):
            joint_info = self._parse_joint(joint, body_name)
            self._joints.append(joint_info)
        
        # 解析惯性参数
        inertial = body_element.find('inertial')
        if inertial is not None:
            inertial_info = self._parse_inertial(inertial, body_name)
            self._inertials.append(inertial_info)
        
        # 递归解析子body
        for child_body in body_element.findall('body'):
            self._parse_body(child_body, body_name)
    
    def _parse_joint(self, joint_element: ET.Element, body_name: str) -> Dict[str, Any]:
        """
        解析关节信息
        
        Args:
            joint_element: joint XML元素
            body_name: 所属body名称
        
        Returns:
            关节信息字典
        """
        joint_name = joint_element.get('name', f'joint_{len(self._joints)}')
        pos = self._parse_vector(joint_element.get('pos', '0 0 0'))
        axis = self._parse_vector(joint_element.get('axis', '0 0 1'))
        
        # 解析关节限制
        range_str = joint_element.get('range')
        joint_range = None
        if range_str:
            range_values = [float(x) for x in range_str.split()]
            if len(range_values) == 2:
                joint_range = (range_values[0], range_values[1])
        
        return {
            'name': joint_name,
            'body': body_name,
            'pos': pos,
            'axis': axis,
            'range': joint_range,
            'type': joint_element.get('type', 'hinge')  # 默认为铰链关节
        }
    
    def _parse_inertial(self, inertial_element: ET.Element, body_name: str) -> Dict[str, Any]:
        """
        解析惯性参数
        
        Args:
            inertial_element: inertial XML元素
            body_name: 所属body名称
        
        Returns:
            惯性参数字典
        """
        pos = self._parse_vector(inertial_element.get('pos', '0 0 0'))
        quat = self._parse_vector(inertial_element.get('quat', '1 0 0 0'))
        mass = float(inertial_element.get('mass', '1.0'))
        
        # 解析对角惯量
        diaginertia_str = inertial_element.get('diaginertia')
        diaginertia = None
        if diaginertia_str:
            diag_values = [float(x) for x in diaginertia_str.split()]
            if len(diag_values) == 3:
                diaginertia = np.array(diag_values)
        
        return {
            'body': body_name,
            'pos': pos,
            'quat': quat,
            'mass': mass,
            'diaginertia': diaginertia
        }
    
    def _parse_vector(self, vector_str: str) -> np.ndarray:
        """
        解析向量字符串
        
        Args:
            vector_str: 向量字符串，如 "1.0 2.0 3.0"
        
        Returns:
            numpy数组
        """
        return np.array([float(x) for x in vector_str.split()])
    
    def get_joint_count(self) -> int:
        """获取关节数量"""
        return len(self._joints)
    
    def get_joint_names(self) -> List[str]:
        """获取关节名称列表"""
        return [joint['name'] for joint in self._joints]
    
    def get_joint_limits(self) -> Tuple[List[float], List[float]]:
        """
        获取关节限制
        
        Returns:
            (最小位置列表, 最大位置列表)
        """
        min_positions = []
        max_positions = []
        
        for joint in self._joints:
            if joint['range']:
                min_positions.append(joint['range'][0])
                max_positions.append(joint['range'][1])
            else:
                # 默认限制
                min_positions.append(-np.pi)
                max_positions.append(np.pi)
        
        return min_positions, max_positions
    
    def extract_dynamics_parameters(self) -> DynamicsParameters:
        """
        提取动力学参数
        
        Returns:
            DynamicsParameters对象
        """
        n_joints = self.get_joint_count()
        
        # 提取质量
        masses = []
        centers_of_mass = []
        inertias = []
        
        # 按关节顺序组织惯性参数
        for i, joint in enumerate(self._joints):
            body_name = joint['body']
            
            # 查找对应的惯性参数
            inertial_info = None
            for inertial in self._inertials:
                if inertial['body'] == body_name:
                    inertial_info = inertial
                    break
            
            if inertial_info:
                masses.append(inertial_info['mass'])
                centers_of_mass.append(inertial_info['pos'].tolist())
                
                # 将对角惯量转换为3x3惯量矩阵
                if inertial_info['diaginertia'] is not None:
                    diag = inertial_info['diaginertia']
                    inertia_matrix = np.diag(diag).tolist()
                else:
                    # 默认惯量矩阵
                    inertia_matrix = np.eye(3).tolist()
                
                inertias.append(inertia_matrix)
            else:
                # 默认参数
                masses.append(1.0)
                centers_of_mass.append([0.0, 0.0, 0.0])
                inertias.append(np.eye(3).tolist())
        
        # 默认摩擦系数和重力
        friction_coeffs = [0.1] * n_joints  # 可以根据需要调整
        gravity = [0.0, 0.0, -9.81]
        
        return DynamicsParameters(
            masses=masses,
            centers_of_mass=centers_of_mass,
            inertias=inertias,
            friction_coeffs=friction_coeffs,
            gravity=gravity
        )
    
    def extract_kinodynamic_limits(self) -> KinodynamicLimits:
        """
        提取运动学动力学限制
        
        Returns:
            KinodynamicLimits对象
        """
        min_positions, max_positions = self.get_joint_limits()
        n_joints = len(min_positions)
        
        # ER15-1400的典型速度和加速度限制（基于工业机器人标准）
        max_velocities = [3.14, 2.5, 3.14, 4.0, 4.0, 6.0][:n_joints]  # rad/s
        max_accelerations = [15.0, 12.0, 15.0, 20.0, 20.0, 30.0][:n_joints]  # rad/s²
        max_jerks = [100.0, 80.0, 100.0, 150.0, 150.0, 200.0][:n_joints]  # rad/s³
        max_torques = [200.0, 180.0, 120.0, 80.0, 50.0, 30.0][:n_joints]  # Nm
        
        # 如果关节数不足，用默认值填充
        while len(max_velocities) < n_joints:
            max_velocities.append(2.0)
            max_accelerations.append(10.0)
            max_jerks.append(50.0)
            max_torques.append(50.0)
        
        return KinodynamicLimits(
            max_joint_positions=max_positions,
            min_joint_positions=min_positions,
            max_joint_velocities=max_velocities,
            max_joint_accelerations=max_accelerations,
            max_joint_jerks=max_jerks,
            max_joint_torques=max_torques
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息摘要
        
        Returns:
            模型信息字典
        """
        return {
            'name': self.model_name,
            'n_joints': self.get_joint_count(),
            'joint_names': self.get_joint_names(),
            'n_bodies': len(self._bodies),
            'mjcf_path': str(self.mjcf_path)
        }