import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';
import { useAppStore } from '@/stores/appStore';
import { getRobotPreset } from '@/utils/robotPresets';
import ER15RobotModel from './ER15RobotModel';

interface RobotViewerProps {
  robot: MiniRobot;
}

// 智能机械臂模型组件 - 根据DH参数判断机械臂类型
const SmartRobotModel: React.FC<{ robot: MiniRobot }> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  // 判断是否为ER15-1400 (基于特征DH参数)
  const isER15 = robot.dhParams.length === 6 && 
                 robot.dhParams[0].d === 430 && 
                 robot.dhParams[1].a === 180 &&
                 robot.dhParams[2].a === 580;

  if (isER15) {
    return <ER15RobotModel robot={robot} />;
  }

  // 默认通用机械臂模型
  return <GenericRobotModel robot={robot} />;
};

// 通用机械臂模型组件
const GenericRobotModel: React.FC<{ robot: MiniRobot }> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  // 简化的机械臂渲染 - 使用基本几何体表示各关节
  const renderJoint = (index: number, position: [number, number, number]) => {
    return (
      <group key={index} position={position}>
        {/* 关节 */}
        <mesh>
          <cylinderGeometry args={[20, 20, 40]} />
          <meshStandardMaterial color={index === 0 ? '#ff6b6b' : '#4ecdc4'} />
        </mesh>
        
        {/* 连杆 */}
        {index < 5 && (
          <mesh position={[0, 30, 0]}>
            <boxGeometry args={[15, 60, 15]} />
            <meshStandardMaterial color="#95a5a6" />
          </mesh>
        )}
        
        {/* 关节标签 */}
        <mesh position={[0, 0, 25]}>
          <sphereGeometry args={[5]} />
          <meshBasicMaterial color="#e74c3c" />
        </mesh>
      </group>
    );
  };

  // 根据DH参数计算关节位置 (简化版)
  const calculateJointPositions = () => {
    const positions: [number, number, number][] = [];
    let z = 0;
    
    robot.dhParams.forEach((param, index) => {
      positions.push([0, 0, z]);
      z += param.d + 50; // 简化的高度计算
    });
    
    return positions;
  };

  const jointPositions = calculateJointPositions();

  return (
    <group ref={groupRef}>
      {/* 基座 */}
      <mesh position={[0, 0, -20]}>
        <cylinderGeometry args={[40, 40, 40]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>
      
      {/* 各关节 */}
      {jointPositions.map((position, index) => 
        renderJoint(index, position)
      )}
    </group>
  );
};

const RobotViewer: React.FC<RobotViewerProps> = ({ robot }) => {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [300, 300, 300], fov: 60 }}
        style={{ background: '#f8f9fa' }}
      >
        {/* 光照 */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <pointLight position={[-10, -10, -5]} intensity={0.4} />
        
        {/* 控制器 */}
        <OrbitControls 
          enablePan 
          enableZoom 
          enableRotate 
          maxDistance={1000}
          minDistance={100}
        />
        
        {/* 坐标系和网格 */}
        <Grid 
          args={[500, 500]} 
          cellSize={50} 
          cellThickness={0.5} 
          cellColor="#bdc3c7"
          sectionSize={100}
          sectionThickness={1}
          sectionColor="#7f8c8d"
        />
        
        {/* 自定义坐标轴 */}
        <group>
          {/* X轴 - 红色 */}
          <mesh position={[50, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <cylinderGeometry args={[1, 1, 100]} />
            <meshBasicMaterial color="#ff0000" />
          </mesh>
          {/* Y轴 - 绿色 */}
          <mesh position={[0, 50, 0]}>
            <cylinderGeometry args={[1, 1, 100]} />
            <meshBasicMaterial color="#00ff00" />
          </mesh>
          {/* Z轴 - 蓝色 */}
          <mesh position={[0, 0, 50]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[1, 1, 100]} />
            <meshBasicMaterial color="#0000ff" />
          </mesh>
        </group>
        
        {/* 机械臂模型 */}
        <SmartRobotModel robot={robot} />
      </Canvas>
    </div>
  );
};

export default RobotViewer;
