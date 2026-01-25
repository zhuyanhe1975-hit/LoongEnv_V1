import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Axes } from '@react-three/drei';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';

interface RobotViewerProps {
  robot: MiniRobot;
}

// 简化的机械臂模型组件
const RobotModel: React.FC<{ robot: MiniRobot }> = ({ robot }) => {
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
        <Axes scale={[100, 100, 100]} />
        
        {/* 机械臂模型 */}
        <RobotModel robot={robot} />
      </Canvas>
    </div>
  );
};

export default RobotViewer;
