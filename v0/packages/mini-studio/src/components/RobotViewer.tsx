import React, { useRef, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';
import ER15RobotModel from './ER15RobotModel';

interface RobotViewerProps {
  robot: MiniRobot;
}

// ER15-1400专用机械臂模型组件
const RobotModel: React.FC<{ robot: MiniRobot }> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  // 直接使用ER15-1400模型
  return (
    <group ref={groupRef}>
      <ER15RobotModel robot={robot} />
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
        <Suspense fallback={null}>
          <RobotModel robot={robot} />
        </Suspense>
      </Canvas>
    </div>
  );
};

export default RobotViewer;
