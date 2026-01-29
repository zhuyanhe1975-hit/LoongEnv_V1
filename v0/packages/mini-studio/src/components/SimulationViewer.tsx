import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { TrajectoryPoint } from '@/types';

interface SimulationViewerProps {
  trajectoryPoints: TrajectoryPoint[];
  currentTime: number;
  totalTime: number;
  isRunning: boolean;
}

// 动画机械臂组件
const AnimatedRobot: React.FC<{
  currentPosition: [number, number, number];
  targetPosition: [number, number, number];
}> = ({ currentPosition, targetPosition }) => {
  const groupRef = useRef<THREE.Group>(null);

  // 简化的机械臂渲染
  return (
    <group 
      ref={groupRef} 
      position={[currentPosition[0] / 10, currentPosition[2] / 10, -currentPosition[1] / 10]}
    >
      {/* 机械臂基座 */}
      <mesh position={[0, -5, 0]}>
        <cylinderGeometry args={[3, 3, 10]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>
      
      {/* 机械臂臂段 */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[2, 8, 2]} />
        <meshStandardMaterial color="#3498db" />
      </mesh>
      
      {/* 末端执行器 */}
      <mesh position={[0, 5, 0]}>
        <sphereGeometry args={[1.5]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>
      
      {/* 目标位置指示 */}
      <mesh position={[
        (targetPosition[0] - currentPosition[0]) / 10,
        (targetPosition[2] - currentPosition[2]) / 10,
        -(targetPosition[1] - currentPosition[1]) / 10
      ]}>
        <sphereGeometry args={[1]} />
        <meshBasicMaterial color="#f39c12" opacity={0.5} transparent />
      </mesh>
    </group>
  );
};

// 轨迹路径组件
const AnimatedPath: React.FC<{
  points: TrajectoryPoint[];
  currentTime: number;
  totalTime: number;
}> = ({ points, currentTime, totalTime }) => {
  const pathRef = useRef<THREE.Mesh>(null);

  const { pathGeometry, currentPosition, targetPosition } = useMemo(() => {
    if (points.length < 2) {
      return {
        pathGeometry: null,
        currentPosition: [0, 0, 500] as [number, number, number],
        targetPosition: [0, 0, 500] as [number, number, number],
      };
    }

    // 创建路径几何体
    const pathPoints = points.map(p => 
      new THREE.Vector3(p.position[0] / 10, p.position[2] / 10, -p.position[1] / 10)
    );
    const curve = new THREE.CatmullRomCurve3(pathPoints);
    const pathGeometry = new THREE.TubeGeometry(curve, 64, 0.5, 8, false);

    // 计算当前位置
    const progress = Math.min(currentTime / totalTime, 1);
    const segmentLength = 1 / (points.length - 1);
    const segmentIndex = Math.floor(progress / segmentLength);
    const segmentProgress = (progress % segmentLength) / segmentLength;

    let currentPosition: [number, number, number];
    let targetPosition: [number, number, number];

    if (segmentIndex >= points.length - 1) {
      currentPosition = points[points.length - 1].position;
      targetPosition = points[points.length - 1].position;
    } else {
      const startPoint = points[segmentIndex].position;
      const endPoint = points[segmentIndex + 1].position;
      
      currentPosition = [
        startPoint[0] + (endPoint[0] - startPoint[0]) * segmentProgress,
        startPoint[1] + (endPoint[1] - startPoint[1]) * segmentProgress,
        startPoint[2] + (endPoint[2] - startPoint[2]) * segmentProgress,
      ];
      targetPosition = endPoint;
    }

    return { pathGeometry, currentPosition, targetPosition };
  }, [points, currentTime, totalTime]);

  return (
    <group>
      {/* 轨迹路径 */}
      {pathGeometry && (
        <mesh ref={pathRef} geometry={pathGeometry}>
          <meshStandardMaterial color="#1890ff" opacity={0.6} transparent />
        </mesh>
      )}
      
      {/* 轨迹点 */}
      {points.map((point, index) => (
        <group key={point.id} position={[point.position[0] / 10, point.position[2] / 10, -point.position[1] / 10]}>
          <mesh>
            <sphereGeometry args={[2]} />
            <meshStandardMaterial color="#52c41a" />
          </mesh>
          <Text
            position={[0, 5, 0]}
            fontSize={3}
            color="#666"
            anchorX="center"
            anchorY="middle"
          >
            {point.name}
          </Text>
        </group>
      ))}
      
      {/* 动画机械臂 */}
      <AnimatedRobot 
        currentPosition={currentPosition}
        targetPosition={targetPosition}
      />
    </group>
  );
};

const SimulationViewer: React.FC<SimulationViewerProps> = ({
  trajectoryPoints,
  currentTime,
  totalTime,
  isRunning,
}) => {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [80, 80, 80], fov: 60 }}
        style={{ background: isRunning ? '#f0f8ff' : '#f8f9fa' }}
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
          maxDistance={200}
          minDistance={30}
        />
        
        {/* 坐标系和网格 */}
        <Grid 
          args={[100, 100]} 
          cellSize={5} 
          cellThickness={0.5} 
          cellColor="#bdc3c7"
          sectionSize={10}
          sectionThickness={1}
          sectionColor="#7f8c8d"
        />
        
        {/* 自定义坐标轴 */}
        <group>
          {/* X轴 - 红色 */}
          <mesh position={[10, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <cylinderGeometry args={[0.2, 0.2, 20]} />
            <meshBasicMaterial color="#ff0000" />
          </mesh>
          {/* Y轴 - 绿色 */}
          <mesh position={[0, 10, 0]}>
            <cylinderGeometry args={[0.2, 0.2, 20]} />
            <meshBasicMaterial color="#00ff00" />
          </mesh>
          {/* Z轴 - 蓝色 */}
          <mesh position={[0, 0, 10]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.2, 0.2, 20]} />
            <meshBasicMaterial color="#0000ff" />
          </mesh>
        </group>
        
        {/* 动画路径和机械臂 */}
        <AnimatedPath 
          points={trajectoryPoints}
          currentTime={currentTime}
          totalTime={totalTime}
        />
        
        {/* 状态指示 */}
        <Text
          position={[0, 60, 0]}
          fontSize={6}
          color={isRunning ? '#52c41a' : '#666'}
          anchorX="center"
          anchorY="middle"
        >
          {isRunning ? '仿真运行中...' : '仿真已停止'}
        </Text>
        
        {/* 时间显示 */}
        <Text
          position={[0, 50, 0]}
          fontSize={4}
          color="#666"
          anchorX="center"
          anchorY="middle"
        >
          {`${currentTime.toFixed(1)}s / ${totalTime}s`}
        </Text>
      </Canvas>
    </div>
  );
};

export default SimulationViewer;
