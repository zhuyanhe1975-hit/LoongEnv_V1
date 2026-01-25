import React, { useRef, useState } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Grid, Axes, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { TrajectoryPoint } from '@/types';

interface TrajectoryViewerProps {
  trajectoryPoints: TrajectoryPoint[];
  selectedPoint: string | null;
  onPointSelect: (id: string | null) => void;
  onAddPoint: (position: [number, number, number]) => void;
}

// 轨迹点组件
const TrajectoryPointMesh: React.FC<{
  point: TrajectoryPoint;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ point, isSelected, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
    }
  });

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    onSelect();
  };

  return (
    <group position={[point.position[0] / 10, point.position[2] / 10, -point.position[1] / 10]}>
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[isSelected ? 8 : hovered ? 6 : 4]} />
        <meshStandardMaterial 
          color={isSelected ? '#ff4d4f' : hovered ? '#1890ff' : '#52c41a'} 
        />
      </mesh>
      
      <Text
        position={[0, 10, 0]}
        fontSize={6}
        color={isSelected ? '#ff4d4f' : '#666'}
        anchorX="center"
        anchorY="middle"
      >
        {point.name}
      </Text>
    </group>
  );
};

// 轨迹路径组件
const TrajectoryPath: React.FC<{ points: TrajectoryPoint[] }> = ({ points }) => {
  if (points.length < 2) return null;

  const pathPoints = points.map(p => 
    new THREE.Vector3(p.position[0] / 10, p.position[2] / 10, -p.position[1] / 10)
  );

  const curve = new THREE.CatmullRomCurve3(pathPoints);
  const pathGeometry = new THREE.TubeGeometry(curve, 64, 1, 8, false);

  return (
    <mesh geometry={pathGeometry}>
      <meshStandardMaterial color="#1890ff" opacity={0.6} transparent />
    </mesh>
  );
};

// 工作空间边界
const WorkspaceBounds: React.FC = () => {
  const size = 50; // 500mm / 10
  
  return (
    <group>
      {/* 工作空间边界框 */}
      <mesh>
        <boxGeometry args={[size * 2, size * 2, size * 2]} />
        <meshBasicMaterial 
          color="#1890ff" 
          opacity={0.1} 
          transparent 
          wireframe 
        />
      </mesh>
      
      {/* 工作平面 */}
      <mesh position={[0, -size, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[size * 2, size * 2]} />
        <meshBasicMaterial 
          color="#f0f0f0" 
          opacity={0.3} 
          transparent 
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
};

const TrajectoryViewer: React.FC<TrajectoryViewerProps> = ({
  trajectoryPoints,
  selectedPoint,
  onPointSelect,
  onAddPoint,
}) => {
  const handleCanvasClick = (e: ThreeEvent<MouseEvent>) => {
    // 点击空白区域添加轨迹点
    if (e.intersections.length === 0) {
      const point = e.point;
      const position: [number, number, number] = [
        Math.round(point.x * 10),
        Math.round(-point.z * 10),
        Math.round(point.y * 10),
      ];
      onAddPoint(position);
    }
  };

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [80, 80, 80], fov: 60 }}
        style={{ background: '#f8f9fa' }}
        onClick={handleCanvasClick}
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
        <Axes scale={[20, 20, 20]} />
        
        {/* 工作空间边界 */}
        <WorkspaceBounds />
        
        {/* 轨迹路径 */}
        <TrajectoryPath points={trajectoryPoints} />
        
        {/* 轨迹点 */}
        {trajectoryPoints.map((point) => (
          <TrajectoryPointMesh
            key={point.id}
            point={point}
            isSelected={selectedPoint === point.id}
            onSelect={() => onPointSelect(point.id)}
          />
        ))}
        
        {/* 坐标轴标签 */}
        <Text
          position={[25, 0, 0]}
          fontSize={4}
          color="#e74c3c"
          anchorX="center"
          anchorY="middle"
        >
          X
        </Text>
        <Text
          position={[0, 25, 0]}
          fontSize={4}
          color="#27ae60"
          anchorX="center"
          anchorY="middle"
        >
          Z
        </Text>
        <Text
          position={[0, 0, -25]}
          fontSize={4}
          color="#3498db"
          anchorX="center"
          anchorY="middle"
        >
          Y
        </Text>
      </Canvas>
    </div>
  );
};

export default TrajectoryViewer;
