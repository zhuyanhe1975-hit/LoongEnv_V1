import React, { useRef, Suspense } from 'react';
import { useLoader } from '@react-three/fiber';
import { STLLoader } from 'three-stdlib';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';

interface ER15RobotModelProps {
  robot: MiniRobot;
}

// STL模型组件
const STLPart: React.FC<{ 
  url: string; 
  position: [number, number, number]; 
  rotation?: [number, number, number]; 
  color: string;
  scale?: number;
}> = ({ url, position, rotation = [0, 0, 0], color, scale = 0.1 }) => {
  const geometry = useLoader(STLLoader, url);
  
  return (
    <mesh position={position} rotation={rotation} scale={[scale, scale, scale]}>
      <primitive object={geometry} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

// 加载指示器
const LoadingFallback: React.FC = () => (
  <mesh>
    <boxGeometry args={[50, 50, 50]} />
    <meshStandardMaterial color="#cccccc" wireframe />
  </mesh>
);

// ER15-1400 STL模型组件
const ER15RobotModel: React.FC<ER15RobotModelProps> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  return (
    <group ref={groupRef}>
      <Suspense fallback={<LoadingFallback />}>
        {/* 基座 - b_link.STL */}
        <STLPart 
          url="/models/er15/b_link.STL" 
          position={[0, 0, 0]} 
          color="#2c3e50"
          scale={0.1}
        />
        
        {/* 关节1 - l_1.STL */}
        <STLPart 
          url="/models/er15/l_1.STL" 
          position={[0, 0, 43]} 
          rotation={[0, 0, 0]}
          color="#ff6b6b"
          scale={0.1}
        />
        
        {/* 关节2 - l_2.STL */}
        <STLPart 
          url="/models/er15/l_2.STL" 
          position={[18, 0, 43]} 
          rotation={[Math.PI/2, -Math.PI/2, 0]}
          color="#4ecdc4"
          scale={0.1}
        />
        
        {/* 关节3 - l_3.STL */}
        <STLPart 
          url="/models/er15/l_3.STL" 
          position={[76, 0, 43]} 
          rotation={[0, 0, -Math.PI/2]}
          color="#45b7d1"
          scale={0.1}
        />
        
        {/* 关节4 - l_4.STL */}
        <STLPart 
          url="/models/er15/l_4.STL" 
          position={[92, -64, 43]} 
          rotation={[Math.PI, 0, -Math.PI/2]}
          color="#96ceb4"
          scale={0.1}
        />
        
        {/* 关节5 - l_5.STL */}
        <STLPart 
          url="/models/er15/l_5.STL" 
          position={[92, -64, 107]} 
          rotation={[0, 0, -Math.PI/2]}
          color="#feca57"
          scale={0.1}
        />
        
        {/* 关节6 - l_6.STL */}
        <STLPart 
          url="/models/er15/l_6.STL" 
          position={[92, -75.6, 107]} 
          rotation={[Math.PI, 0, Math.PI/2]}
          color="#ff9ff3"
          scale={0.1}
        />
      </Suspense>
      
      {/* 工作空间指示 (1400mm半径) */}
      <mesh position={[0, 0, 0]}>
        <torusGeometry args={[140, 2, 8, 32]} />
        <meshBasicMaterial color="#3498db" opacity={0.3} transparent />
      </mesh>
    </group>
  );
};

export default ER15RobotModel;
