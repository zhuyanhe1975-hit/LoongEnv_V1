import React, { useRef, useEffect, useState } from 'react';
import { useLoader } from '@react-three/fiber';
import { STLLoader } from 'three-stdlib';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';

interface ER15RobotModelProps {
  robot: MiniRobot;
}

// STL模型组件
const STLModel: React.FC<{ url: string; position: [number, number, number]; rotation?: [number, number, number]; color: string }> = ({ 
  url, 
  position, 
  rotation = [0, 0, 0], 
  color 
}) => {
  try {
    const geometry = useLoader(STLLoader, url);
    return (
      <mesh position={position} rotation={rotation} geometry={geometry}>
        <meshStandardMaterial color={color} />
      </mesh>
    );
  } catch (error) {
    console.warn(`Failed to load STL: ${url}`, error);
    return null;
  }
};

// ER15-1400 专用3D模型组件
const ER15RobotModel: React.FC<ER15RobotModelProps> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  return (
    <group ref={groupRef}>
      {/* 基座 - b_link.STL */}
      <STLModel 
        url="/models/er15/b_link.STL" 
        position={[0, 0, 0]} 
        color="#2c3e50" 
      />
      
      {/* 关节1 - l_1.STL */}
      <STLModel 
        url="/models/er15/l_1.STL" 
        position={[0, 0, -43]} 
        color="#ff6b6b" 
      />
      
      {/* 关节2 - l_2.STL */}
      <STLModel 
        url="/models/er15/l_2.STL" 
        position={[18, 0, 0]} 
        rotation={[0, 0, -Math.PI/2]} 
        color="#4ecdc4" 
      />
      
      {/* 关节3 - l_3.STL */}
      <STLModel 
        url="/models/er15/l_3.STL" 
        position={[76, 0, 0]} 
        rotation={[0, 0, -Math.PI/2]} 
        color="#45b7d1" 
      />
      
      {/* 关节4 - l_4.STL */}
      <STLModel 
        url="/models/er15/l_4.STL" 
        position={[92, 0, -64]} 
        rotation={[Math.PI, 0, -Math.PI/2]} 
        color="#96ceb4" 
      />
      
      {/* 关节5 - l_5.STL */}
      <STLModel 
        url="/models/er15/l_5.STL" 
        position={[92, 0, 107]} 
        rotation={[0, 0, -Math.PI/2]} 
        color="#feca57" 
      />
      
      {/* 关节6 - l_6.STL */}
      <STLModel 
        url="/models/er15/l_6.STL" 
        position={[92, -11.6, 107]} 
        rotation={[Math.PI, 0, Math.PI/2]} 
        color="#ff9ff3" 
      />
      
      {/* 工作空间指示 (1400mm半径) */}
      <mesh position={[0, 0, 0]}>
        <torusGeometry args={[140, 2, 8, 32]} />
        <meshBasicMaterial color="#3498db" opacity={0.3} transparent />
      </mesh>
    </group>
  );
};

export default ER15RobotModel;
