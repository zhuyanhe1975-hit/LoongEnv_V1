import React, { useRef, Suspense } from 'react';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';

interface ER15RobotModelProps {
  robot: MiniRobot;
}

// 简化的ER15-1400模型组件 (使用基本几何体)
const ER15RobotModel: React.FC<ER15RobotModelProps> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  return (
    <group ref={groupRef}>
      {/* 基座 */}
      <mesh position={[0, 0, -10]}>
        <cylinderGeometry args={[35, 35, 20]} />
        <meshStandardMaterial color="#2c3e50" />
      </mesh>
      
      {/* 关节1 - 基座旋转 */}
      <mesh position={[0, 0, 43]}>
        <cylinderGeometry args={[25, 25, 50]} />
        <meshStandardMaterial color="#ff6b6b" />
      </mesh>
      
      {/* 关节2 - 大臂 */}
      <group position={[18, 0, 43]}>
        <mesh>
          <boxGeometry args={[20, 80, 20]} />
          <meshStandardMaterial color="#4ecdc4" />
        </mesh>
        <mesh position={[29, 0, 0]}>
          <boxGeometry args={[58, 15, 15]} />
          <meshStandardMaterial color="#4ecdc4" />
        </mesh>
      </group>
      
      {/* 关节3 - 小臂 */}
      <group position={[76, 0, 43]}>
        <mesh>
          <boxGeometry args={[18, 60, 18]} />
          <meshStandardMaterial color="#45b7d1" />
        </mesh>
        <mesh position={[8, -32, 0]}>
          <boxGeometry args={[16, 64, 16]} />
          <meshStandardMaterial color="#45b7d1" />
        </mesh>
      </group>
      
      {/* 关节4 - 腕部旋转 */}
      <mesh position={[92, -64, 43]}>
        <cylinderGeometry args={[15, 15, 40]} />
        <meshStandardMaterial color="#96ceb4" />
      </mesh>
      
      {/* 关节5 - 腕部俯仰 */}
      <mesh position={[92, -64, 107]}>
        <boxGeometry args={[12, 30, 12]} />
        <meshStandardMaterial color="#feca57" />
      </mesh>
      
      {/* 关节6 - 末端旋转 */}
      <mesh position={[92, -75.6, 107]}>
        <cylinderGeometry args={[10, 10, 20]} />
        <meshStandardMaterial color="#ff9ff3" />
      </mesh>
      
      {/* 连杆 */}
      <mesh position={[47, 0, 43]} rotation={[0, 0, Math.PI/2]}>
        <cylinderGeometry args={[8, 8, 58]} />
        <meshStandardMaterial color="#95a5a6" />
      </mesh>
      
      <mesh position={[84, -32, 43]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[8, 8, 64]} />
        <meshStandardMaterial color="#95a5a6" />
      </mesh>
      
      <mesh position={[92, -64, 75]} rotation={[0, 0, 0]}>
        <cylinderGeometry args={[6, 6, 64]} />
        <meshStandardMaterial color="#95a5a6" />
      </mesh>
      
      {/* ER15-1400 标识 */}
      <mesh position={[0, 0, 15]}>
        <boxGeometry args={[60, 5, 10]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>
      
      {/* 工作空间指示 (1400mm半径) */}
      <mesh position={[0, 0, 0]}>
        <torusGeometry args={[140, 2, 8, 32]} />
        <meshBasicMaterial color="#3498db" opacity={0.3} transparent />
      </mesh>
    </group>
  );
};

export default ER15RobotModel;
