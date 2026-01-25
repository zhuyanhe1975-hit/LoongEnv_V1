import React, { useRef } from 'react';
import * as THREE from 'three';
import type { MiniRobot } from '@/types';

interface ER15RobotModelProps {
  robot: MiniRobot;
}

// ER15-1400 专用3D模型组件
const ER15RobotModel: React.FC<ER15RobotModelProps> = ({ robot }) => {
  const groupRef = useRef<THREE.Group>(null);

  // ER15-1400的特定几何形状和颜色
  const renderER15Joint = (index: number, position: [number, number, number]) => {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'];
    const sizes = [
      [25, 50, 25],   // Joint 1 - 基座
      [20, 80, 20],   // Joint 2 - 大臂
      [18, 60, 18],   // Joint 3 - 小臂
      [15, 40, 15],   // Joint 4 - 腕部1
      [12, 30, 12],   // Joint 5 - 腕部2
      [10, 20, 10],   // Joint 6 - 末端
    ];

    // 防止数组越界
    if (index >= sizes.length) return null;

    return (
      <group key={index} position={position}>
        {/* 关节主体 */}
        <mesh>
          <boxGeometry args={sizes[index]} />
          <meshStandardMaterial color={colors[index]} />
        </mesh>
        
        {/* 连杆 */}
        {index < 5 && (
          <mesh position={[0, sizes[index][1] / 2 + 20, 0]}>
            <cylinderGeometry args={[8, 8, 40]} />
            <meshStandardMaterial color="#95a5a6" />
          </mesh>
        )}
        
        {/* 关节标识 */}
        <mesh position={[0, 0, sizes[index][2] / 2 + 5]}>
          <sphereGeometry args={[3]} />
          <meshBasicMaterial color="#e74c3c" />
        </mesh>
      </group>
    );
  };

  // 根据ER15-1400的DH参数计算关节位置
  const calculateER15Positions = () => {
    const positions: [number, number, number][] = [];
    
    // ER15-1400的6个关节位置 (简化版)
    positions.push([0, 0, 0]);           // Joint 1 - 基座
    positions.push([0, 0, 43]);          // Joint 2 (d=430mm -> 43 in scale)
    positions.push([18, 0, 43]);         // Joint 3 (a=180mm -> 18 in scale)
    positions.push([76, 0, 43]);         // Joint 4 (a=580mm -> 58 in scale)
    positions.push([92, 0, 107]);        // Joint 5 (a=160mm, d=640mm)
    positions.push([92, -11.6, 107]);    // Joint 6 (d=116mm -> 11.6 in scale)
    
    return positions;
  };

  const jointPositions = calculateER15Positions();

  return (
    <group ref={groupRef}>
      {/* ER15-1400 基座 */}
      <mesh position={[0, 0, -10]}>
        <cylinderGeometry args={[35, 35, 20]} />
        <meshStandardMaterial color="#2c3e50" />
      </mesh>
      
      {/* ER15-1400 标识 */}
      <mesh position={[0, 0, 15]}>
        <boxGeometry args={[60, 5, 10]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>
      
      {/* 各关节 */}
      {jointPositions.map((position, index) => 
        renderER15Joint(index, position)
      )}
      
      {/* 工作空间指示 (1400mm半径) */}
      <mesh position={[0, 0, 0]}>
        <torusGeometry args={[140, 2, 8, 32]} />
        <meshBasicMaterial color="#3498db" opacity={0.3} transparent />
      </mesh>
    </group>
  );
};

export default ER15RobotModel;
