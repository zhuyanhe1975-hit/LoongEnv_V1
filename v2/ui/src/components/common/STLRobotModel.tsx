import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import STLModel from './STLLoader';

interface STLRobotModelProps {
  jointPositions?: number[];
}

const STLRobotModel: React.FC<STLRobotModelProps> = ({ 
  jointPositions = [0, 0, 0, 0, 0, 0] 
}) => {
  // 关节引用
  const joint1Ref = useRef<THREE.Group>(null);
  const joint2Ref = useRef<THREE.Group>(null);
  const joint3Ref = useRef<THREE.Group>(null);
  const joint4Ref = useRef<THREE.Group>(null);
  const joint5Ref = useRef<THREE.Group>(null);
  const joint6Ref = useRef<THREE.Group>(null);

  // 更新关节角度
  useEffect(() => {
    if (joint1Ref.current) joint1Ref.current.rotation.z = jointPositions[0];
    if (joint2Ref.current) joint2Ref.current.rotation.y = jointPositions[1];
    if (joint3Ref.current) joint3Ref.current.rotation.y = jointPositions[2];
    if (joint4Ref.current) joint4Ref.current.rotation.x = jointPositions[3];
    if (joint5Ref.current) joint5Ref.current.rotation.y = jointPositions[4];
    if (joint6Ref.current) joint6Ref.current.rotation.x = jointPositions[5];
  }, [jointPositions]);

  return (
    <group>
      {/* 基座 - b_link.STL */}
      <STLModel 
        url="/models/b_link.STL" 
        position={[0, 0, 0]} 
        color="#666666"
        scale={[0.001, 0.001, 0.001]} // STL文件通常以mm为单位，需要缩放
      />
      
      {/* 关节1 - 底部旋转 */}
      <group ref={joint1Ref} position={[0, 0, 0.43]}>
        <STLModel 
          url="/models/l_1.STL" 
          position={[0, 0, 0]} 
          color="#2196F3"
          scale={[0.001, 0.001, 0.001]}
        />
        
        {/* 关节2 - 大臂 */}
        <group ref={joint2Ref} position={[0.18, 0, 0.3]}>
          <STLModel 
            url="/models/l_2.STL" 
            position={[0, 0, 0]} 
            color="#FF5722"
            scale={[0.001, 0.001, 0.001]}
          />
          
          {/* 关节3 - 小臂 */}
          <group ref={joint3Ref} position={[0.58, 0, 0]}>
            <STLModel 
              url="/models/l_3.STL" 
              position={[0, 0, 0]} 
              color="#4CAF50"
              scale={[0.001, 0.001, 0.001]}
            />
            
            {/* 关节4 - 手腕旋转 */}
            <group ref={joint4Ref} position={[0.16, 0, 0]}>
              <STLModel 
                url="/models/l_4.STL" 
                position={[0, 0, 0]} 
                color="#9C27B0"
                scale={[0.001, 0.001, 0.001]}
              />
              
              {/* 关节5 - 手腕俯仰 */}
              <group ref={joint5Ref} position={[0, 0, -0.64]}>
                <STLModel 
                  url="/models/l_5.STL" 
                  position={[0, 0, 0]} 
                  color="#FF9800"
                  scale={[0.001, 0.001, 0.001]}
                />
                
                {/* 关节6 - 末端执行器 */}
                <group ref={joint6Ref} position={[0, 0, -0.116]}>
                  <STLModel 
                    url="/models/l_6.STL" 
                    position={[0, 0, 0]} 
                    color="#607D8B"
                    scale={[0.001, 0.001, 0.001]}
                  />
                  
                  {/* 末端坐标系 */}
                  <group position={[0, 0, -0.03]}>
                    <mesh>
                      <boxGeometry args={[0.1, 0.005, 0.005]} />
                      <meshBasicMaterial color="#ff0000" />
                    </mesh>
                    <mesh position={[0, 0.05, 0]}>
                      <boxGeometry args={[0.005, 0.1, 0.005]} />
                      <meshBasicMaterial color="#00ff00" />
                    </mesh>
                    <mesh position={[0, 0, 0.05]}>
                      <boxGeometry args={[0.005, 0.005, 0.1]} />
                      <meshBasicMaterial color="#0000ff" />
                    </mesh>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
      
      {/* 基座坐标系 */}
      <group position={[0, 0, 0.02]}>
        <mesh position={[0.075, 0, 0]}>
          <boxGeometry args={[0.15, 0.005, 0.005]} />
          <meshBasicMaterial color="#ff4444" />
        </mesh>
        <mesh position={[0, 0.075, 0]}>
          <boxGeometry args={[0.005, 0.15, 0.005]} />
          <meshBasicMaterial color="#44ff44" />
        </mesh>
        <mesh position={[0, 0, 0.075]}>
          <boxGeometry args={[0.005, 0.005, 0.15]} />
          <meshBasicMaterial color="#4444ff" />
        </mesh>
      </group>
    </group>
  );
};

export default STLRobotModel;