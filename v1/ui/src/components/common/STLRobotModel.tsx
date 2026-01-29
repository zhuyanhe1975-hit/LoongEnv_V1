import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import STLModel from './STLLoader';

interface STLRobotModelProps {
  jointPositions?: number[];
}

const STLRobotModel: React.FC<STLRobotModelProps> = ({ 
  jointPositions = [0, 0, 0, 0, 0, 0] 
}) => {
  const qFromRPY = (rpy: [number, number, number]) => {
    const [r, p, y] = rpy;
    // URDF rpy 约定：Rz(yaw) * Ry(pitch) * Rx(roll)
    const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), r);
    const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), p);
    const qz = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), y);
    return qz.multiply(qy).multiply(qx);
  };

  // 关节“运动轴”引用（与 URDF 保持一致：joint origin 的固定 rpy 不应被关节角覆盖）
  const joint1AxisRef = useRef<THREE.Group>(null);
  const joint2AxisRef = useRef<THREE.Group>(null);
  const joint3AxisRef = useRef<THREE.Group>(null);
  const joint4AxisRef = useRef<THREE.Group>(null);
  const joint5AxisRef = useRef<THREE.Group>(null);
  const joint6AxisRef = useRef<THREE.Group>(null);

  // 使用正确的缩放比例：1:1 (用户确认)
  const scale: [number, number, number] = [1.0, 1.0, 1.0];

  // 更新关节角度 - 根据URDF中的axis定义
  useEffect(() => {
    // Joint 1: axis="0 0 1" (Z轴旋转)
    if (joint1AxisRef.current) joint1AxisRef.current.rotation.z = jointPositions[0];
    
    // Joint 2: axis="0 0 1" (Z轴旋转，但在局部坐标系中)
    if (joint2AxisRef.current) joint2AxisRef.current.rotation.z = jointPositions[1];
    
    // Joint 3: axis="0 0 1" (Z轴旋转)
    if (joint3AxisRef.current) joint3AxisRef.current.rotation.z = jointPositions[2];
    
    // Joint 4: axis="0 0 1" (Z轴旋转)
    if (joint4AxisRef.current) joint4AxisRef.current.rotation.z = jointPositions[3];
    
    // Joint 5: axis="0 0 1" (Z轴旋转)
    if (joint5AxisRef.current) joint5AxisRef.current.rotation.z = jointPositions[4];
    
    // Joint 6: axis="0 0 1" (Z轴旋转)
    if (joint6AxisRef.current) joint6AxisRef.current.rotation.z = jointPositions[5];
  }, [jointPositions]);

  return (
    <group>
      {/* 基座 - base_link */}
      {/* URDF: <visual><origin xyz="0 0 0" rpy="0 0 0" /> */}
      <STLModel 
        url="/models/b_link.STL" 
        position={[0, 0, 0]} 
        rotation={[0, 0, 0]}
        color="#666666"
        scale={scale}
      />
      
      {/* joint_1: origin xyz="0 0 0.43" rpy="0 0 0" then rotate about axis */}
      <group position={[0, 0, 0.43]} quaternion={qFromRPY([0, 0, 0])}>
        <group ref={joint1AxisRef}>
          {/* link_1 visual: origin xyz="0 0 -0.43" rpy="0 0 0" */}
          <STLModel 
            url="/models/l_1.STL" 
            position={[0, 0, -0.43]} 
            rotation={[0, 0, 0]}
            color="#0000CC"
            scale={scale}
          />

          {/* joint_2: origin xyz="0.18 0 0" rpy="1.5707963267 -1.5707963267 0" */}
          <group position={[0.18, 0, 0]} quaternion={qFromRPY([Math.PI / 2, -Math.PI / 2, 0])}>
            <group ref={joint2AxisRef}>
              {/* link_2 visual */}
              <group quaternion={qFromRPY([0, 0, -Math.PI / 2])}>
                <STLModel 
                  url="/models/l_2.STL" 
                  position={[0, 0, 0]} 
                  rotation={[0, 0, 0]}
                  color="#FF0000"
                  scale={scale}
                />
              </group>

              {/* joint_3: origin xyz="0.58 0 0" rpy="0 0 0" */}
              <group position={[0.58, 0, 0]} quaternion={qFromRPY([0, 0, 0])}>
                <group ref={joint3AxisRef}>
                  {/* link_3 visual */}
                  <group quaternion={qFromRPY([0, 0, -Math.PI / 2])}>
                    <STLModel 
                      url="/models/l_3.STL" 
                      position={[0, 0, 0]} 
                      rotation={[0, 0, 0]}
                      color="#0000CC"
                      scale={scale}
                    />
                  </group>

                  {/* joint_4: origin xyz="0.16 -0.64 0" rpy="-1.5707963267 0 3.141592653" */}
                  <group position={[0.16, -0.64, 0]} quaternion={qFromRPY([-Math.PI / 2, 0, Math.PI])}>
                    <group ref={joint4AxisRef}>
                      {/* link_4 visual */}
                      <group quaternion={qFromRPY([Math.PI, 0, -Math.PI / 2])}>
                        <STLModel 
                          url="/models/l_4.STL" 
                          position={[0, 0, -0.64]} 
                          rotation={[0, 0, 0]}
                          color="#00CCCC"
                          scale={scale}
                        />
                      </group>

                      {/* joint_5: origin xyz="0 0 0" rpy="-1.5707963267 0 3.141592653" */}
                      <group position={[0, 0, 0]} quaternion={qFromRPY([-Math.PI / 2, 0, Math.PI])}>
                        <group ref={joint5AxisRef}>
                          {/* link_5 visual */}
                          <group quaternion={qFromRPY([0, 0, -Math.PI / 2])}>
                            <STLModel 
                              url="/models/l_5.STL" 
                              position={[0, 0, 0]} 
                              rotation={[0, 0, 0]}
                              color="#FF0000"
                              scale={scale}
                            />
                          </group>

                          {/* joint_6: origin xyz="0 -0.116 0" rpy="1.5707963267 0 0" */}
                          <group position={[0, -0.116, 0]} quaternion={qFromRPY([Math.PI / 2, 0, 0])}>
                            <group ref={joint6AxisRef}>
                              {/* link_6 visual */}
                              <group quaternion={qFromRPY([Math.PI, 0, Math.PI / 2])}>
                                <STLModel 
                                  url="/models/l_6.STL" 
                                  position={[0, 0, 0]} 
                                  rotation={[0, 0, 0]}
                                  color="#E6E6E6"
                                  scale={scale}
                                />
                              </group>

                              {/* 末端坐标系 */}
                              <group position={[0, 0, -0.03]}>
                                <mesh position={[0.05, 0, 0]}>
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
