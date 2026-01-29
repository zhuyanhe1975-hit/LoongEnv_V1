import React, { useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder } from '@react-three/drei';
import { Card, CardHeader, CardContent, Box as MuiBox, Typography, Button } from '@mui/material';
import { useAppSelector } from '../../store';
import * as THREE from 'three';
import { getRobotSpecs } from '../../utils/urdfParser';
import Test3D from './Test3D';
import ErrorBoundary from './ErrorBoundary';
import { detectWebGL, logWebGLInfo } from '../../utils/webglDetection';
import STLRobotModel from './STLRobotModel';

// 简单的测试立方体
const TestCube: React.FC = () => {
  return (
    <Box args={[1, 1, 1]} position={[0, 0, 0.5]}>
      <meshStandardMaterial color="#ff6b6b" />
    </Box>
  );
};

// 简化的机器人模型（使用基本几何体）
const SimpleRobotModel: React.FC<{ jointPositions?: number[] }> = ({ 
  jointPositions = [0, 0, 0, 0, 0, 0] 
}) => {
  const joint1Ref = useRef<THREE.Group>(null);
  const joint2Ref = useRef<THREE.Group>(null);
  const joint3Ref = useRef<THREE.Group>(null);
  const joint4Ref = useRef<THREE.Group>(null);
  const joint5Ref = useRef<THREE.Group>(null);
  const joint6Ref = useRef<THREE.Group>(null);

  useEffect(() => {
    // 更新关节角度
    if (joint1Ref.current) joint1Ref.current.rotation.z = jointPositions[0];
    if (joint2Ref.current) joint2Ref.current.rotation.z = jointPositions[1];
    if (joint3Ref.current) joint3Ref.current.rotation.z = jointPositions[2];
    if (joint4Ref.current) joint4Ref.current.rotation.z = jointPositions[3];
    if (joint5Ref.current) joint5Ref.current.rotation.z = jointPositions[4];
    if (joint6Ref.current) joint6Ref.current.rotation.z = jointPositions[5];
  }, [jointPositions]);

  return (
    <group>
      {/* 基座 */}
      <Cylinder args={[0.2, 0.2, 0.1]} position={[0, 0, 0.05]}>
        <meshStandardMaterial color="#666666" />
      </Cylinder>
      
      {/* 关节1 - 底部旋转 */}
      <group ref={joint1Ref} position={[0, 0, 0.43]}>
        <Cylinder args={[0.1, 0.1, 0.3]} position={[0, 0, 0.15]}>
          <meshStandardMaterial color="#2196F3" />
        </Cylinder>
        
        {/* 关节2 - 大臂 */}
        <group ref={joint2Ref} position={[0.18, 0, 0.3]} rotation={[Math.PI/2, 0, 0]}>
          <Box args={[0.58, 0.1, 0.1]} position={[0.29, 0, 0]}>
            <meshStandardMaterial color="#FF5722" />
          </Box>
          
          {/* 关节3 - 小臂 */}
          <group ref={joint3Ref} position={[0.58, 0, 0]}>
            <Box args={[0.16, 0.1, 0.1]} position={[0.08, 0, 0]}>
              <meshStandardMaterial color="#4CAF50" />
            </Box>
            
            {/* 关节4 - 手腕旋转 */}
            <group ref={joint4Ref} position={[0.16, 0, 0]} rotation={[0, Math.PI/2, 0]}>
              <Cylinder args={[0.05, 0.05, 0.64]} position={[0, 0, -0.32]}>
                <meshStandardMaterial color="#9C27B0" />
              </Cylinder>
              
              {/* 关节5 - 手腕俯仰 */}
              <group ref={joint5Ref} position={[0, 0, -0.64]} rotation={[Math.PI/2, 0, 0]}>
                <Cylinder args={[0.04, 0.04, 0.1]} position={[0, 0, -0.05]}>
                  <meshStandardMaterial color="#FF9800" />
                </Cylinder>
                
                {/* 关节6 - 末端执行器 */}
                <group ref={joint6Ref} position={[0, 0, -0.116]}>
                  <Box args={[0.08, 0.08, 0.03]} position={[0, 0, -0.015]}>
                    <meshStandardMaterial color="#607D8B" />
                  </Box>
                  
                  {/* 末端坐标系 */}
                  <group position={[0, 0, -0.03]}>
                    <Box args={[0.1, 0.005, 0.005]} position={[0.05, 0, 0]}>
                      <meshStandardMaterial color="#ff0000" />
                    </Box>
                    <Box args={[0.005, 0.1, 0.005]} position={[0, 0.05, 0]}>
                      <meshStandardMaterial color="#00ff00" />
                    </Box>
                    <Box args={[0.005, 0.005, 0.1]} position={[0, 0, 0.05]}>
                      <meshStandardMaterial color="#0000ff" />
                    </Box>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
      
      {/* 基座坐标系 */}
      <group position={[0, 0, 0.02]}>
        <Box args={[0.15, 0.005, 0.005]} position={[0.075, 0, 0]}>
          <meshStandardMaterial color="#ff4444" />
        </Box>
        <Box args={[0.005, 0.15, 0.005]} position={[0, 0.075, 0]}>
          <meshStandardMaterial color="#44ff44" />
        </Box>
        <Box args={[0.005, 0.005, 0.15]} position={[0, 0, 0.075]}>
          <meshStandardMaterial color="#4444ff" />
        </Box>
      </group>
    </group>
  );
};

// 3D场景组件
const Robot3DScene: React.FC<{ 
  jointPositions?: number[]; 
  showTestCube?: boolean;
  useSTLModel?: boolean;
}> = ({ 
  jointPositions = [0, 0, 0, 0, 0, 0], 
  showTestCube = false,
  useSTLModel = true
}) => {
  console.log('Robot3DScene rendering...', { jointPositions, showTestCube, useSTLModel });
  
  return (
    <Canvas
      camera={{ position: [3, 3, 2], fov: 50 }}
      style={{ 
        background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
        width: '100%',
        height: '100%',
        display: 'block'
      }}
      onCreated={(state) => {
        console.log('Robot3DScene Canvas created:', state);
      }}
      onError={(error) => {
        console.error('Robot3DScene Canvas error:', error);
      }}
    >
      {/* 照明 */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <directionalLight position={[-5, 5, 5]} intensity={0.3} />
      <pointLight position={[0, 0, 3]} intensity={0.2} />
      
      {/* 显示内容 */}
      {showTestCube ? (
        <TestCube />
      ) : useSTLModel ? (
        <STLRobotModel jointPositions={jointPositions} />
      ) : (
        <SimpleRobotModel jointPositions={jointPositions} />
      )}
      
      {/* 地面网格 */}
      <gridHelper args={[4, 40, '#ffffff', '#cccccc']} position={[0, 0, 0]} />
      
      {/* 工作空间指示 */}
      <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, 0, 0.01]}>
        <ringGeometry args={[1.4, 1.5, 64]} />
        <meshBasicMaterial color="#ffff00" transparent opacity={0.2} />
      </mesh>
      
      {/* 轨道控制 */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI / 2}
        minDistance={1}
        maxDistance={10}
        target={[0, 0, 0.5]}
      />
    </Canvas>
  );
};

// 主要的3D查看器组件
const Robot3DViewer: React.FC = () => {
  const { currentState, isConnected } = useAppSelector((state) => state.robot);
  const robotSpecs = getRobotSpecs();
  const [showTestCube, setShowTestCube] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [useSimpleTest, setUseSimpleTest] = useState(false);
  const [useSTLModel, setUseSTLModel] = useState(true); // 默认使用STL模型
  const [webglInfo, setWebglInfo] = useState<any>(null);
  
  // 使用真实的关节位置数据
  const jointPositions = currentState?.jointPositions || [0, 0, 0, 0, 0, 0];
  
  // 调试信息 - 减少依赖项避免频繁重新渲染
  useEffect(() => {
    // 检测WebGL支持
    const info = detectWebGL();
    setWebglInfo(info);
  }, []); // 只在组件挂载时执行一次

  useEffect(() => {
    if (debugMode) {
      logWebGLInfo();
      console.log('Robot3DViewer Debug Info:');
      console.log('- isConnected:', isConnected);
      console.log('- jointPositions:', jointPositions);
      console.log('- robotSpecs:', robotSpecs);
      console.log('- showTestCube:', showTestCube);
      console.log('- useSimpleTest:', useSimpleTest);
      console.log('- webglInfo:', webglInfo);
    }
  }, [debugMode]); // 只在调试模式切换时执行
  
  return (
    <Card sx={{ height: 400 }}>
      <CardHeader 
        title={`${robotSpecs.name} 3D模型`}
        subheader={
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <span>
              {isConnected ? `实时姿态显示 - 工作半径${robotSpecs.reach}mm` : "离线模式"}
            </span>
            <Button 
              size="small" 
              variant="outlined"
              onClick={() => setShowTestCube(!showTestCube)}
            >
              {showTestCube ? '显示机器人' : '显示测试立方体'}
            </Button>
            <Button 
              size="small" 
              variant="outlined"
              onClick={() => setUseSTLModel(!useSTLModel)}
            >
              {useSTLModel ? '简化模型' : 'STL模型'}
            </Button>
            <Button 
              size="small" 
              variant="outlined"
              onClick={() => setUseSimpleTest(!useSimpleTest)}
            >
              {useSimpleTest ? '使用复杂渲染' : '使用简单测试'}
            </Button>
            <Button 
              size="small" 
              variant="outlined"
              onClick={() => setDebugMode(!debugMode)}
            >
              {debugMode ? '关闭调试' : '开启调试'}
            </Button>
          </div>
        }
      />
      <CardContent sx={{ height: 320, p: 1 }}>
        <MuiBox sx={{ height: '100%', position: 'relative' }}>
          <ErrorBoundary>
            {useSimpleTest ? (
              <Test3D />
            ) : (
              <Robot3DScene 
                jointPositions={jointPositions} 
                showTestCube={showTestCube} 
                useSTLModel={useSTLModel}
              />
            )}
          </ErrorBoundary>
          
          {/* 关节角度信息 */}
          {!showTestCube && (
            <MuiBox
              sx={{
                position: 'absolute',
                top: 8,
                left: 8,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                color: 'white',
                p: 1,
                borderRadius: 1,
                fontSize: '0.7rem',
                maxWidth: '150px',
              }}
            >
              <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                关节角度 (弧度):
              </Typography>
              {jointPositions.map((angle: number, index: number) => {
                const degrees = (angle * 180 / Math.PI).toFixed(1);
                return (
                  <Typography key={index} variant="caption" display="block" sx={{ fontSize: '0.65rem' }}>
                    J{index+1}: {angle.toFixed(3)} ({degrees}°)
                  </Typography>
                );
              })}
            </MuiBox>
          )}
          
          {/* 机器人规格信息 */}
          <MuiBox
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              color: 'white',
              p: 1,
              borderRadius: 1,
              fontSize: '0.7rem',
            }}
          >
            <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', mb: 0.5 }}>
              显示状态:
            </Typography>
            <Typography variant="caption" display="block" sx={{ fontSize: '0.65rem' }}>
              模式: {useSimpleTest ? '简单测试' : showTestCube ? '测试立方体' : useSTLModel ? 'STL模型' : '简化模型'}
            </Typography>
            <Typography variant="caption" display="block" sx={{ fontSize: '0.65rem' }}>
              连接: {isConnected ? '已连接' : '离线'}
            </Typography>
            <Typography variant="caption" display="block" sx={{ fontSize: '0.65rem' }}>
              调试: {debugMode ? '开启' : '关闭'}
            </Typography>
          </MuiBox>
          
          {/* WebGL状态指示 */}
          {webglInfo && !webglInfo.supported && (
            <MuiBox
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                backgroundColor: 'rgba(244, 67, 54, 0.9)',
                color: 'white',
                p: 2,
                borderRadius: 1,
                textAlign: 'center',
                zIndex: 1000,
              }}
            >
              <Typography variant="h6" gutterBottom>
                WebGL不支持
              </Typography>
              <Typography variant="body2">
                {webglInfo.error || '您的浏览器不支持WebGL，无法显示3D模型'}
              </Typography>
            </MuiBox>
          )}
          
          {/* 状态指示 */}
          {!isConnected && (
            <MuiBox
              sx={{
                position: 'absolute',
                bottom: 8,
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(244, 67, 54, 0.9)',
                color: 'white',
                p: 1,
                borderRadius: 1,
                textAlign: 'center',
              }}
            >
              <Typography variant="caption">
                机器人未连接 - 显示默认姿态
              </Typography>
            </MuiBox>
          )}
          
          {/* 调试信息 */}
          {debugMode && (
            <MuiBox
              sx={{
                position: 'absolute',
                bottom: 8,
                right: 8,
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                color: 'white',
                p: 1,
                borderRadius: 1,
                fontSize: '0.6rem',
                maxWidth: '200px',
              }}
            >
              <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                调试信息:
              </Typography>
              <Typography variant="caption" display="block">
                Three.js: {THREE.REVISION}
              </Typography>
              <Typography variant="caption" display="block">
                Canvas: {useSimpleTest ? 'Simple' : showTestCube ? 'Test' : 'Robot'}
              </Typography>
              <Typography variant="caption" display="block">
                WebGL: {webglInfo?.supported ? webglInfo.version : 'Not Supported'}
              </Typography>
              <Typography variant="caption" display="block">
                Joints: [{jointPositions.map((j: number) => j.toFixed(2)).join(', ')}]
              </Typography>
            </MuiBox>
          )}
        </MuiBox>
      </CardContent>
    </Card>
  );
};

export default Robot3DViewer;