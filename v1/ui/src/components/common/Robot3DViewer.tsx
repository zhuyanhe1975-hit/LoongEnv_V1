import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Card, CardContent, CardHeader, Box as MuiBox, Typography } from '@mui/material';

import URDFRobotModel from './URDFRobotModel';
import ErrorBoundary from './ErrorBoundary';
import { detectWebGL } from '../../utils/webglDetection';
import { useAppSelector } from '../../store';

interface Robot3DViewerProps {
  height?: number | string;
}

const Robot3DViewer: React.FC<Robot3DViewerProps> = ({ height = 320 }) => {
  const { currentState, isConnected, modelUrdfUrl, modelName } = useAppSelector((state) => state.robot);
  const { backendStatus, backendProgress, backendResults } = useAppSelector((state) => state.tuning);

  const [webglInfo, setWebglInfo] = useState<{ supported: boolean; error?: string } | null>(null);

  const jointPositions = currentState?.jointPositions || [0, 0, 0, 0, 0, 0];

  useEffect(() => {
    setWebglInfo(detectWebGL());
  }, []);

  return (
    <Card
      sx={{
        height,
        minHeight: typeof height === 'number' ? height : 320,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <CardHeader
        title={`${modelName} 3D模型`}
        subheader={isConnected ? '实时姿态显示' : '离线模式'}
        sx={{ py: 1 }}
        titleTypographyProps={{ sx: { fontSize: 14, fontWeight: 800 } }}
        subheaderTypographyProps={{ sx: { fontSize: 12, fontWeight: 600 } }}
      />

      <CardContent sx={{ flexGrow: 1, minHeight: 0, p: 1 }}>
        <MuiBox sx={{ height: '100%', position: 'relative' }}>
          <ErrorBoundary>
            <Canvas
              camera={{
                position: [2.2, 2.2, 1.6],
                fov: 50,
                near: 0.001,
                far: 1000,
                up: [0, 0, 1],
              }}
              dpr={[1, 2]}
              gl={{ antialias: true, powerPreference: 'high-performance' }}
              style={{
                background: '#101418',
                width: '100%',
                height: '100%',
                display: 'block',
              }}
              onCreated={(state) => {
                state.camera.up.set(0, 0, 1);
                state.camera.position.set(2.2, 2.2, 1.6);
                state.camera.lookAt(0, 0, 0.6);
              }}
            >
              <ambientLight intensity={0.4} />
              <directionalLight position={[10, 10, 5]} intensity={0.8} />
              <directionalLight position={[-5, 5, 5]} intensity={0.3} />
              <pointLight position={[0, 0, 3]} intensity={0.2} />

              <group rotation={[Math.PI / 2, 0, 0]}>
                <gridHelper args={[4, 20, '#2a3540', '#1a222a']} />
              </group>

              <URDFRobotModel jointPositions={jointPositions} urdfUrl={modelUrdfUrl} />

              <OrbitControls
                enablePan
                enableZoom
                enableRotate
                enableDamping
                dampingFactor={0.08}
                maxPolarAngle={Math.PI * 0.95}
                minDistance={0.1}
                maxDistance={50}
                target={[0, 0, 0.6]}
              />
            </Canvas>
          </ErrorBoundary>

          {!isConnected && (
            <MuiBox
              sx={{
                position: 'absolute',
                bottom: 8,
                left: 8,
                backgroundColor: 'rgba(244, 67, 54, 0.85)',
                color: 'white',
                px: 1,
                py: 0.5,
                borderRadius: 1,
              }}
            >
              <Typography variant="caption" sx={{ fontWeight: 700 }}>
                未连接（显示默认姿态）
              </Typography>
            </MuiBox>
          )}

          <MuiBox
            sx={{
              position: 'absolute',
              bottom: 8,
              right: 8,
              backgroundColor: 'rgba(0, 0, 0, 0.65)',
              color: 'white',
              px: 1,
              py: 0.75,
              borderRadius: 1,
              maxWidth: 220,
            }}
          >
            <Typography variant="caption" display="block" sx={{ fontWeight: 700 }}>
              参数优化: {backendStatus === 'running' ? `运行中 ${backendProgress}%` : backendStatus === 'completed' ? '已完成' : backendStatus === 'error' ? '错误' : '空闲'}
            </Typography>
            {typeof backendResults?.overallImprovement === 'number' && backendStatus !== 'running' && (
              <Typography variant="caption" display="block">
                提升: {Number(backendResults.overallImprovement).toFixed(2)}%
              </Typography>
            )}
          </MuiBox>

          {webglInfo && !webglInfo.supported && (
            <MuiBox
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                p: 2,
                backgroundColor: 'rgba(0,0,0,0.75)',
              }}
            >
              <Typography color="white">
                WebGL 不支持：{webglInfo.error || '无法渲染 3D'}
              </Typography>
            </MuiBox>
          )}
        </MuiBox>
      </CardContent>
    </Card>
  );
};

export default Robot3DViewer;
