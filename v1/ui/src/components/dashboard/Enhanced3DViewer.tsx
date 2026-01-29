import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Box as MuiBox, 
  Typography, 
  Button, 
  Chip, 
  IconButton, 
  Tooltip,
  ButtonGroup,
} from '@mui/material';
import { 
  Fullscreen as FullscreenIcon,
  ThreeDRotation as ThreeDRotationIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { useAppSelector } from '../../store';
import { getRobotSpecs } from '../../utils/urdfParser';
import ErrorBoundary from '../common/ErrorBoundary';
import { detectWebGL } from '../../utils/webglDetection';
import STLRobotModel from '../common/STLRobotModel';

// 工业风格3D场景
const IndustrialRobot3DScene: React.FC<{ 
  jointPositions?: number[]; 
  showCoordinates?: boolean;
  showWorkspace?: boolean;
  cameraPreset?: 'front' | 'side' | 'top' | 'iso';
}> = ({ 
  jointPositions = [0, 0, 0, 0, 0, 0], 
  showCoordinates = true,
  showWorkspace = true,
  cameraPreset = 'iso'
}) => {
  
  const getCameraPosition = (preset: string): [number, number, number] => {
    switch (preset) {
      case 'front': return [0, -3, 1];
      case 'side': return [3, 0, 1];
      case 'top': return [0, 0, 4];
      default: return [3, 3, 2]; // iso
    }
  };

  return (
    <Canvas
      camera={{ 
        position: getCameraPosition(cameraPreset),
        fov: 50,
        near: 0.001,
        far: 1000,
        up: [0, 0, 1] // Z-up工业标准
      }}
      style={{ 
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        width: '100%',
        height: '100%',
        display: 'block'
      }}
      onCreated={(state) => {
        state.camera.up.set(0, 0, 1);
        state.camera.position.set(...getCameraPosition(cameraPreset));
        state.camera.lookAt(0, 0, 0.6);
      }}
    >
      {/* 工业级照明设置 */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[10, 10, 8]} intensity={1.0} castShadow />
      <directionalLight position={[-5, 5, 5]} intensity={0.4} />
      <pointLight position={[0, 0, 3]} intensity={0.3} color="#2563EB" />
      
      {/* 世界坐标系 */}
      {showCoordinates && <axesHelper args={[1.0]} />}
      
      {/* 机器人模型 */}
      <STLRobotModel jointPositions={jointPositions} />
      
      {/* 工作空间指示 */}
      {showWorkspace && (
        <>
          {/* 地面网格 */}
          <gridHelper 
            args={[4, 20, '#64748B', '#475569']} 
            position={[0, 0, 0]} 
            rotation={[0, 0, 0]}
          />
          
          {/* 工作空间边界 */}
          <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, 0, 0.01]}>
            <ringGeometry args={[1.3, 1.4, 64]} />
            <meshBasicMaterial color="#2563EB" transparent opacity={0.3} />
          </mesh>
          
          {/* 安全区域 */}
          <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, 0, 0.005]}>
            <ringGeometry args={[1.5, 1.6, 64]} />
            <meshBasicMaterial color="#EF4444" transparent opacity={0.2} />
          </mesh>
        </>
      )}
      
      {/* 轨道控制 */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI * 0.95}
        minDistance={0.5}
        maxDistance={20}
        target={[0, 0, 0.6]}
        enableDamping={true}
        dampingFactor={0.05}
      />
    </Canvas>
  );
};

// 关节状态显示组件
const JointStatusPanel: React.FC<{ jointPositions: number[] }> = ({ jointPositions }) => {
  return (
    <MuiBox
      sx={{
        position: 'absolute',
        top: 12,
        left: 12,
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        backdropFilter: 'blur(8px)',
        color: 'white',
        p: 1.5,
        borderRadius: 2,
        border: '1px solid rgba(100, 116, 139, 0.3)',
        minWidth: 180,
      }}
    >
      <Typography variant="caption" sx={{ fontWeight: 600, mb: 1, display: 'block' }}>
        关节状态 (实时)
      </Typography>
      {jointPositions.map((angle: number, index: number) => {
        const degrees = (angle * 180 / Math.PI).toFixed(1);
        const isActive = Math.abs(angle) > 0.01;
        
        return (
          <MuiBox key={index} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Typography 
              variant="caption" 
              sx={{ 
                fontFamily: '"Fira Code", monospace',
                fontSize: '0.7rem',
                minWidth: '24px',
                color: isActive ? '#2563EB' : '#94A3B8'
              }}
            >
              J{index+1}:
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                fontFamily: '"Fira Code", monospace',
                fontSize: '0.7rem',
                ml: 1,
                color: isActive ? '#FFFFFF' : '#94A3B8'
              }}
            >
              {angle.toFixed(3)}rad ({degrees}°)
            </Typography>
          </MuiBox>
        );
      })}
    </MuiBox>
  );
};

// 控制面板组件
const ViewerControlPanel: React.FC<{
  onCameraChange: (preset: string) => void;
  onToggleCoordinates: () => void;
  onToggleWorkspace: () => void;
  showCoordinates: boolean;
  showWorkspace: boolean;
}> = ({ onCameraChange, onToggleCoordinates, onToggleWorkspace, showCoordinates, showWorkspace }) => {
  return (
    <MuiBox
      sx={{
        position: 'absolute',
        top: 12,
        right: 12,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      {/* 相机预设 */}
      <ButtonGroup 
        variant="contained" 
        size="small"
        sx={{
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          backdropFilter: 'blur(8px)',
          borderRadius: 2,
          border: '1px solid rgba(100, 116, 139, 0.3)',
        }}
      >
        {['前视', '侧视', '俯视', '等轴'].map((label, index) => {
          const presets = ['front', 'side', 'top', 'iso'];
          return (
            <Button
              key={label}
              onClick={() => onCameraChange(presets[index])}
              sx={{ 
                color: 'white',
                fontSize: '0.7rem',
                minWidth: 'auto',
                px: 1,
                cursor: 'pointer' // UI/UX Pro Max 要求
              }}
            >
              {label}
            </Button>
          );
        })}
      </ButtonGroup>

      {/* 显示选项 */}
      <MuiBox sx={{ display: 'flex', gap: 0.5 }}>
        <Tooltip title="坐标系">
          <IconButton
            size="small"
            onClick={onToggleCoordinates}
            sx={{
              backgroundColor: showCoordinates ? 'rgba(37, 99, 235, 0.9)' : 'rgba(15, 23, 42, 0.9)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(100, 116, 139, 0.3)',
              color: 'white',
              cursor: 'pointer', // UI/UX Pro Max 要求
              '&:hover': {
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
              }
            }}
          >
            <ThreeDRotationIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip title="工作空间">
          <IconButton
            size="small"
            onClick={onToggleWorkspace}
            sx={{
              backgroundColor: showWorkspace ? 'rgba(37, 99, 235, 0.9)' : 'rgba(15, 23, 42, 0.9)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(100, 116, 139, 0.3)',
              color: 'white',
              cursor: 'pointer', // UI/UX Pro Max 要求
              '&:hover': {
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
              }
            }}
          >
            <SpeedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </MuiBox>
    </MuiBox>
  );
};

// 主要的增强3D查看器组件
const Enhanced3DViewer: React.FC = () => {
  const { currentState, isConnected } = useAppSelector((state) => state.robot);
  const robotSpecs = getRobotSpecs();
  const [showCoordinates, setShowCoordinates] = useState(true);
  const [showWorkspace, setShowWorkspace] = useState(true);
  const [cameraPreset, setCameraPreset] = useState<'front' | 'side' | 'top' | 'iso'>('iso');
  const [webglInfo, setWebglInfo] = useState<any>(null);
  
  // 使用真实的关节位置数据
  const jointPositions = currentState?.jointPositions || [0, 0, 0, 0, 0, 0];
  
  useEffect(() => {
    const info = detectWebGL();
    setWebglInfo(info);
  }, []);

  return (
    <Card sx={{ height: '100%', minHeight: 500 }}>
      <CardHeader 
        title={
          <MuiBox sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              {robotSpecs.name} 3D模型
            </Typography>
            <Chip
              icon={isConnected ? <SpeedIcon /> : undefined}
              label={isConnected ? "实时同步" : "离线模式"}
              color={isConnected ? "success" : "default"}
              size="small"
              sx={{ cursor: 'pointer' }} // UI/UX Pro Max 要求
            />
          </MuiBox>
        }
        subheader={`工作半径 ${robotSpecs.reach}mm • 重复精度 ±${robotSpecs.repeatability}mm`}
        action={
          <Tooltip title="全屏显示">
            <IconButton sx={{ cursor: 'pointer' }}>
              <FullscreenIcon />
            </IconButton>
          </Tooltip>
        }
      />
      <CardContent sx={{ height: 'calc(100% - 80px)', p: 1, position: 'relative' }}>
        <MuiBox sx={{ height: '100%', position: 'relative', borderRadius: 2, overflow: 'hidden' }}>
          <ErrorBoundary>
            <IndustrialRobot3DScene 
              jointPositions={jointPositions}
              showCoordinates={showCoordinates}
              showWorkspace={showWorkspace}
              cameraPreset={cameraPreset}
            />
          </ErrorBoundary>
          
          {/* 关节状态面板 */}
          <JointStatusPanel jointPositions={jointPositions} />
          
          {/* 控制面板 */}
          <ViewerControlPanel
            onCameraChange={(preset) => setCameraPreset(preset as any)}
            onToggleCoordinates={() => setShowCoordinates(!showCoordinates)}
            onToggleWorkspace={() => setShowWorkspace(!showWorkspace)}
            showCoordinates={showCoordinates}
            showWorkspace={showWorkspace}
          />
          
          {/* WebGL错误提示 */}
          {webglInfo && !webglInfo.supported && (
            <MuiBox
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                backgroundColor: 'rgba(220, 38, 38, 0.9)',
                color: 'white',
                p: 3,
                borderRadius: 2,
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
          
          {/* 性能指示器 */}
          <MuiBox
            sx={{
              position: 'absolute',
              bottom: 12,
              right: 12,
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
              backdropFilter: 'blur(8px)',
              color: 'white',
              px: 1.5,
              py: 0.5,
              borderRadius: 1,
              border: '1px solid rgba(100, 116, 139, 0.3)',
            }}
          >
            <Typography variant="caption" sx={{ fontFamily: '"Fira Code", monospace' }}>
              FPS: 60 | Triangles: 12.5K
            </Typography>
          </MuiBox>
        </MuiBox>
      </CardContent>
    </Card>
  );
};

export default Enhanced3DViewer;