import React, { useEffect, useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { Box, Typography, Card, CardContent, Button, Alert } from '@mui/material';
import { detectWebGL } from '../../utils/webglDetection';
import * as THREE from 'three';

// 最简单的3D组件
const SimpleCube: React.FC = () => {
  console.log('SimpleCube rendering...');
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color="red" />
    </mesh>
  );
};

// 诊断查看器
const DiagnosticViewer: React.FC = () => {
  const [webglInfo, setWebglInfo] = useState<any>(null);
  const [canvasError, setCanvasError] = useState<string | null>(null);
  const [renderTest, setRenderTest] = useState<'none' | 'basic' | 'advanced'>('none');
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const info = detectWebGL();
    setWebglInfo(info);
    console.log('WebGL Detection:', info);
  }, []);

  useEffect(() => {
    // 检查容器尺寸
    if (containerRef.current && renderTest !== 'none') {
      const rect = containerRef.current.getBoundingClientRect();
      console.log('Canvas Container Size:', {
        width: rect.width,
        height: rect.height,
        offsetWidth: containerRef.current.offsetWidth,
        offsetHeight: containerRef.current.offsetHeight,
        clientWidth: containerRef.current.clientWidth,
        clientHeight: containerRef.current.clientHeight
      });
    }
  }, [renderTest]);

  const handleCanvasError = (error: any) => {
    console.error('Canvas Error:', error);
    setCanvasError(error.toString());
  };

  const handleCanvasCreated = (state: any) => {
    console.log('Canvas Created Successfully:', {
      renderer: state.gl.getParameter(state.gl.RENDERER),
      version: state.gl.getParameter(state.gl.VERSION),
      vendor: state.gl.getParameter(state.gl.VENDOR),
      canvas: state.gl.canvas,
      size: state.size,
      canvasSize: {
        width: state.gl.canvas.width,
        height: state.gl.canvas.height,
        clientWidth: state.gl.canvas.clientWidth,
        clientHeight: state.gl.canvas.clientHeight
      }
    });
  };

  return (
    <Card sx={{ m: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          3D渲染诊断工具
        </Typography>
        
        {/* WebGL信息 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            WebGL支持状态:
          </Typography>
          {webglInfo ? (
            <Box sx={{ pl: 2 }}>
              <Typography variant="body2">
                支持: {webglInfo.supported ? '✅ 是' : '❌ 否'}
              </Typography>
              {webglInfo.supported && (
                <>
                  <Typography variant="body2">版本: {webglInfo.version}</Typography>
                  <Typography variant="body2">渲染器: {webglInfo.renderer}</Typography>
                  <Typography variant="body2">供应商: {webglInfo.vendor}</Typography>
                  <Typography variant="body2">最大纹理尺寸: {webglInfo.maxTextureSize}</Typography>
                </>
              )}
              {webglInfo.error && (
                <Typography variant="body2" color="error">
                  错误: {webglInfo.error}
                </Typography>
              )}
            </Box>
          ) : (
            <Typography variant="body2">检测中...</Typography>
          )}
        </Box>

        {/* Three.js信息 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Three.js信息:
          </Typography>
          <Typography variant="body2" sx={{ pl: 2 }}>
            版本: r{THREE.REVISION}
          </Typography>
        </Box>

        {/* 渲染测试按钮 */}
        <Box sx={{ mb: 2 }}>
          <Button 
            variant="outlined" 
            onClick={() => setRenderTest('basic')}
            sx={{ mr: 1 }}
          >
            基础渲染测试
          </Button>
          <Button 
            variant="outlined" 
            onClick={() => setRenderTest('advanced')}
            sx={{ mr: 1 }}
          >
            高级渲染测试
          </Button>
          <Button 
            variant="outlined" 
            onClick={() => {
              setRenderTest('none');
              setCanvasError(null);
            }}
          >
            清除测试
          </Button>
        </Box>

        {/* 错误显示 */}
        {canvasError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Canvas错误: {canvasError}
          </Alert>
        )}

        {/* 渲染测试区域 */}
        {renderTest !== 'none' && (
          <Box 
            ref={containerRef}
            sx={{ 
              width: '100%',
              height: 300, 
              border: '2px solid #ccc', 
              borderRadius: 1,
              position: 'relative',
              backgroundColor: '#f5f5f5',
              display: 'block'
            }}
          >
            <Typography 
              variant="caption" 
              sx={{ 
                position: 'absolute', 
                top: 4, 
                left: 8, 
                zIndex: 10,
                backgroundColor: 'rgba(255,255,255,0.8)',
                padding: '2px 4px',
                borderRadius: 1
              }}
            >
              {renderTest === 'basic' ? '基础测试' : '高级测试'}
            </Typography>
            
            {renderTest === 'basic' && (
              <Canvas
                style={{ 
                  width: '100%', 
                  height: '100%',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  left: 0
                }}
                onCreated={handleCanvasCreated}
                onError={handleCanvasError}
                gl={{ 
                  antialias: true,
                  alpha: true,
                  preserveDrawingBuffer: true
                }}
              >
                <SimpleCube />
              </Canvas>
            )}
            
            {renderTest === 'advanced' && (
              <Canvas
                camera={{ position: [2, 2, 2] }}
                style={{ 
                  width: '100%', 
                  height: '100%',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  left: 0
                }}
                onCreated={handleCanvasCreated}
                onError={handleCanvasError}
                gl={{ 
                  antialias: true,
                  alpha: true,
                  preserveDrawingBuffer: true
                }}
              >
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} />
                <SimpleCube />
                <mesh position={[2, 0, 0]}>
                  <sphereGeometry args={[0.5]} />
                  <meshStandardMaterial color="blue" />
                </mesh>
                <mesh position={[-2, 0, 0]}>
                  <cylinderGeometry args={[0.5, 0.5, 1]} />
                  <meshStandardMaterial color="green" />
                </mesh>
              </Canvas>
            )}
          </Box>
        )}

        {/* 浏览器信息 */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            浏览器信息:
          </Typography>
          <Typography variant="body2" sx={{ pl: 2, fontSize: '0.8rem', wordBreak: 'break-all' }}>
            User Agent: {navigator.userAgent}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default DiagnosticViewer;