import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box } from '@react-three/drei';

const Test3D: React.FC = () => {
  console.log('Test3D component rendering...');
  
  return (
    <div style={{ 
      width: '100%', 
      height: '400px', 
      border: '2px solid red',
      backgroundColor: '#f0f0f0'
    }}>
      <Canvas
        onCreated={(state) => {
          console.log('Canvas created successfully:', state);
        }}
        onError={(error) => {
          console.error('Canvas error:', error);
        }}
        style={{ 
          width: '100%', 
          height: '100%',
          display: 'block'
        }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} />
        <Box args={[1, 1, 1]} position={[0, 0, 0]}>
          <meshStandardMaterial color="hotpink" />
        </Box>
        <OrbitControls />
      </Canvas>
    </div>
  );
};

export default Test3D;