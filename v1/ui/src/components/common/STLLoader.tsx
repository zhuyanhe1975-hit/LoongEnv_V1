import React, { useRef, useEffect, useState, useMemo } from 'react';
import { STLLoader } from 'three-stdlib';
import * as THREE from 'three';

interface STLModelProps {
  url: string;
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: [number, number, number];
  color?: string;
  opacity?: number;
}

const STLModel: React.FC<STLModelProps> = ({
  url,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = [1, 1, 1],
  color = '#2196F3',
  opacity = 1.0
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 使用useMemo来缓存材质，避免重复创建
  const material = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: color,
      opacity: opacity,
      transparent: opacity < 1.0,
      side: THREE.DoubleSide,
      roughness: 0.3,
      metalness: 0.1,
    });
  }, [color, opacity]);

  useEffect(() => {
    let isMounted = true;
    const loader = new STLLoader();
    
    console.log(`Loading STL model: ${url}`);
    
    loader.load(
      url,
      (loadedGeometry: THREE.BufferGeometry) => {
        if (!isMounted) return;
        
        console.log(`STL model loaded successfully: ${url}`);
        
        // 计算边界框
        loadedGeometry.computeBoundingBox();
        const boundingBox = loadedGeometry.boundingBox;
        
        if (boundingBox) {
          const size = new THREE.Vector3();
          boundingBox.getSize(size);
          console.log(`STL model size: ${size.x.toFixed(3)} x ${size.y.toFixed(3)} x ${size.z.toFixed(3)}`);
          
          // 获取几何体中心点
          const center = new THREE.Vector3();
          boundingBox.getCenter(center);
          console.log(`STL model center: ${center.x.toFixed(3)}, ${center.y.toFixed(3)}, ${center.z.toFixed(3)}`);
        }
        
        // 计算法向量
        loadedGeometry.computeVertexNormals();
        
        setGeometry(loadedGeometry);
        setLoading(false);
        setError(null);
      },
      (progress: ProgressEvent<EventTarget>) => {
        if (!isMounted) return;
        if (progress.total > 0) {
          const percent = (progress.loaded / progress.total) * 100;
          console.log(`STL loading progress (${url}): ${percent.toFixed(1)}%`);
        }
      },
      (err: unknown) => {
        if (!isMounted) return;
        console.error(`STL loading error (${url}):`, err);
        setError(`Failed to load ${url}`);
        setLoading(false);
      }
    );

    return () => {
      isMounted = false;
    };
  }, [url]);

  if (loading) {
    // 加载中显示半透明占位几何体
    return (
      <mesh ref={meshRef} position={position} rotation={rotation} scale={scale}>
        <boxGeometry args={[0.1, 0.1, 0.1]} />
        <meshStandardMaterial color="#cccccc" opacity={0.3} transparent wireframe />
      </mesh>
    );
  }

  if (error || !geometry) {
    // 错误时显示红色线框几何体
    console.warn(`Using fallback geometry for: ${url}`);
    return (
      <mesh ref={meshRef} position={position} rotation={rotation} scale={scale}>
        <boxGeometry args={[0.08, 0.08, 0.08]} />
        <meshStandardMaterial color="#ff4444" opacity={0.7} transparent wireframe />
      </mesh>
    );
  }

  return (
    <mesh
      ref={meshRef}
      position={position}
      rotation={rotation}
      scale={scale}
      geometry={geometry}
      material={material}
    />
  );
};

export default STLModel;
