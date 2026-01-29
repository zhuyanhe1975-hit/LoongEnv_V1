import React, { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import STLModel from './STLLoader';

type Vec3 = [number, number, number];

type URDFLinkVisual = {
  xyz: Vec3;
  rpy: Vec3;
  meshUrl: string;
  color?: string;
};

type URDFJoint = {
  name: string;
  parent: string;
  child: string;
  originXyz: Vec3;
  originRpy: Vec3;
  axis: Vec3;
};

type ParsedURDF = {
  root: string;
  links: Map<string, { visual?: URDFLinkVisual }>;
  jointsByParent: Map<string, URDFJoint[]>;
};

const parseVec3 = (value: string | null | undefined, fallback: Vec3 = [0, 0, 0]): Vec3 => {
  if (!value) return fallback;
  const parts = value
    .trim()
    .split(/\s+/g)
    .filter(Boolean)
    .slice(0, 3)
    .map((v) => Number.parseFloat(v));
  if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) return fallback;
  return [parts[0], parts[1], parts[2]];
};

const parseColor = (rgba: string | null | undefined): string | undefined => {
  if (!rgba) return undefined;
  const [r, g, b] = rgba
    .trim()
    .split(/\s+/g)
    .filter(Boolean)
    .slice(0, 3)
    .map((v) => Number.parseFloat(v));
  if ([r, g, b].some((n) => Number.isNaN(n))) return undefined;
  const to255 = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
};

const meshFilenameToUrl = (filename: string): string => {
  const cleaned = filename.trim().replace(/^file:\/\//, '');
  const base = cleaned.replace(/^[.][/\\]/, '').replace(/^\/+/, '');
  return `/models/${base}`;
};

const qFromRPY = (rpy: Vec3) => {
  const [r, p, y] = rpy;
  // URDF rpy 约定：Rz(yaw) * Ry(pitch) * Rx(roll)
  // 用显式四元数相乘，避免 three.js Euler 顺序/内外旋语义差异导致装配错误。
  const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), r);
  const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), p);
  const qz = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), y);
  return qz.multiply(qy).multiply(qx);
};

const parseURDFText = (urdfText: string): ParsedURDF => {
  const xml = new DOMParser().parseFromString(urdfText, 'application/xml');
  const robot = xml.querySelector('robot');
  if (!robot) {
    throw new Error('URDF parse failed: <robot> not found');
  }

  const links = new Map<string, { visual?: URDFLinkVisual }>();
  for (const linkEl of Array.from(robot.querySelectorAll('link'))) {
    const linkName = linkEl.getAttribute('name') || '';
    if (!linkName) continue;

    const visualEl = linkEl.querySelector('visual');
    const originEl = visualEl?.querySelector('origin');
    const meshEl = visualEl?.querySelector('geometry mesh');
    const colorEl = visualEl?.querySelector('material color');

    const filename = meshEl?.getAttribute('filename');
    if (filename) {
      links.set(linkName, {
        visual: {
          xyz: parseVec3(originEl?.getAttribute('xyz')),
          rpy: parseVec3(originEl?.getAttribute('rpy')),
          meshUrl: meshFilenameToUrl(filename),
          color: parseColor(colorEl?.getAttribute('rgba')),
        },
      });
    } else {
      links.set(linkName, {});
    }
  }

  const joints: URDFJoint[] = [];
  const childLinks = new Set<string>();
  for (const jointEl of Array.from(robot.querySelectorAll('joint'))) {
    const name = jointEl.getAttribute('name') || '';
    if (!name) continue;
    const parent = jointEl.querySelector('parent')?.getAttribute('link') || '';
    const child = jointEl.querySelector('child')?.getAttribute('link') || '';
    if (!parent || !child) continue;
    const originEl = jointEl.querySelector('origin');
    const axisEl = jointEl.querySelector('axis');

    joints.push({
      name,
      parent,
      child,
      originXyz: parseVec3(originEl?.getAttribute('xyz')),
      originRpy: parseVec3(originEl?.getAttribute('rpy')),
      axis: parseVec3(axisEl?.getAttribute('xyz'), [0, 0, 1]),
    });
    childLinks.add(child);
  }

  let root = '';
  for (const linkName of links.keys()) {
    if (!childLinks.has(linkName)) {
      root = linkName;
      break;
    }
  }
  if (!root) {
    root = 'base_link';
  }

  const jointsByParent = new Map<string, URDFJoint[]>();
  for (const joint of joints) {
    const list = jointsByParent.get(joint.parent) || [];
    list.push(joint);
    jointsByParent.set(joint.parent, list);
  }

  return { root, links, jointsByParent };
};

export interface URDFRobotModelProps {
  urdfUrl?: string;
  jointPositions?: number[];
  scale?: Vec3;
}

const URDFRobotModel: React.FC<URDFRobotModelProps> = ({
  urdfUrl = '/models/ER15-1400.urdf',
  jointPositions = [0, 0, 0, 0, 0, 0],
  scale = [1, 1, 1],
}) => {
  const [urdfText, setUrdfText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(urdfUrl);
        if (!res.ok) throw new Error(`fetch ${urdfUrl} failed: ${res.status}`);
        const text = await res.text();
        if (!cancelled) {
          setUrdfText(text);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setUrdfText(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [urdfUrl]);

  const parsed = useMemo(() => {
    if (!urdfText) return null;
    try {
      return parseURDFText(urdfText);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [urdfText]);

  const jointAngleByName = useMemo(() => {
    const map: Record<string, number> = {};
    for (let i = 0; i < jointPositions.length; i++) {
      map[`joint_${i + 1}`] = jointPositions[i] || 0;
    }
    return map;
  }, [jointPositions]);

  const renderLink = (linkName: string): React.ReactNode => {
    if (!parsed) return null;

    const link = parsed.links.get(linkName);
    const childJoints = parsed.jointsByParent.get(linkName) || [];

    return (
      <group key={linkName}>
        {link?.visual && (
          <group position={link.visual.xyz} quaternion={qFromRPY(link.visual.rpy)}>
            <STLModel
              url={link.visual.meshUrl}
              position={[0, 0, 0]}
              rotation={[0, 0, 0]}
              scale={scale}
              color={link.visual.color || '#888888'}
            />
          </group>
        )}

        {childJoints.map((joint) => {
          const axisVec = new THREE.Vector3(joint.axis[0], joint.axis[1], joint.axis[2]);
          if (axisVec.lengthSq() < 1e-12) axisVec.set(0, 0, 1);
          axisVec.normalize();

          const angle = jointAngleByName[joint.name] ?? 0;
          const jointAxisQuat = new THREE.Quaternion().setFromAxisAngle(axisVec, angle);

          return (
            <group
              key={joint.name}
              position={joint.originXyz}
              quaternion={qFromRPY(joint.originRpy)}
            >
              <group quaternion={jointAxisQuat}>
                {renderLink(joint.child)}
              </group>
            </group>
          );
        })}
      </group>
    );
  };

  if (error) {
    return (
      <group>
        <mesh>
          <boxGeometry args={[0.2, 0.2, 0.2]} />
          <meshStandardMaterial color="#ff4444" />
        </mesh>
      </group>
    );
  }

  if (!parsed) {
    return (
      <group>
        <mesh>
          <boxGeometry args={[0.15, 0.15, 0.15]} />
          <meshStandardMaterial color="#cccccc" opacity={0.35} transparent />
        </mesh>
      </group>
    );
  }

  return <group>{renderLink(parsed.root)}</group>;
};

export default URDFRobotModel;
