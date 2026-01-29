// URDF解析器 - 提取ER15-1400机器人参数
export interface URDFJoint {
  name: string;
  type: 'revolute' | 'prismatic' | 'fixed';
  parent: string;
  child: string;
  origin: {
    xyz: [number, number, number];
    rpy: [number, number, number];
  };
  axis: [number, number, number];
  limits: {
    lower: number;
    upper: number;
    effort: number;
    velocity: number;
  };
}

export interface URDFLink {
  name: string;
  inertial: {
    origin: {
      xyz: [number, number, number];
      rpy: [number, number, number];
    };
    mass: number;
    inertia: {
      ixx: number;
      ixy: number;
      ixz: number;
      iyy: number;
      iyz: number;
      izz: number;
    };
  };
  visual: {
    origin: {
      xyz: [number, number, number];
      rpy: [number, number, number];
    };
    geometry: {
      mesh?: {
        filename: string;
      };
    };
    material: {
      color: [number, number, number, number]; // RGBA
    };
  };
}

export interface RobotURDF {
  name: string;
  links: URDFLink[];
  joints: URDFJoint[];
}

// ER15-1400机器人的URDF参数（从真实URDF文件提取）
export const ER15_1400_URDF: RobotURDF = {
  name: "ER15-1400",
  links: [
    {
      name: "base_link",
      inertial: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
        mass: 100,
        inertia: {
          ixx: 0, ixy: 0, ixz: 0,
          iyy: 0, iyz: 0, izz: 100
        }
      },
      visual: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
        geometry: { mesh: { filename: "./b_link.STL" } },
        material: { color: [1.0, 0, 0, 1.0] }
      }
    },
    {
      name: "link_1",
      inertial: {
        origin: { xyz: [0.09835, -0.02908, -0.0995], rpy: [0, 0, 0] },
        mass: 54.52,
        inertia: {
          ixx: 1.16916852, ixy: 0.0865367, ixz: -0.47354118,
          iyy: 1.39934751, iyz: 0.11859959, izz: 1.00920236
        }
      },
      visual: {
        origin: { xyz: [0, 0, -0.43], rpy: [0, 0, 0] },
        geometry: { mesh: { filename: "./l_1.STL" } },
        material: { color: [0, 0, 0.6, 1.0] }
      }
    },
    {
      name: "link_2",
      inertial: {
        origin: { xyz: [0.25263, -0.00448, 0.15471], rpy: [0, 0, 0] },
        mass: 11.11,
        inertia: {
          ixx: 0.04507715, ixy: -0.00764148, ixz: -0.01800527,
          iyy: 0.58269106, iyz: 0.00057833, izz: 0.60235638
        }
      },
      visual: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, -1.5707963267] },
        geometry: { mesh: { filename: "./l_2.STL" } },
        material: { color: [1.0, 0, 0, 1.0] }
      }
    },
    {
      name: "link_3",
      inertial: {
        origin: { xyz: [0.03913, -0.02495, 0.03337], rpy: [0, 0, 0] },
        mass: 25.03,
        inertia: {
          ixx: 0.33717585, ixy: 0.06955124, ixz: 0.00142677,
          iyy: 0.38576036, iyz: -0.00313441, izz: 0.24095087
        }
      },
      visual: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, -1.5707963267] },
        geometry: { mesh: { filename: "./l_3.STL" } },
        material: { color: [0, 0, 0.8, 1.0] }
      }
    },
    {
      name: "link_4",
      inertial: {
        origin: { xyz: [-0.00132, -0.0012, -0.30035], rpy: [0, 0, 0] },
        mass: 10.81,
        inertia: {
          ixx: 0.28066314, ixy: -0.00003381, ixz: 0.00084678,
          iyy: 0.27142738, iyz: 0.00437676, izz: 0.04425281
        }
      },
      visual: {
        origin: { xyz: [0, 0, -0.64], rpy: [3.141592653, 0, -1.5707963267] },
        geometry: { mesh: { filename: "./l_4.STL" } },
        material: { color: [0, 0.9, 0.9, 1.0] }
      }
    },
    {
      name: "link_5",
      inertial: {
        origin: { xyz: [0.0004, -0.03052, 0.01328], rpy: [0, 0, 0] },
        mass: 4.48,
        inertia: {
          ixx: 0.01710138, ixy: -0.00002606, ixz: 0.00000867,
          iyy: 0.01098115, iyz: -0.00175535, izz: 0.01408541
        }
      },
      visual: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, -1.5707963267] },
        geometry: { mesh: { filename: "./l_5.STL" } },
        material: { color: [1.0, 0, 0, 1.0] }
      }
    },
    {
      name: "link_6",
      inertial: {
        origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
        mass: 0.28,
        inertia: {
          ixx: 0.0001346961, ixy: 0.0000076, ixz: -0.00000827,
          iyy: 0.0001645611, iyz: 0.000118982, izz: 0.001539171
        }
      },
      visual: {
        origin: { xyz: [0, 0, 0], rpy: [3.141592653, 0, 1.5707963267] },
        geometry: { mesh: { filename: "./l_6.STL" } },
        material: { color: [0.9, 0.9, 0.9, 1.0] }
      }
    }
  ],
  joints: [
    {
      name: "joint_1",
      type: "revolute",
      parent: "base_link",
      child: "link_1",
      origin: { xyz: [0, 0, 0.43], rpy: [0, 0, 0] },
      axis: [0, 0, 1],
      limits: { lower: -2.967, upper: 2.967, effort: 0, velocity: 0 }
    },
    {
      name: "joint_2",
      type: "revolute",
      parent: "link_1",
      child: "link_2",
      origin: { xyz: [0.18, 0, 0], rpy: [1.5707963267, -1.5707963267, 0] },
      axis: [0, 0, 1],
      limits: { lower: -2.7925, upper: 1.5707963267, effort: 0, velocity: 0 }
    },
    {
      name: "joint_3",
      type: "revolute",
      parent: "link_2",
      child: "link_3",
      origin: { xyz: [0.58, 0, 0], rpy: [0, 0, 0] },
      axis: [0, 0, 1],
      limits: { lower: -1.4835, upper: 3.0543, effort: 0, velocity: 0 }
    },
    {
      name: "joint_4",
      type: "revolute",
      parent: "link_3",
      child: "link_4",
      origin: { xyz: [0.16, -0.64, 0], rpy: [-1.5707963267, 0, 3.141592653] },
      axis: [0, 0, 1],
      limits: { lower: -3.316, upper: 3.316, effort: 0, velocity: 0 }
    },
    {
      name: "joint_5",
      type: "revolute",
      parent: "link_4",
      child: "link_5",
      origin: { xyz: [0, 0, 0], rpy: [-1.5707963267, 0, 3.141592653] },
      axis: [0, 0, 1],
      limits: { lower: -2.2689, upper: 2.2689, effort: 0, velocity: 0 }
    },
    {
      name: "joint_6",
      type: "revolute",
      parent: "link_5",
      child: "link_6",
      origin: { xyz: [0, -0.116, 0], rpy: [1.5707963267, 0, 0] },
      axis: [0, 0, 1],
      limits: { lower: -6.2832, upper: 6.2832, effort: 0, velocity: 0 }
    }
  ]
};

// DH参数计算（基于URDF数据）
export interface DHParameters {
  a: number;      // 连杆长度
  alpha: number;  // 连杆扭转角
  d: number;      // 连杆偏移
  theta: number;  // 关节角度（变量）
}

// ER15-1400的DH参数表
export const ER15_1400_DH_PARAMS: DHParameters[] = [
  { a: 0.18,  alpha: Math.PI/2,  d: 0.43,  theta: 0 },  // Joint 1
  { a: 0.58,  alpha: 0,          d: 0,     theta: 0 },  // Joint 2  
  { a: 0.16,  alpha: -Math.PI/2, d: 0,     theta: 0 },  // Joint 3
  { a: 0,     alpha: Math.PI/2,  d: 0.64,  theta: 0 },  // Joint 4
  { a: 0,     alpha: -Math.PI/2, d: 0,     theta: 0 },  // Joint 5
  { a: 0,     alpha: 0,          d: 0.116, theta: 0 }   // Joint 6
];

// 正向运动学计算（基于DH参数）
export function forwardKinematics(jointAngles: number[]): number[] {
  if (jointAngles.length !== 6) {
    throw new Error('ER15-1400需要6个关节角度');
  }

  // 使用DH参数计算正向运动学
  let T = createIdentityMatrix();
  
  for (let i = 0; i < 6; i++) {
    const dh = ER15_1400_DH_PARAMS[i];
    const theta = jointAngles[i] + dh.theta;
    
    const Ti = createDHTransform(dh.a, dh.alpha, dh.d, theta);
    T = multiplyMatrices(T, Ti);
  }
  
  // 提取位置和姿态
  const x = T[0][3];
  const y = T[1][3];
  const z = T[2][3];
  
  // 提取欧拉角 (简化版本)
  const rx = Math.atan2(T[2][1], T[2][2]);
  const ry = Math.atan2(-T[2][0], Math.sqrt(T[2][1]**2 + T[2][2]**2));
  const rz = Math.atan2(T[1][0], T[0][0]);
  
  return [x, y, z, rx, ry, rz];
}

// 创建DH变换矩阵
function createDHTransform(a: number, alpha: number, d: number, theta: number): number[][] {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  
  return [
    [ct, -st*ca,  st*sa, a*ct],
    [st,  ct*ca, -ct*sa, a*st],
    [0,   sa,     ca,    d   ],
    [0,   0,      0,     1   ]
  ];
}

// 创建单位矩阵
function createIdentityMatrix(): number[][] {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

// 矩阵乘法
function multiplyMatrices(A: number[][], B: number[][]): number[][] {
  const result = Array(4).fill(0).map(() => Array(4).fill(0));
  
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  
  return result;
}

// 工作空间计算
export function calculateWorkspace(): {
  maxReach: number;
  minReach: number;
  height: number;
} {
  // 基于DH参数计算工作空间
  const L1 = ER15_1400_DH_PARAMS[0].d;  // 0.43m
  const L2 = ER15_1400_DH_PARAMS[1].a;  // 0.58m  
  const L3 = ER15_1400_DH_PARAMS[2].a;  // 0.16m
  const L4 = ER15_1400_DH_PARAMS[3].d;  // 0.64m
  const L6 = ER15_1400_DH_PARAMS[5].d;  // 0.116m
  
  const maxReach = L2 + L3 + L4 + L6;  // 最大工作半径
  const minReach = Math.abs(L2 - L3 - L4 - L6);  // 最小工作半径
  const height = L1 + L4 + L6;  // 最大工作高度
  
  return {
    maxReach: maxReach,
    minReach: minReach, 
    height: height
  };
}

// 关节限制检查
export function checkJointLimits(jointAngles: number[]): boolean[] {
  return jointAngles.map((angle, index) => {
    const joint = ER15_1400_URDF.joints[index];
    return angle >= joint.limits.lower && angle <= joint.limits.upper;
  });
}

// 获取机器人规格信息
export function getRobotSpecs() {
  const workspace = calculateWorkspace();
  const totalMass = ER15_1400_URDF.links.reduce((sum, link) => sum + link.inertial.mass, 0);
  
  return {
    name: "ER15-1400",
    manufacturer: "Elite Robot",
    dof: 6,
    payload: 15, // kg
    reach: Math.round(workspace.maxReach * 1000), // mm
    repeatability: 0.1, // mm
    totalMass: Math.round(totalMass), // kg
    workspace: workspace,
    jointLimits: ER15_1400_URDF.joints.map(joint => ({
      name: joint.name,
      lower: joint.limits.lower,
      upper: joint.limits.upper,
      range: joint.limits.upper - joint.limits.lower
    }))
  };
}