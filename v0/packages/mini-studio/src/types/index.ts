// 项目数据类型
export interface MiniProject {
  id: string;
  name: string;
  description: string;
  robotType: 'industrial_6axis';
  createdAt: string;
  updatedAt: string;
}

// DH参数
export interface DHParameter {
  joint: number;
  a: number;    // 连杆长度 (mm)
  d: number;    // 连杆偏移 (mm)
  theta: number; // 关节角度 (度)
  alpha: number; // 连杆扭转角 (度)
}

// 关节限制
export interface JointLimit {
  joint: number;
  min: number;  // 最小角度 (度)
  max: number;  // 最大角度 (度)
}

// 机械臂配置
export interface MiniRobot {
  dhParams: DHParameter[];
  jointLimits: JointLimit[];
}

// 轨迹点
export interface TrajectoryPoint {
  id: string;
  name: string;
  position: [number, number, number]; // [x, y, z] (mm)
  orientation?: [number, number, number]; // [rx, ry, rz] (度)
}

// 仿真指标
export interface SimulationMetrics {
  maxSpeed: number;     // 最大速度 (mm/s)
  avgSpeed: number;     // 平均速度 (mm/s)
  totalTime: number;    // 总时间 (s)
  pathLength: number;   // 路径长度 (mm)
}

// 仿真配置
export interface MiniSimulation {
  duration: number;
  speed: number;
  points: TrajectoryPoint[];
  metrics: SimulationMetrics;
}

// 位姿
export interface Pose {
  position: [number, number, number];
  orientation: [number, number, number];
}

// 应用状态
export interface AppState {
  // 当前项目
  currentProject: MiniProject | null;
  
  // 机械臂配置
  robot: MiniRobot;
  
  // 轨迹点
  trajectoryPoints: TrajectoryPoint[];
  
  // 仿真状态
  simulation: {
    isRunning: boolean;
    currentTime: number;
    totalTime: number;
    speed: number;
    metrics: SimulationMetrics | null;
  };
  
  // UI状态
  ui: {
    activeTab: 'project' | 'robot' | 'trajectory' | 'simulation';
    selectedPoint: string | null;
  };
}
