// 机器人状态类型定义
export interface RobotState {
  timestamp: number;
  isConnected: boolean;
  jointPositions: number[];
  jointVelocities: number[];
  jointTorques: number[];
  endEffectorPose: number[]; // [x, y, z, rx, ry, rz]
  operationMode: OperationMode;
  safetyStatus: SafetyStatus;
  errorCode: string | null;
  systemLoad: {
    cpu: number;
    memory: number;
    temperature: number;
  };
}

export type SafetyStatus = 'safe' | 'warning' | 'error' | 'offline';
export type OperationMode = 'manual' | 'automatic' | 'simulation' | 'maintenance';

// 轨迹相关类型
export interface TrajectoryPoint {
  position: number[];
  velocity: number[];
  acceleration: number[];
  jerk: number[];
  time: number;
  pathParameter: number;
}

export interface Trajectory {
  id: string;
  name: string;
  description?: string;
  points: TrajectoryPoint[];
  duration: number;
  createdAt: Date;
  updatedAt: Date;
}

// 参数调优相关类型
export interface TuningConfig {
  parameterType: 'control_gains' | 'trajectory_params' | 'vibration_params';
  optimizationMethod: 'differential_evolution' | 'grid_search' | 'bayesian';
  maxIterations: number;
  populationSize: number;
  targetMetrics: string[];
  constraints: Record<string, [number, number]>;
}

export interface TuningResult {
  taskId: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  bestParameters: Record<string, number>;
  bestPerformance: number;
  optimizationHistory: number[];
  startTime: Date;
  endTime?: Date;
}

// 系统监控相关类型
export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkLatency: number;
  controlLoopFrequency: number;
  algorithmExecutionTime: Record<string, number>;
  timestamp: number;
}

export interface CollisionStatus {
  isCollisionDetected: boolean;
  collisionPairs: Array<{
    link1: string;
    link2: string;
    distance: number;
    minDistance: number;
  }>;
  safetyMargin: number;
}

// API响应类型
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: number;
}

// WebSocket消息类型
export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: number;
}

// 用户界面状态类型
export interface UIState {
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  notifications: Notification[];
  selectedRobot?: string;
  currentPage: string;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  autoHide?: boolean;
}

// 图表数据类型
export interface ChartDataPoint {
  timestamp: number;
  value: number;
  label?: string;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
  unit?: string;
}

// 3D可视化相关类型
export interface RobotModel {
  id: string;
  name: string;
  joints: JointInfo[];
  links: LinkInfo[];
  meshUrls: string[];
}

export interface JointInfo {
  id: string;
  name: string;
  type: 'revolute' | 'prismatic' | 'fixed';
  axis: [number, number, number];
  limits: {
    lower: number;
    upper: number;
    velocity: number;
    effort: number;
  };
  currentPosition: number;
  currentVelocity: number;
  currentTorque: number;
}

export interface LinkInfo {
  id: string;
  name: string;
  mass: number;
  centerOfMass: [number, number, number];
  inertia: number[][];
  meshUrl?: string;
  color?: string;
}