/**
 * 后端服务连接器
 * 
 * 连接React UI和Python后端API，提供真实的机器人控制功能
 */

import { RobotState } from '../types/robot';

const API_BASE_URL = '/api';

export interface BackendConfig {
  baseUrl: string;
  timeout: number;
}

export interface TuningConfig {
  method: 'differential_evolution' | 'gradient_descent' | 'basin_hopping';
  maxIterations: number;
  tolerance: number;
  populationSize: number;
  parameterTypes: string[];
  performanceWeights: {
    trackingAccuracy: number;
    settlingTime: number;
    overshoot: number;
    energyEfficiency: number;
    vibrationSuppression: number;
    safetyMargin: number;
  };
}

export interface TuningStatus {
  running: boolean;
  progress: number;
  results: any;
}

export interface TrajectoryPlanRequest {
  waypoints: Array<{
    position: number[];
    velocity?: number[];
    time?: number;
  }>;
  optimizeTime?: boolean;
  trajectoryParams?: {
    maxVelocity?: number;
    maxAcceleration?: number;
    maxJerk?: number;
    smoothness?: number;
  };
}

export interface ControlGains {
  kp: number[];
  ki: number[];
  kd: number[];
  controlMode: string;
}

export class BackendService {
  private config: BackendConfig;
  private isConnected = false;
  private callbacks: ((state: RobotState) => void)[] = [];
  private updateInterval: number | null = null;
  private readonly statusUpdateIntervalMs = 50;

  constructor(config: Partial<BackendConfig> = {}) {
    this.config = {
      baseUrl: API_BASE_URL,
      timeout: 10000,
      ...config,
    };
  }

  // 连接后端服务
  async connect(): Promise<boolean> {
    try {
      const response = await this.request('/health');
      
      if (response.status === 'healthy') {
        this.isConnected = true;
        this.startStatusUpdates();
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Backend connection failed:', error);
      return false;
    }
  }

  // 断开连接
  disconnect(): void {
    this.isConnected = false;
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  // 检查连接状态
  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  // 订阅状态更新
  onStateUpdate(callback: (state: RobotState) => void): () => void {
    this.callbacks.push(callback);
    
    return () => {
      const index = this.callbacks.indexOf(callback);
      if (index > -1) {
        this.callbacks.splice(index, 1);
      }
    };
  }

  // 获取机器人状态
  async getRobotStatus(): Promise<RobotState> {
    const response = await this.request('/robot/status');
    return {
      timestamp: response.timestamp,
      isConnected: response.isConnected,
      jointPositions: response.jointPositions,
      jointVelocities: response.jointVelocities,
      jointTorques: response.jointTorques,
      endEffectorPose: response.endEffectorPose || [0, 0, 0, 0, 0, 0],
      operationMode: response.operationMode,
      safetyStatus: response.safetyStatus,
      errorCode: response.errorCode,
      systemLoad: response.systemLoad,
    };
  }

  // 获取机器人规格
  async getRobotSpecs(): Promise<any> {
    return await this.request('/robot/specs');
  }

  // 启动参数调优
  async startParameterTuning(config: TuningConfig): Promise<void> {
    await this.request('/tuning/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // 获取调优状态
  async getTuningStatus(): Promise<TuningStatus> {
    return await this.request('/tuning/status');
  }

  // 停止参数调优
  async stopParameterTuning(): Promise<void> {
    await this.request('/tuning/stop', { method: 'POST' });
  }

  // 规划轨迹
  async planTrajectory(request: TrajectoryPlanRequest): Promise<any> {
    return await this.request('/trajectory/plan', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // 获取当前轨迹
  async getCurrentTrajectory(): Promise<any> {
    return await this.request('/trajectory/current');
  }

  // 获取控制增益
  async getControlGains(): Promise<ControlGains> {
    return await this.request('/control/gains');
  }

  // 设置控制增益
  async setControlGains(gains: Partial<ControlGains>): Promise<void> {
    await this.request('/control/gains', {
      method: 'POST',
      body: JSON.stringify(gains),
    });
  }

  // 获取性能指标
  async getPerformanceMetrics(): Promise<any> {
    return await this.request('/monitoring/performance');
  }

  // 启动仿真
  async startSimulation(duration: number = 10.0): Promise<void> {
    await this.request('/simulation/start', {
      method: 'POST',
      body: JSON.stringify({ duration }),
    });
  }

  // 停止仿真
  async stopSimulation(): Promise<void> {
    await this.request('/simulation/stop', { method: 'POST' });
  }

  // 对比优化前后性能
  async comparePerformance(originalParams: any, optimizedParams: any): Promise<any> {
    return await this.request('/tuning/compare', {
      method: 'POST',
      body: JSON.stringify({
        originalParams,
        optimizedParams,
      }),
    });
  }

  // 通用请求方法
  private async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  // 启动状态更新循环
  private startStatusUpdates(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    this.updateInterval = setInterval(async () => {
      if (!this.isConnected) return;

      try {
        const state = await this.getRobotStatus();
        this.callbacks.forEach(callback => callback(state));
      } catch (error) {
        console.error('Failed to update robot status:', error);
        // 连接失败时断开
        this.disconnect();
      }
    }, this.statusUpdateIntervalMs); // 20Hz：更平滑的实时姿态更新
  }
}

// 全局后端服务实例
export const backendService = new BackendService();
