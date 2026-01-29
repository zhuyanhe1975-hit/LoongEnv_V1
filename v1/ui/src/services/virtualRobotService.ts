import { RobotState } from '../types/robot';
import { forwardKinematics, checkJointLimits, ER15_1400_URDF, getRobotSpecs } from '../utils/urdfParser';

export interface VirtualRobotConfig {
  updateInterval: number; // 更新间隔 (ms)
  enableNoise: boolean;   // 是否添加噪声
  simulationSpeed: number; // 仿真速度倍率
}

export class VirtualRobotService {
  private isConnected = false;
  private updateInterval: number | null = null;
  private config: VirtualRobotConfig;
  private currentJointPositions: number[] = [0, 0, 0, 0, 0, 0];
  private targetJointPositions: number[] = [0, 0, 0, 0, 0, 0];
  private jointVelocities: number[] = [0, 0, 0, 0, 0, 0];
  private callbacks: ((state: RobotState) => void)[] = [];
  private startTime = Date.now();
  private robotSpecs = getRobotSpecs();

  constructor(config: Partial<VirtualRobotConfig> = {}) {
    this.config = {
      updateInterval: 100, // 降低到10Hz减少闪烁
      enableNoise: false,  // 默认关闭噪声
      simulationSpeed: 1.0,
      ...config,
    };
  }

  // 连接虚拟机器人
  connect(): Promise<boolean> {
    return new Promise((resolve) => {
      if (this.isConnected) {
        resolve(true);
        return;
      }

      // 模拟连接延迟
      setTimeout(() => {
        this.isConnected = true;
        this.startSimulation();
        resolve(true);
      }, 1000);
    });
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
    
    // 返回取消订阅函数
    return () => {
      const index = this.callbacks.indexOf(callback);
      if (index > -1) {
        this.callbacks.splice(index, 1);
      }
    };
  }

  // 设置目标关节位置（带限制检查）
  setTargetJointPositions(positions: number[]): void {
    if (positions.length === 6) {
      // 检查关节限制并限制到安全范围
      const clampedPositions = positions.map((pos, index) => {
        const joint = ER15_1400_URDF.joints[index];
        return Math.max(joint.limits.lower, Math.min(joint.limits.upper, pos));
      });
      
      this.targetJointPositions = clampedPositions;
    }
  }

  // 执行轨迹
  executeTrajectory(trajectory: { positions: number[]; time: number }[]): void {
    // 简单的轨迹执行模拟
    trajectory.forEach((point) => {
      setTimeout(() => {
        this.setTargetJointPositions(point.positions);
      }, point.time * 1000);
    });
  }

  // 启动仿真循环
  private startSimulation(): void {
    this.updateInterval = setInterval(() => {
      this.updateSimulation();
    }, this.config.updateInterval);
  }

  // 更新仿真状态
  private updateSimulation(): void {
    if (!this.isConnected) return;

    // 模拟关节运动 - 改进的PD控制器
    const dt = this.config.updateInterval / 1000;
    const kp = 3.0; // 比例增益
    const kd = 0.8; // 微分增益

    for (let i = 0; i < 6; i++) {
      const error = this.targetJointPositions[i] - this.currentJointPositions[i];
      const velocity = this.jointVelocities[i];
      
      // PD控制
      const acceleration = kp * error - kd * velocity;
      
      // 更新速度和位置
      this.jointVelocities[i] += acceleration * dt;
      this.currentJointPositions[i] += this.jointVelocities[i] * dt;

      // 添加噪声
      if (this.config.enableNoise) {
        this.currentJointPositions[i] += (Math.random() - 0.5) * 0.001;
        this.jointVelocities[i] += (Math.random() - 0.5) * 0.01;
      }
      
      // 应用关节限制
      const joint = ER15_1400_URDF.joints[i];
      this.currentJointPositions[i] = Math.max(
        joint.limits.lower, 
        Math.min(joint.limits.upper, this.currentJointPositions[i])
      );
    }

    // 生成更真实的运动模式
    const time = (Date.now() - this.startTime) / 1000;
    
    // 关节1: 缓慢旋转
    this.targetJointPositions[0] = Math.sin(time * 0.3) * 1.5;
    
    // 关节2: 上下摆动
    this.targetJointPositions[1] = Math.sin(time * 0.4) * 0.8 - 0.5;
    
    // 关节3: 配合关节2的运动
    this.targetJointPositions[2] = Math.cos(time * 0.4) * 0.6 + 0.3;
    
    // 关节4-6: 末端姿态调整
    this.targetJointPositions[3] = Math.sin(time * 0.6) * 0.5;
    this.targetJointPositions[4] = Math.cos(time * 0.5) * 0.4;
    this.targetJointPositions[5] = Math.sin(time * 0.8) * 1.0;

    // 使用真实的正向运动学计算末端执行器位置
    const endEffectorPose = forwardKinematics(this.currentJointPositions);

    // 计算关节力矩（简化模型）
    const jointTorques = this.currentJointPositions.map((pos, index) => {
      const link = ER15_1400_URDF.links[index + 1]; // +1因为base_link
      const mass = link ? link.inertial.mass : 1.0;
      const gravity = 9.81;
      
      // 简化的重力补偿力矩
      const gravityTorque = mass * gravity * Math.sin(pos) * 0.1;
      
      // 添加一些动态力矩
      const dynamicTorque = this.jointVelocities[index] * 0.5;
      
      return gravityTorque + dynamicTorque + (Math.random() - 0.5) * 2;
    });

    // 生成虚拟状态数据
    const robotState: RobotState = {
      timestamp: Date.now(),
      isConnected: true,
      jointPositions: [...this.currentJointPositions],
      jointVelocities: [...this.jointVelocities],
      jointTorques: jointTorques,
      endEffectorPose,
      operationMode: 'automatic',
      safetyStatus: this.checkSafetyStatus(),
      errorCode: null,
      systemLoad: {
        cpu: 30 + Math.random() * 20,
        memory: 50 + Math.random() * 30,
        temperature: 35 + Math.random() * 10,
      },
    };

    // 通知所有订阅者
    this.callbacks.forEach(callback => callback(robotState));
  }

  // 安全状态检查
  private checkSafetyStatus(): 'safe' | 'warning' | 'error' | 'offline' {
    // 检查关节限制
    const limitsOk = checkJointLimits(this.currentJointPositions);
    const hasLimitViolation = limitsOk.some(ok => !ok);
    
    // 检查速度限制
    const maxVelocity = Math.max(...this.jointVelocities.map(Math.abs));
    const velocityWarning = maxVelocity > 2.0;
    
    // 随机错误模拟
    const randomError = Math.random() > 0.998;
    
    if (randomError || hasLimitViolation) {
      return 'error';
    } else if (velocityWarning) {
      return 'warning';
    } else {
      return 'safe';
    }
  }

  // 获取当前状态快照
  getCurrentState(): RobotState | null {
    if (!this.isConnected) return null;

    const endEffectorPose = forwardKinematics(this.currentJointPositions);
    
    return {
      timestamp: Date.now(),
      isConnected: true,
      jointPositions: [...this.currentJointPositions],
      jointVelocities: [...this.jointVelocities],
      jointTorques: this.currentJointPositions.map(() => 0),
      endEffectorPose,
      operationMode: 'automatic',
      safetyStatus: 'safe',
      errorCode: null,
      systemLoad: {
        cpu: 45,
        memory: 68,
        temperature: 42,
      },
    };
  }

  // 获取机器人规格信息
  getRobotSpecs() {
    return this.robotSpecs;
  }
}

// 全局虚拟机器人服务实例
export const virtualRobotService = new VirtualRobotService();