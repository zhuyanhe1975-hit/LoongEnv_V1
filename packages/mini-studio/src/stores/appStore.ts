import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { AppState, MiniProject, TrajectoryPoint, DHParameter, JointLimit } from '@/types';

// 默认6轴机械臂DH参数 (标准工业机械臂)
const defaultDHParams: DHParameter[] = [
  { joint: 1, a: 0, d: 0, theta: 0, alpha: 90 },
  { joint: 2, a: 420, d: 0, theta: -90, alpha: 0 },
  { joint: 3, a: 400, d: 0, theta: 0, alpha: 90 },
  { joint: 4, a: 0, d: 380, theta: 0, alpha: -90 },
  { joint: 5, a: 0, d: 0, theta: 0, alpha: 90 },
  { joint: 6, a: 0, d: 65, theta: 0, alpha: 0 },
];

// 默认关节限制
const defaultJointLimits: JointLimit[] = [
  { joint: 1, min: -180, max: 180 },
  { joint: 2, min: -90, max: 135 },
  { joint: 3, min: -180, max: 70 },
  { joint: 4, min: -180, max: 180 },
  { joint: 5, min: -120, max: 120 },
  { joint: 6, min: -360, max: 360 },
];

interface AppStore extends AppState {
  // 项目操作
  createProject: (name: string, description: string) => void;
  loadProject: (project: MiniProject) => void;
  updateProject: (updates: Partial<MiniProject>) => void;
  
  // 机械臂操作
  updateDHParam: (joint: number, param: Partial<DHParameter>) => void;
  updateJointLimit: (joint: number, limit: Partial<JointLimit>) => void;
  resetRobotToDefault: () => void;
  
  // 轨迹操作
  addTrajectoryPoint: (point: Omit<TrajectoryPoint, 'id'>) => void;
  updateTrajectoryPoint: (id: string, updates: Partial<TrajectoryPoint>) => void;
  deleteTrajectoryPoint: (id: string) => void;
  clearTrajectory: () => void;
  
  // 仿真操作
  startSimulation: () => void;
  stopSimulation: () => void;
  updateSimulationTime: (time: number) => void;
  setSimulationSpeed: (speed: number) => void;
  
  // UI操作
  setActiveTab: (tab: AppState['ui']['activeTab']) => void;
  selectPoint: (id: string | null) => void;
}

export const useAppStore = create<AppStore>()(
  subscribeWithSelector((set, get) => ({
      // 初始状态
      currentProject: null,
      robot: {
        dhParams: defaultDHParams,
        jointLimits: defaultJointLimits,
      },
      trajectoryPoints: [],
      simulation: {
        isRunning: false,
        currentTime: 0,
        totalTime: 10,
        speed: 1,
        metrics: null,
      },
      ui: {
        activeTab: 'project',
        selectedPoint: null,
      },

      // 项目操作
      createProject: (name, description) => {
        const project: MiniProject = {
          id: Date.now().toString(),
          name,
          description,
          robotType: 'industrial_6axis',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        set({ currentProject: project });
      },

      loadProject: (project) => {
        set({ currentProject: project });
      },

      updateProject: (updates) => {
        const { currentProject } = get();
        if (currentProject) {
          set({
            currentProject: {
              ...currentProject,
              ...updates,
              updatedAt: new Date().toISOString(),
            },
          });
        }
      },

      // 机械臂操作
      updateDHParam: (joint, param) => {
        const { robot } = get();
        const newDHParams = robot.dhParams.map(p =>
          p.joint === joint ? { ...p, ...param } : p
        );
        set({
          robot: { ...robot, dhParams: newDHParams },
        });
      },

      updateJointLimit: (joint, limit) => {
        const { robot } = get();
        const newJointLimits = robot.jointLimits.map(l =>
          l.joint === joint ? { ...l, ...limit } : l
        );
        set({
          robot: { ...robot, jointLimits: newJointLimits },
        });
      },

      resetRobotToDefault: () => {
        set({
          robot: {
            dhParams: defaultDHParams,
            jointLimits: defaultJointLimits,
          },
        });
      },

      // 轨迹操作
      addTrajectoryPoint: (point) => {
        const { trajectoryPoints } = get();
        const newPoint: TrajectoryPoint = {
          ...point,
          id: Date.now().toString(),
        };
        set({
          trajectoryPoints: [...trajectoryPoints, newPoint],
        });
      },

      updateTrajectoryPoint: (id, updates) => {
        const { trajectoryPoints } = get();
        const newPoints = trajectoryPoints.map(p =>
          p.id === id ? { ...p, ...updates } : p
        );
        set({ trajectoryPoints: newPoints });
      },

      deleteTrajectoryPoint: (id) => {
        const { trajectoryPoints } = get();
        set({
          trajectoryPoints: trajectoryPoints.filter(p => p.id !== id),
        });
      },

      clearTrajectory: () => {
        set({ trajectoryPoints: [] });
      },

      // 仿真操作
      startSimulation: () => {
        set(state => ({
          simulation: { ...state.simulation, isRunning: true, currentTime: 0 },
        }));
      },

      stopSimulation: () => {
        set(state => ({
          simulation: { ...state.simulation, isRunning: false },
        }));
      },

      updateSimulationTime: (time) => {
        set(state => ({
          simulation: { ...state.simulation, currentTime: time },
        }));
      },

      setSimulationSpeed: (speed) => {
        set(state => ({
          simulation: { ...state.simulation, speed },
        }));
      },

      // UI操作
      setActiveTab: (tab) => {
        set(state => ({
          ui: { ...state.ui, activeTab: tab },
        }));
      },

      selectPoint: (id) => {
        set(state => ({
          ui: { ...state.ui, selectedPoint: id },
        }));
      },
    })
  )
);
