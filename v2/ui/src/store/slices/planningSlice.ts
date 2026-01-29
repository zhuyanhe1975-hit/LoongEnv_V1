import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Waypoint {
  id: string;
  position: [number, number, number];
  orientation: [number, number, number, number];
  velocity?: number;
  timestamp?: number;
}

export interface TrajectoryPlan {
  id: string;
  name: string;
  waypoints: Waypoint[];
  algorithm: 'topp' | 'rrt' | 'quintic' | 'spline';
  parameters: {
    maxVelocity: number;
    maxAcceleration: number;
    maxJerk: number;
    smoothness: number;
  };
  status: 'idle' | 'planning' | 'completed' | 'error';
  results?: {
    trajectoryLength: number;
    executionTime: number;
    smoothness: number;
    energyConsumption: number;
  };
  createdAt: number;
  updatedAt: number;
}

interface PlanningState {
  currentPlan: TrajectoryPlan | null;
  plans: TrajectoryPlan[];
  isPlanning: boolean;
  selectedAlgorithm: 'topp' | 'rrt' | 'quintic' | 'spline';
  parameters: {
    maxVelocity: number;
    maxAcceleration: number;
    maxJerk: number;
    smoothness: number;
  };
  waypoints: Waypoint[];
  planningProgress: number;
  error: string | null;
}

const initialState: PlanningState = {
  currentPlan: null,
  plans: [],
  isPlanning: false,
  selectedAlgorithm: 'topp',
  parameters: {
    maxVelocity: 2.0,
    maxAcceleration: 1.5,
    maxJerk: 3.0,
    smoothness: 0.8,
  },
  waypoints: [],
  planningProgress: 0,
  error: null,
};

const planningSlice = createSlice({
  name: 'planning',
  initialState,
  reducers: {
    setAlgorithm: (state, action: PayloadAction<'topp' | 'rrt' | 'quintic' | 'spline'>) => {
      state.selectedAlgorithm = action.payload;
    },
    
    setParameters: (state, action: PayloadAction<Partial<PlanningState['parameters']>>) => {
      state.parameters = { ...state.parameters, ...action.payload };
    },
    
    addWaypoint: (state, action: PayloadAction<Omit<Waypoint, 'id'>>) => {
      const waypoint: Waypoint = {
        ...action.payload,
        id: `waypoint_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };
      state.waypoints.push(waypoint);
    },
    
    updateWaypoint: (state, action: PayloadAction<{ id: string; updates: Partial<Waypoint> }>) => {
      const index = state.waypoints.findIndex(wp => wp.id === action.payload.id);
      if (index !== -1) {
        state.waypoints[index] = { ...state.waypoints[index], ...action.payload.updates };
      }
    },
    
    removeWaypoint: (state, action: PayloadAction<string>) => {
      state.waypoints = state.waypoints.filter(wp => wp.id !== action.payload);
    },
    
    clearWaypoints: (state) => {
      state.waypoints = [];
    },
    
    startPlanning: (state) => {
      state.isPlanning = true;
      state.planningProgress = 0;
      state.error = null;
    },
    
    updatePlanningProgress: (state, action: PayloadAction<number>) => {
      state.planningProgress = action.payload;
    },
    
    completePlanning: (state, action: PayloadAction<TrajectoryPlan>) => {
      state.isPlanning = false;
      state.planningProgress = 100;
      state.currentPlan = action.payload;
      state.plans.push(action.payload);
    },
    
    failPlanning: (state, action: PayloadAction<string>) => {
      state.isPlanning = false;
      state.planningProgress = 0;
      state.error = action.payload;
    },
    
    selectPlan: (state, action: PayloadAction<string>) => {
      const plan = state.plans.find(p => p.id === action.payload);
      if (plan) {
        state.currentPlan = plan;
        state.waypoints = plan.waypoints;
        state.selectedAlgorithm = plan.algorithm;
        state.parameters = plan.parameters;
      }
    },
    
    deletePlan: (state, action: PayloadAction<string>) => {
      state.plans = state.plans.filter(p => p.id !== action.payload);
      if (state.currentPlan?.id === action.payload) {
        state.currentPlan = null;
      }
    },
    
    clearError: (state) => {
      state.error = null;
    },
  },
});

export const {
  setAlgorithm,
  setParameters,
  addWaypoint,
  updateWaypoint,
  removeWaypoint,
  clearWaypoints,
  startPlanning,
  updatePlanningProgress,
  completePlanning,
  failPlanning,
  selectPlan,
  deletePlan,
  clearError,
} = planningSlice.actions;

export default planningSlice.reducer;