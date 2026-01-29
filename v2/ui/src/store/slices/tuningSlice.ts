import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface PIDParameters {
  kp: number;
  ki: number;
  kd: number;
}

export interface DynamicsParameters {
  mass: number;
  friction: number;
  damping: number;
  inertia: number[];
}

export interface SafetyParameters {
  maxVelocity: number;
  maxAcceleration: number;
  maxTorque: number;
  emergencyStopThreshold: number;
  workspaceLimit: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

export interface TuningSession {
  id: string;
  name: string;
  timestamp: number;
  parameters: {
    pid: PIDParameters;
    dynamics: DynamicsParameters;
    safety: SafetyParameters;
  };
  results: {
    performance: number;
    stability: number;
    efficiency: number;
    responseTime: number;
  };
  status: 'running' | 'completed' | 'failed';
}

export interface OptimizationResult {
  iteration: number;
  parameters: PIDParameters;
  performance: number;
  timestamp: number;
}

interface TuningState {
  currentSession: TuningSession | null;
  sessions: TuningSession[];
  
  // Current parameters
  pidParameters: PIDParameters;
  dynamicsParameters: DynamicsParameters;
  safetyParameters: SafetyParameters;
  
  // Auto-tuning state
  isAutoTuning: boolean;
  autoTuningProgress: number;
  optimizationHistory: OptimizationResult[];
  
  // Manual tuning
  isApplyingParameters: boolean;
  lastAppliedAt: number | null;
  
  // Performance monitoring
  currentPerformance: {
    responseTime: number;
    overshoot: number;
    settlingTime: number;
    steadyStateError: number;
  };
  
  // UI state
  selectedTab: number;
  showAdvancedSettings: boolean;
  
  error: string | null;
}

const initialState: TuningState = {
  currentSession: null,
  sessions: [],
  
  pidParameters: {
    kp: 1.2,
    ki: 0.8,
    kd: 0.3,
  },
  
  dynamicsParameters: {
    mass: 50.0,
    friction: 0.1,
    damping: 0.05,
    inertia: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
  },
  
  safetyParameters: {
    maxVelocity: 2.0,
    maxAcceleration: 5.0,
    maxTorque: 100.0,
    emergencyStopThreshold: 150.0,
    workspaceLimit: {
      min: [-1.0, -1.0, 0.0],
      max: [1.0, 1.0, 2.0],
    },
  },
  
  isAutoTuning: false,
  autoTuningProgress: 0,
  optimizationHistory: [],
  
  isApplyingParameters: false,
  lastAppliedAt: null,
  
  currentPerformance: {
    responseTime: 0.0,
    overshoot: 0.0,
    settlingTime: 0.0,
    steadyStateError: 0.0,
  },
  
  selectedTab: 0,
  showAdvancedSettings: false,
  
  error: null,
};

const tuningSlice = createSlice({
  name: 'tuning',
  initialState,
  reducers: {
    // PID parameter updates
    setPIDParameters: (state, action: PayloadAction<Partial<PIDParameters>>) => {
      state.pidParameters = { ...state.pidParameters, ...action.payload };
    },
    
    setPIDParameter: (state, action: PayloadAction<{ param: keyof PIDParameters; value: number }>) => {
      state.pidParameters[action.payload.param] = action.payload.value;
    },
    
    // Dynamics parameter updates
    setDynamicsParameters: (state, action: PayloadAction<Partial<DynamicsParameters>>) => {
      state.dynamicsParameters = { ...state.dynamicsParameters, ...action.payload };
    },
    
    // Safety parameter updates
    setSafetyParameters: (state, action: PayloadAction<Partial<SafetyParameters>>) => {
      state.safetyParameters = { ...state.safetyParameters, ...action.payload };
    },
    
    // Auto-tuning
    startAutoTuning: (state) => {
      state.isAutoTuning = true;
      state.autoTuningProgress = 0;
      state.optimizationHistory = [];
      state.error = null;
    },
    
    updateAutoTuningProgress: (state, action: PayloadAction<number>) => {
      state.autoTuningProgress = action.payload;
    },
    
    addOptimizationResult: (state, action: PayloadAction<OptimizationResult>) => {
      state.optimizationHistory.push(action.payload);
      // Update current parameters with best result
      if (action.payload.performance > (state.optimizationHistory[state.optimizationHistory.length - 2]?.performance || 0)) {
        state.pidParameters = action.payload.parameters;
      }
    },
    
    completeAutoTuning: (state, action: PayloadAction<PIDParameters>) => {
      state.isAutoTuning = false;
      state.autoTuningProgress = 100;
      state.pidParameters = action.payload;
    },
    
    failAutoTuning: (state, action: PayloadAction<string>) => {
      state.isAutoTuning = false;
      state.autoTuningProgress = 0;
      state.error = action.payload;
    },
    
    // Parameter application
    startApplyingParameters: (state) => {
      state.isApplyingParameters = true;
      state.error = null;
    },
    
    completeApplyingParameters: (state) => {
      state.isApplyingParameters = false;
      state.lastAppliedAt = Date.now();
    },
    
    failApplyingParameters: (state, action: PayloadAction<string>) => {
      state.isApplyingParameters = false;
      state.error = action.payload;
    },
    
    // Performance monitoring
    updatePerformance: (state, action: PayloadAction<Partial<TuningState['currentPerformance']>>) => {
      state.currentPerformance = { ...state.currentPerformance, ...action.payload };
    },
    
    // Session management
    startTuningSession: (state, action: PayloadAction<{ name: string }>) => {
      const session: TuningSession = {
        id: `session_${Date.now()}`,
        name: action.payload.name,
        timestamp: Date.now(),
        parameters: {
          pid: { ...state.pidParameters },
          dynamics: { ...state.dynamicsParameters },
          safety: { ...state.safetyParameters },
        },
        results: {
          performance: 0,
          stability: 0,
          efficiency: 0,
          responseTime: 0,
        },
        status: 'running',
      };
      state.currentSession = session;
      state.sessions.push(session);
    },
    
    updateSessionResults: (state, action: PayloadAction<Partial<TuningSession['results']>>) => {
      if (state.currentSession) {
        state.currentSession.results = { ...state.currentSession.results, ...action.payload };
        // Update in sessions array
        const index = state.sessions.findIndex(s => s.id === state.currentSession!.id);
        if (index !== -1) {
          state.sessions[index] = { ...state.currentSession };
        }
      }
    },
    
    completeSession: (state) => {
      if (state.currentSession) {
        state.currentSession.status = 'completed';
        const index = state.sessions.findIndex(s => s.id === state.currentSession!.id);
        if (index !== -1) {
          state.sessions[index] = { ...state.currentSession };
        }
      }
    },
    
    loadSession: (state, action: PayloadAction<string>) => {
      const session = state.sessions.find(s => s.id === action.payload);
      if (session) {
        state.pidParameters = session.parameters.pid;
        state.dynamicsParameters = session.parameters.dynamics;
        state.safetyParameters = session.parameters.safety;
        state.currentSession = session;
      }
    },
    
    deleteSession: (state, action: PayloadAction<string>) => {
      state.sessions = state.sessions.filter(s => s.id !== action.payload);
      if (state.currentSession?.id === action.payload) {
        state.currentSession = null;
      }
    },
    
    // UI state
    setSelectedTab: (state, action: PayloadAction<number>) => {
      state.selectedTab = action.payload;
    },
    
    toggleAdvancedSettings: (state) => {
      state.showAdvancedSettings = !state.showAdvancedSettings;
    },
    
    // Reset and defaults
    resetToDefaults: (state) => {
      state.pidParameters = initialState.pidParameters;
      state.dynamicsParameters = initialState.dynamicsParameters;
      state.safetyParameters = initialState.safetyParameters;
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Backend integration actions
    setTuningParameters: (state, action: PayloadAction<any>) => {
      // Update parameters from backend
      if (action.payload.pid) {
        state.pidParameters = { ...state.pidParameters, ...action.payload.pid };
      }
      if (action.payload.dynamics) {
        state.dynamicsParameters = { ...state.dynamicsParameters, ...action.payload.dynamics };
      }
      if (action.payload.safety) {
        state.safetyParameters = { ...state.safetyParameters, ...action.payload.safety };
      }
    },
    
    setTuningStatus: (state, action: PayloadAction<{ running: boolean; progress: number }>) => {
      state.isAutoTuning = action.payload.running;
      state.autoTuningProgress = action.payload.progress;
    },
    
    setTuningResults: (state, action: PayloadAction<any>) => {
      if (action.payload.success) {
        state.isAutoTuning = false;
        state.autoTuningProgress = 100;
        
        // Update parameters with optimal results
        if (action.payload.results?.control_gains?.optimalParameters) {
          const optimal = action.payload.results.control_gains.optimalParameters;
          if (optimal.kp && optimal.ki && optimal.kd) {
            state.pidParameters = {
              kp: Array.isArray(optimal.kp) ? optimal.kp[0] : optimal.kp,
              ki: Array.isArray(optimal.ki) ? optimal.ki[0] : optimal.ki,
              kd: Array.isArray(optimal.kd) ? optimal.kd[0] : optimal.kd,
            };
          }
        }
        
        // Update performance metrics
        if (action.payload.results?.control_gains?.bestPerformance) {
          state.currentPerformance = {
            ...state.currentPerformance,
            responseTime: action.payload.results.control_gains.bestPerformance,
          };
        }
      } else {
        state.isAutoTuning = false;
        state.error = action.payload.error || 'Tuning failed';
      }
    },
    
    updateTuningProgress: (state, action: PayloadAction<number>) => {
      state.autoTuningProgress = action.payload;
    },
  },
});

export const {
  setPIDParameters,
  setPIDParameter,
  setDynamicsParameters,
  setSafetyParameters,
  startAutoTuning,
  updateAutoTuningProgress,
  addOptimizationResult,
  completeAutoTuning,
  failAutoTuning,
  startApplyingParameters,
  completeApplyingParameters,
  failApplyingParameters,
  updatePerformance,
  startTuningSession,
  updateSessionResults,
  completeSession,
  loadSession,
  deleteSession,
  setSelectedTab,
  toggleAdvancedSettings,
  resetToDefaults,
  clearError,
  setTuningParameters,
  setTuningStatus,
  setTuningResults,
  updateTuningProgress,
} = tuningSlice.actions;

export default tuningSlice.reducer;