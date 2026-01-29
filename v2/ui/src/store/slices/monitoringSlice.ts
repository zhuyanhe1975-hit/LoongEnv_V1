import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SystemMetrics, CollisionStatus, ChartDataPoint } from '../../types/robot';

interface MonitoringSliceState {
  systemMetrics: SystemMetrics | null;
  collisionStatus: CollisionStatus | null;
  performanceData: {
    cpuUsage: ChartDataPoint[];
    memoryUsage: ChartDataPoint[];
    networkLatency: ChartDataPoint[];
    controlFrequency: ChartDataPoint[];
  };
  isMonitoring: boolean;
  lastUpdate: number;
}

const initialState: MonitoringSliceState = {
  systemMetrics: null,
  collisionStatus: null,
  performanceData: {
    cpuUsage: [],
    memoryUsage: [],
    networkLatency: [],
    controlFrequency: [],
  },
  isMonitoring: false,
  lastUpdate: 0,
};

const monitoringSlice = createSlice({
  name: 'monitoring',
  initialState,
  reducers: {
    updateSystemMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.systemMetrics = action.payload;
      state.lastUpdate = Date.now();
      
      // Update performance data arrays
      const timestamp = action.payload.timestamp;
      const maxDataPoints = 100;
      
      // Add new data points
      state.performanceData.cpuUsage.push({
        timestamp,
        value: action.payload.cpuUsage,
      });
      
      state.performanceData.memoryUsage.push({
        timestamp,
        value: action.payload.memoryUsage,
      });
      
      state.performanceData.networkLatency.push({
        timestamp,
        value: action.payload.networkLatency,
      });
      
      state.performanceData.controlFrequency.push({
        timestamp,
        value: action.payload.controlLoopFrequency,
      });
      
      // Keep only the last N data points
      Object.keys(state.performanceData).forEach((key) => {
        const dataArray = state.performanceData[key as keyof typeof state.performanceData];
        if (dataArray.length > maxDataPoints) {
          state.performanceData[key as keyof typeof state.performanceData] = 
            dataArray.slice(-maxDataPoints);
        }
      });
    },
    
    updateCollisionStatus: (state, action: PayloadAction<CollisionStatus>) => {
      state.collisionStatus = action.payload;
      state.lastUpdate = Date.now();
    },
    
    startMonitoring: (state) => {
      state.isMonitoring = true;
    },
    
    stopMonitoring: (state) => {
      state.isMonitoring = false;
    },
    
    clearPerformanceData: (state) => {
      state.performanceData = {
        cpuUsage: [],
        memoryUsage: [],
        networkLatency: [],
        controlFrequency: [],
      };
    },
    
    resetMonitoring: () => {
      return initialState;
    },
  },
});

export const {
  updateSystemMetrics,
  updateCollisionStatus,
  startMonitoring,
  stopMonitoring,
  clearPerformanceData,
  resetMonitoring,
} = monitoringSlice.actions;

export default monitoringSlice.reducer;