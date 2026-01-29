import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RobotState, SafetyStatus, OperationMode } from '../../types/robot';

interface RobotSliceState {
  currentState: RobotState | null;
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastUpdate: number;
  selectedRobotId?: string;
  modelUrdfUrl: string;
  modelName: string;
}

const initialState: RobotSliceState = {
  currentState: null,
  isConnected: false,
  connectionStatus: 'disconnected',
  lastUpdate: 0,
  modelUrdfUrl: '/models/ER15-1400.urdf',
  modelName: 'ER15-1400',
};

const robotSlice = createSlice({
  name: 'robot',
  initialState,
  reducers: {
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
      state.connectionStatus = action.payload ? 'connected' : 'disconnected';
    },
    
    updateRobotState: (state, action: PayloadAction<RobotState>) => {
      state.currentState = action.payload;
      state.isConnected = action.payload.isConnected;
      state.lastUpdate = Date.now();
    },

    setRobotModel: (state, action: PayloadAction<{ urdfUrl: string; name?: string }>) => {
      state.modelUrdfUrl = action.payload.urdfUrl;
      if (action.payload.name) {
        state.modelName = action.payload.name;
        return;
      }
      const basename = action.payload.urdfUrl.split('/').pop() || 'Robot';
      state.modelName = basename.replace(/\.urdf$/i, '');
    },
    
    updateJointPositions: (state, action: PayloadAction<number[]>) => {
      if (state.currentState) {
        state.currentState.jointPositions = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    updateJointVelocities: (state, action: PayloadAction<number[]>) => {
      if (state.currentState) {
        state.currentState.jointVelocities = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    updateJointTorques: (state, action: PayloadAction<number[]>) => {
      if (state.currentState) {
        state.currentState.jointTorques = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    updateEndEffectorPose: (state, action: PayloadAction<RobotState['endEffectorPose']>) => {
      if (state.currentState) {
        state.currentState.endEffectorPose = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    updateSafetyStatus: (state, action: PayloadAction<SafetyStatus>) => {
      if (state.currentState) {
        state.currentState.safetyStatus = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    updateOperationMode: (state, action: PayloadAction<OperationMode>) => {
      if (state.currentState) {
        state.currentState.operationMode = action.payload;
        state.lastUpdate = Date.now();
      }
    },
    
    selectRobot: (state, action: PayloadAction<string>) => {
      state.selectedRobotId = action.payload;
    },
    
    resetRobotState: (state) => {
      state.currentState = null;
      state.isConnected = false;
      state.connectionStatus = 'disconnected';
      state.lastUpdate = 0;
    },
  },
});

export const {
  setConnectionStatus,
  updateRobotState,
  setRobotModel,
  updateJointPositions,
  updateJointVelocities,
  updateJointTorques,
  updateEndEffectorPose,
  updateSafetyStatus,
  updateOperationMode,
  selectRobot,
  resetRobotState,
} = robotSlice.actions;

export default robotSlice.reducer;
