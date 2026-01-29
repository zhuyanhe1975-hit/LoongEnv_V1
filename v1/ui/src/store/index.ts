import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';

import robotReducer from './slices/robotSlice';
import uiReducer from './slices/uiSlice';
import monitoringReducer from './slices/monitoringSlice';
import planningReducer from './slices/planningSlice';
import tuningReducer from './slices/tuningSlice';

export const store = configureStore({
  reducer: {
    robot: robotReducer,
    ui: uiReducer,
    monitoring: monitoringReducer,
    planning: planningReducer,
    tuning: tuningReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
        ignoredPaths: ['robot.timestamp', 'monitoring.lastUpdate'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;