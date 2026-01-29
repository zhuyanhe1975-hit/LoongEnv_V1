import React from 'react';
import { backendService } from '../../services/backendService';
import { useAppDispatch, useAppSelector } from '../../store';
import { setBackendTuningResults, setBackendTuningState } from '../../store/slices/tuningSlice';

const TuningStatusPoller: React.FC = () => {
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((state) => state.robot.isConnected);

  React.useEffect(() => {
    if (!isConnected) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const status = await backendService.getTuningStatus();
        if (cancelled) return;

        dispatch(
          setBackendTuningState({
            status: status.running ? 'running' : 'idle',
            progress: status.progress,
          })
        );

        if (!status.running && status.results) {
          dispatch(setBackendTuningResults(status.results));
        }
      } catch (error) {
        if (cancelled) return;
        dispatch(
          setBackendTuningState({
            status: 'error',
            error: error instanceof Error ? error.message : String(error),
          })
        );
      }
    };

    tick();
    const interval = window.setInterval(tick, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [dispatch, isConnected]);

  return null;
};

export default TuningStatusPoller;

