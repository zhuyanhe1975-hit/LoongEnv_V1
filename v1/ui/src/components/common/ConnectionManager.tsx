import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material';
import {
  Wifi as ConnectIcon,
  WifiOff as DisconnectIcon,
  SmartToy as RobotIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setConnectionStatus, updateRobotState } from '../../store/slices/robotSlice';
import { backendService } from '../../services/backendService';
import { getRobotSpecs } from '../../utils/urdfParser';

interface ConnectionManagerProps {
  open: boolean;
  onClose: () => void;
}

const ConnectionManager: React.FC<ConnectionManagerProps> = ({ open, onClose }) => {
  const dispatch = useAppDispatch();
  const { isConnected } = useAppSelector((state) => state.robot);
  
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [robotSpecs, setRobotSpecs] = useState<any>(null);

  useEffect(() => {
    // 订阅后端服务状态更新
    const unsubscribe = backendService.onStateUpdate((state) => {
      dispatch(updateRobotState(state));
    });

    // 加载机器人规格
    const loadSpecs = async () => {
      try {
        const specs = await backendService.getRobotSpecs();
        setRobotSpecs(specs);
      } catch (error) {
        // 使用本地规格作为备用
        setRobotSpecs(getRobotSpecs());
      }
    };

    loadSpecs();
    return unsubscribe;
  }, [dispatch]);

  const handleConnect = async () => {
    setIsConnecting(true);
    setConnectionError(null);

    try {
      const success = await backendService.connect();
      if (success) {
        dispatch(setConnectionStatus(true));
        onClose();
      } else {
        setConnectionError('连接后端服务失败');
      }
    } catch (error) {
      setConnectionError('连接过程中发生错误: ' + (error as Error).message);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleDisconnect = () => {
    backendService.disconnect();
    dispatch(setConnectionStatus(false));
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <RobotIcon color="primary" />
          <Typography variant="h6">
            机器人连接管理
          </Typography>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {/* 连接状态 */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            连接状态
          </Typography>
          <Alert 
            severity={isConnected ? 'success' : 'warning'}
            icon={isConnected ? <ConnectIcon /> : <DisconnectIcon />}
          >
            {isConnected ? '虚拟机器人已连接' : '虚拟机器人未连接'}
          </Alert>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* 机器人信息 */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            机器人信息
          </Typography>
          {robotSpecs ? (
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
              <Typography variant="body2" color="text.secondary">
                机器人型号:
              </Typography>
              <Typography variant="body2">
                {robotSpecs.name}
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                自由度:
              </Typography>
              <Typography variant="body2">
                {robotSpecs.dof}轴
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                负载能力:
              </Typography>
              <Typography variant="body2">
                {robotSpecs.payload}kg
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                工作半径:
              </Typography>
              <Typography variant="body2">
                {robotSpecs.reach}mm
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                重复精度:
              </Typography>
              <Typography variant="body2">
                ±{robotSpecs.repeatability}mm
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                机器人重量:
              </Typography>
              <Typography variant="body2">
                {robotSpecs.totalMass}kg
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                控制模式:
              </Typography>
              <Typography variant="body2">
                Python后端 + 实时仿真
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                运动学模型:
              </Typography>
              <Typography variant="body2">
                基于URDF + DH参数
              </Typography>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              加载机器人信息中...
            </Typography>
          )}
        </Box>

        {/* 错误信息 */}
        {connectionError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {connectionError}
          </Alert>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>
          取消
        </Button>
        
        {isConnected ? (
          <Button
            onClick={handleDisconnect}
            color="error"
            variant="contained"
            startIcon={<DisconnectIcon />}
          >
            断开连接
          </Button>
        ) : (
          <Button
            onClick={handleConnect}
            color="primary"
            variant="contained"
            disabled={isConnecting}
            startIcon={isConnecting ? <CircularProgress size={20} /> : <ConnectIcon />}
          >
            {isConnecting ? '连接中...' : '连接机器人系统'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default ConnectionManager;
