import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Circle as CircleIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  NetworkCheck as NetworkIcon,
} from '@mui/icons-material';

import { useAppSelector } from '../../store';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';

const StatusBar: React.FC = () => {
  const { currentState, isConnected, lastUpdate } = useAppSelector((state) => state.robot);
  const { currentPage } = useAppSelector((state) => state.ui);

  const getStatusColor = () => {
    if (!isConnected) return 'error';
    if (!currentState) return 'warning';
    
    switch (currentState.safetyStatus) {
      case 'safe': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusText = () => {
    if (!isConnected) return '未连接';
    if (!currentState) return '等待数据';
    
    switch (currentState.safetyStatus) {
      case 'safe': return '安全';
      case 'warning': return '警告';
      case 'error': return '错误';
      case 'offline': return '离线';
      default: return '未知';
    }
  };

  const formatLastUpdate = () => {
    if (!lastUpdate) return '从未更新';
    try {
      return formatDistanceToNow(new Date(lastUpdate), { 
        addSuffix: true,
        locale: zhCN 
      });
    } catch {
      return '更新时间未知';
    }
  };

  return (
    <Box
      sx={{
        height: 40,
        backgroundColor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        px: 2,
        gap: 2,
        flexShrink: 0,
      }}
    >
      {/* Robot Status */}
      <Tooltip title="机器人状态">
        <Chip
          icon={<CircleIcon />}
          label={getStatusText()}
          color={getStatusColor()}
          size="small"
          variant="outlined"
        />
      </Tooltip>

      <Divider orientation="vertical" flexItem />

      {/* Operation Mode */}
      {currentState && (
        <>
          <Tooltip title="运行模式">
            <Typography variant="caption" color="text.secondary">
              模式: {currentState.operationMode === 'manual' ? '手动' : 
                    currentState.operationMode === 'automatic' ? '自动' :
                    currentState.operationMode === 'simulation' ? '仿真' : '维护'}
            </Typography>
          </Tooltip>
          
          <Divider orientation="vertical" flexItem />
        </>
      )}

      {/* System Metrics */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Tooltip title="控制频率">
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <SpeedIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              1000Hz
            </Typography>
          </Box>
        </Tooltip>

        <Tooltip title="内存使用">
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <MemoryIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              85MB
            </Typography>
          </Box>
        </Tooltip>

        <Tooltip title="网络延迟">
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <NetworkIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              2ms
            </Typography>
          </Box>
        </Tooltip>
      </Box>

      {/* Spacer */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Current Page */}
      <Typography variant="caption" color="text.secondary">
        当前页面: {currentPage}
      </Typography>

      <Divider orientation="vertical" flexItem />

      {/* Last Update */}
      <Tooltip title="最后更新时间">
        <Typography variant="caption" color="text.secondary">
          {formatLastUpdate()}
        </Typography>
      </Tooltip>
    </Box>
  );
};

export default StatusBar;