import React from 'react';
import { Box, Chip, Tooltip } from '@mui/material';
import {
  Wifi as ConnectedIcon,
  WifiOff as DisconnectedIcon,
  Sync as ConnectingIcon,
} from '@mui/icons-material';

interface ConnectionStatusProps {
  isConnected: boolean;
  isConnecting?: boolean;
  showLabel?: boolean;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  isConnecting = false,
  showLabel = true,
}) => {
  const getStatusConfig = () => {
    if (isConnecting) {
      return {
        color: 'warning' as const,
        icon: <ConnectingIcon />,
        label: '连接中',
        tooltip: '正在连接到机器人控制器...',
      };
    }
    
    if (isConnected) {
      return {
        color: 'success' as const,
        icon: <ConnectedIcon />,
        label: '已连接',
        tooltip: '机器人控制器连接正常',
      };
    }
    
    return {
      color: 'error' as const,
      icon: <DisconnectedIcon />,
      label: '未连接',
      tooltip: '机器人控制器连接断开',
    };
  };

  const config = getStatusConfig();

  if (!showLabel) {
    return (
      <Tooltip title={config.tooltip}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            color: `${config.color}.main`,
          }}
        >
          {config.icon}
        </Box>
      </Tooltip>
    );
  }

  return (
    <Tooltip title={config.tooltip}>
      <Chip
        icon={config.icon}
        label={config.label}
        color={config.color}
        size="small"
        variant="outlined"
        sx={{ mr: 2 }}
      />
    </Tooltip>
  );
};

export default ConnectionStatus;