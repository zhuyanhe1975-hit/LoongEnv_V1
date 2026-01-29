import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  SmartToy,
  Battery90,
  Thermostat,
} from '@mui/icons-material';

const RobotStatusCard: React.FC = () => {
  // Mock data for demonstration
  const robotStatus = {
    status: 'running',
    battery: 85,
    temperature: 42,
    position: [0.5, 0.3, 1.2],
    velocity: 0.8,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'idle':
        return 'info';
      case 'error':
        return 'error';
      case 'maintenance':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中';
      case 'idle':
        return '空闲';
      case 'error':
        return '错误';
      case 'maintenance':
        return '维护中';
      default:
        return '未知';
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <SmartToy sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6">机器人状态</Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Chip 
            label={getStatusLabel(robotStatus.status)}
            color={getStatusColor(robotStatus.status) as any}
            variant="filled"
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Battery90 sx={{ mr: 1, fontSize: 20, color: 'success.main' }} />
            <Typography variant="body2">
              电池: {robotStatus.battery}%
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={robotStatus.battery} 
            color="success"
            sx={{ height: 6, borderRadius: 3 }}
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Thermostat sx={{ mr: 1, fontSize: 20, color: 'warning.main' }} />
            <Typography variant="body2">
              温度: {robotStatus.temperature}°C
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={(robotStatus.temperature / 80) * 100} 
            color="warning"
            sx={{ height: 6, borderRadius: 3 }}
          />
        </Box>

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            当前位置 (m):
          </Typography>
          <Typography variant="body2" fontFamily="monospace">
            X: {robotStatus.position[0].toFixed(2)} | 
            Y: {robotStatus.position[1].toFixed(2)} | 
            Z: {robotStatus.position[2].toFixed(2)}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            速度: {robotStatus.velocity.toFixed(2)} m/s
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RobotStatusCard;