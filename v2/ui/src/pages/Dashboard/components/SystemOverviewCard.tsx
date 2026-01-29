import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Grid,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Avatar,
} from '@mui/material';
import {
  SmartToy as RobotIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Thermostat as TempIcon,
} from '@mui/icons-material';

import { useAppSelector } from '../../../store';
import { getRobotSpecs } from '../../../utils/urdfParser';

const SystemOverviewCard: React.FC = () => {
  const { currentState, isConnected } = useAppSelector((state) => state.robot);
  const robotSpecs = getRobotSpecs();

  const systemMetrics = [
    {
      label: 'CPU使用率',
      value: 45,
      unit: '%',
      icon: <SpeedIcon />,
      color: 'primary' as const,
    },
    {
      label: '内存使用',
      value: 68,
      unit: '%',
      icon: <MemoryIcon />,
      color: 'secondary' as const,
    },
    {
      label: '系统温度',
      value: 42,
      unit: '°C',
      icon: <TempIcon />,
      color: 'success' as const,
    },
  ];

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
    if (!isConnected) return '系统离线';
    if (!currentState) return '等待数据';
    
    switch (currentState.safetyStatus) {
      case 'safe': return '系统正常';
      case 'warning': return '系统警告';
      case 'error': return '系统错误';
      case 'offline': return '系统离线';
      default: return '状态未知';
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: 'primary.main' }}>
            <RobotIcon />
          </Avatar>
        }
        title="系统概览"
        subheader="机器人控制系统状态监控"
        action={
          <Chip
            label={getStatusText()}
            color={getStatusColor()}
            variant="outlined"
          />
        }
      />
      <CardContent>
        <Grid container spacing={3}>
          {/* System Metrics */}
          {systemMetrics.map((metric, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Box sx={{ color: `${metric.color}.main`, mr: 1 }}>
                    {metric.icon}
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {metric.label}
                  </Typography>
                </Box>
                <Typography variant="h6" component="div" gutterBottom>
                  {metric.value}{metric.unit}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={metric.value}
                  color={metric.color}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            </Grid>
          ))}

          {/* Robot Information */}
          <Grid item xs={12}>
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                机器人信息
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    机器人型号
                  </Typography>
                  <Typography variant="body1">
                    {robotSpecs.name}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    自由度
                  </Typography>
                  <Typography variant="body1">
                    {robotSpecs.dof}轴
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    工作半径
                  </Typography>
                  <Typography variant="body1">
                    {robotSpecs.reach}mm
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    负载能力
                  </Typography>
                  <Typography variant="body1">
                    {robotSpecs.payload}kg
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    重复精度
                  </Typography>
                  <Typography variant="body1">
                    ±{robotSpecs.repeatability}mm
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    机器人重量
                  </Typography>
                  <Typography variant="body1">
                    {robotSpecs.totalMass}kg
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SystemOverviewCard;