import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  Speed,
  Memory,
  ThermostatAuto,
} from '@mui/icons-material';
import Robot3DViewer from '../../components/common/Robot3DViewer';

const MonitoringPage: React.FC = () => {
  // Mock data for demonstration
  const systemStatus = {
    cpu: 45,
    memory: 62,
    temperature: 38,
    status: 'running',
  };

  const alerts = [
    { id: 1, type: 'warning', message: '关节3温度偏高', time: '2分钟前' },
    { id: 2, type: 'info', message: '系统性能优化完成', time: '15分钟前' },
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        系统监控
      </Typography>
      
      <Grid container spacing={3}>
        {/* 3D Robot Model */}
        <Grid item xs={12} lg={6}>
          <Robot3DViewer />
        </Grid>

        {/* System Status Cards */}
        <Grid item xs={12} lg={6}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Speed sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">CPU使用率</Typography>
                  </Box>
                  <Typography variant="h4" color="primary">
                    {systemStatus.cpu}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={systemStatus.cpu} 
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Memory sx={{ mr: 1, color: 'secondary.main' }} />
                    <Typography variant="h6">内存使用</Typography>
                  </Box>
                  <Typography variant="h4" color="secondary">
                    {systemStatus.memory}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={systemStatus.memory} 
                    color="secondary"
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <ThermostatAuto sx={{ mr: 1, color: 'warning.main' }} />
                    <Typography variant="h6">系统温度</Typography>
                  </Box>
                  <Typography variant="h4" color="warning.main">
                    {systemStatus.temperature}°C
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(systemStatus.temperature / 80) * 100} 
                    color="warning"
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Alerts Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              系统警报
            </Typography>
            {alerts.map((alert) => (
              <Alert 
                key={alert.id}
                severity={alert.type as 'warning' | 'info'}
                sx={{ mb: 1 }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography>{alert.message}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {alert.time}
                  </Typography>
                </Box>
              </Alert>
            ))}
          </Paper>
        </Grid>

        {/* Real-time Chart Placeholder */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              实时性能监控
            </Typography>
            <Box 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                bgcolor: 'grey.100',
                borderRadius: 1
              }}
            >
              <Typography color="text.secondary">
                实时图表将在此处显示
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MonitoringPage;