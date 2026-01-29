import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

import { useAppSelector } from '../../store';

// 工业风格状态卡片
const StatusCard: React.FC<{
  title: string;
  value: string | number;
  unit?: string;
  status: 'safe' | 'warning' | 'error' | 'offline';
  icon: React.ReactNode;
  trend?: number;
}> = ({ title, value, unit, status, icon, trend }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe': return <CheckCircleIcon fontSize="small" />;
      case 'warning': return <WarningIcon fontSize="small" />;
      case 'error': return <ErrorIcon fontSize="small" />;
      default: return null;
    }
  };

  return (
    <Card 
      sx={{ 
        height: '100%',
        cursor: 'pointer', // UI/UX Pro Max 要求
        transition: 'all 0.2s ease-out', // UI/UX Pro Max 推荐
        '&:hover': {
          transform: 'translateY(-2px)',
        }
      }}
    >
      <CardContent sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {icon}
            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
              {title}
            </Typography>
          </Box>
          <Chip
            icon={getStatusIcon(status)}
            label={status.toUpperCase()}
            color={getStatusColor(status) as any}
            size="small"
            sx={{ cursor: 'pointer' }} // UI/UX Pro Max 要求
          />
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5, mb: 1 }}>
          <Typography 
            variant="h4" 
            sx={{ 
              fontFamily: '"Fira Code", monospace',
              fontWeight: 700,
              color: 'text.primary'
            }}
          >
            {value}
          </Typography>
          {unit && (
            <Typography variant="body2" color="text.secondary">
              {unit}
            </Typography>
          )}
        </Box>

        {trend !== undefined && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography 
              variant="caption" 
              color={trend > 0 ? 'success.main' : trend < 0 ? 'error.main' : 'text.secondary'}
              sx={{ fontFamily: '"Fira Code", monospace' }}
            >
              {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              vs last hour
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// 实时数据图表卡片
const RealTimeChart: React.FC<{
  title: string;
  data: number[];
  color: string;
  unit: string;
}> = ({ title, data, color, unit }) => {
  const currentValue = data[data.length - 1] || 0;
  const maxValue = Math.max(...data);
  const progress = maxValue > 0 ? (currentValue / maxValue) * 100 : 0;

  return (
    <Card sx={{ height: '100%', cursor: 'pointer' }}>
      <CardContent sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {title}
          </Typography>
          <Tooltip title="Configure">
            <IconButton size="small" sx={{ cursor: 'pointer' }}>
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography 
            variant="h3" 
            sx={{ 
              fontFamily: '"Fira Code", monospace',
              fontWeight: 700,
              color: color
            }}
          >
            {currentValue.toFixed(2)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {unit}
          </Typography>
        </Box>

        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ 
            height: 8, 
            borderRadius: 4,
            backgroundColor: 'rgba(0,0,0,0.1)',
            '& .MuiLinearProgress-bar': {
              backgroundColor: color,
              borderRadius: 4,
            }
          }} 
        />

        {/* 简化的迷你图表 */}
        <Box sx={{ mt: 2, height: 60, position: 'relative' }}>
          <svg width="100%" height="100%" style={{ overflow: 'visible' }}>
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="2"
              points={data.map((value, index) => 
                `${(index / (data.length - 1)) * 100},${60 - (value / maxValue) * 50}`
              ).join(' ')}
            />
          </svg>
        </Box>
      </CardContent>
    </Card>
  );
};

const IndustrialDashboard: React.FC = () => {
  const { isConnected } = useAppSelector((state) => state.robot);
  
  // 模拟实时数据
  const [realtimeData, setRealtimeData] = React.useState({
    jointPositions: Array.from({ length: 20 }, () => Math.random() * 100),
    velocities: Array.from({ length: 20 }, () => Math.random() * 50),
    torques: Array.from({ length: 20 }, () => Math.random() * 80),
    temperatures: Array.from({ length: 20 }, () => 25 + Math.random() * 15),
  });

  // 模拟数据更新
  React.useEffect(() => {
    const interval = setInterval(() => {
      setRealtimeData(prev => ({
        jointPositions: [...prev.jointPositions.slice(1), Math.random() * 100],
        velocities: [...prev.velocities.slice(1), Math.random() * 50],
        torques: [...prev.torques.slice(1), Math.random() * 80],
        temperatures: [...prev.temperatures.slice(1), 25 + Math.random() * 15],
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{ p: 2 }}>
      {/* 页面标题 */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          机器人运动控制系统
        </Typography>
        <Typography variant="body1" color="text.secondary">
          实时监控 • 精密控制 • 智能分析
        </Typography>
      </Box>

      {/* 状态概览 - 数据密集型布局 */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="系统状态"
            value={isConnected ? "在线" : "离线"}
            status={isConnected ? "safe" : "offline"}
            icon={<CheckCircleIcon color={isConnected ? "success" : "disabled"} />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="关节温度"
            value={realtimeData.temperatures[realtimeData.temperatures.length - 1]?.toFixed(1) || "0.0"}
            unit="°C"
            status="safe"
            icon={<SpeedIcon color="primary" />}
            trend={2.3}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="CPU使用率"
            value="23.5"
            unit="%"
            status="safe"
            icon={<MemoryIcon color="primary" />}
            trend={-1.2}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="控制频率"
            value="1000"
            unit="Hz"
            status="safe"
            icon={<TimelineIcon color="primary" />}
            trend={0.0}
          />
        </Grid>
      </Grid>

      {/* 实时数据图表 */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6} lg={3}>
          <RealTimeChart
            title="关节位置"
            data={realtimeData.jointPositions}
            color="#10B981"
            unit="degrees"
          />
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <RealTimeChart
            title="关节速度"
            data={realtimeData.velocities}
            color="#2563EB"
            unit="deg/s"
          />
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <RealTimeChart
            title="关节力矩"
            data={realtimeData.torques}
            color="#F59E0B"
            unit="Nm"
          />
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <RealTimeChart
            title="系统温度"
            data={realtimeData.temperatures}
            color="#EF4444"
            unit="°C"
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default IndustrialDashboard;