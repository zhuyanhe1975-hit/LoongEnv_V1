import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  IconButton,
  Typography,
  Box,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Schedule,
  PlayArrow,
  MoreVert,
} from '@mui/icons-material';

interface Task {
  id: string;
  name: string;
  status: 'completed' | 'error' | 'running' | 'pending';
  timestamp: string;
  duration?: string;
}

const RecentTasksList: React.FC = () => {
  // Mock data for demonstration
  const recentTasks: Task[] = [
    {
      id: '1',
      name: '轨迹规划任务 #001',
      status: 'completed',
      timestamp: '2分钟前',
      duration: '1.2s',
    },
    {
      id: '2',
      name: 'PID参数优化',
      status: 'running',
      timestamp: '5分钟前',
    },
    {
      id: '3',
      name: '碰撞检测测试',
      status: 'completed',
      timestamp: '10分钟前',
      duration: '0.8s',
    },
    {
      id: '4',
      name: '动力学参数识别',
      status: 'error',
      timestamp: '15分钟前',
    },
    {
      id: '5',
      name: '安全边界验证',
      status: 'pending',
      timestamp: '20分钟前',
    },
  ];

  const getStatusIcon = (status: Task['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'running':
        return <PlayArrow color="primary" />;
      case 'pending':
        return <Schedule color="warning" />;
      default:
        return <Schedule />;
    }
  };

  const getStatusChip = (status: Task['status']) => {
    const statusMap = {
      completed: { label: '已完成', color: 'success' as const },
      error: { label: '错误', color: 'error' as const },
      running: { label: '运行中', color: 'primary' as const },
      pending: { label: '等待中', color: 'warning' as const },
    };

    const config = statusMap[status];
    return <Chip label={config.label} color={config.color} size="small" />;
  };

  return (
    <Card>
      <CardHeader 
        title="最近任务" 
        subheader={`共 ${recentTasks.length} 个任务`}
      />
      <CardContent sx={{ pt: 0 }}>
        {recentTasks.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              py: 4,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              暂无任务记录
            </Typography>
          </Box>
        ) : (
          <List disablePadding>
            {recentTasks.map((task, index) => (
              <ListItem
                key={task.id}
                divider={index < recentTasks.length - 1}
                sx={{ px: 0 }}
              >
                <ListItemIcon>
                  {getStatusIcon(task.status)}
                </ListItemIcon>
                <ListItemText
                  primary={task.name}
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {task.timestamp}
                      </Typography>
                      {task.duration && (
                        <>
                          <Typography variant="caption" color="text.secondary">
                            •
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            耗时: {task.duration}
                          </Typography>
                        </>
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusChip(task.status)}
                    <IconButton edge="end" size="small">
                      <MoreVert />
                    </IconButton>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentTasksList;