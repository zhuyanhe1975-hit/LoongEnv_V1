import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Security,
  CheckCircle,
  Warning,
  Error,
} from '@mui/icons-material';

interface SafetyCheck {
  id: string;
  name: string;
  status: 'ok' | 'warning' | 'error';
  message?: string;
}

const SafetyStatusCard: React.FC = () => {
  // Mock data for demonstration
  const safetyChecks: SafetyCheck[] = [
    {
      id: '1',
      name: '紧急停止',
      status: 'ok',
    },
    {
      id: '2',
      name: '工作空间边界',
      status: 'ok',
    },
    {
      id: '3',
      name: '碰撞检测',
      status: 'warning',
      message: '检测到轻微干扰',
    },
    {
      id: '4',
      name: '力矩限制',
      status: 'ok',
    },
  ];

  const getStatusIcon = (status: SafetyCheck['status']) => {
    switch (status) {
      case 'ok':
        return <CheckCircle color="success" fontSize="small" />;
      case 'warning':
        return <Warning color="warning" fontSize="small" />;
      case 'error':
        return <Error color="error" fontSize="small" />;
      default:
        return <CheckCircle fontSize="small" />;
    }
  };

  const overallStatus = safetyChecks.some(check => check.status === 'error') 
    ? 'error' 
    : safetyChecks.some(check => check.status === 'warning') 
    ? 'warning' 
    : 'ok';

  const getOverallStatusMessage = () => {
    switch (overallStatus) {
      case 'ok':
        return '所有安全检查正常';
      case 'warning':
        return '检测到安全警告';
      case 'error':
        return '检测到安全错误';
      default:
        return '安全状态未知';
    }
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Security sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6">安全状态</Typography>
        </Box>

        <Alert 
          severity={overallStatus === 'ok' ? 'success' : overallStatus}
          sx={{ mb: 2 }}
        >
          {getOverallStatusMessage()}
        </Alert>

        <List dense disablePadding>
          {safetyChecks.map((check) => (
            <ListItem key={check.id} disablePadding>
              <ListItemIcon sx={{ minWidth: 32 }}>
                {getStatusIcon(check.status)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="body2">
                    {check.name}
                  </Typography>
                }
                secondary={
                  check.message && (
                    <Typography variant="caption" color="text.secondary">
                      {check.message}
                    </Typography>
                  )
                }
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default SafetyStatusCard;