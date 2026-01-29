import React from 'react';
import { useLocation } from 'react-router-dom';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';

const StatusBar: React.FC = () => {
  const location = useLocation();
  const { currentState, isConnected, lastUpdate } = useAppSelector((state) => state.robot);

  const statusColor = React.useMemo(() => {
    if (!isConnected) return 'error.main';
    if (!currentState) return 'warning.main';

    switch (currentState.safetyStatus) {
      case 'safe':
        return 'success.main';
      case 'warning':
        return 'warning.main';
      case 'error':
        return 'error.main';
      default:
        return 'text.secondary';
    }
  }, [currentState, isConnected]);

  const statusText = React.useMemo(() => {
    if (!isConnected) return '未连接';
    if (!currentState) return '等待数据';

    switch (currentState.safetyStatus) {
      case 'safe':
        return '安全';
      case 'warning':
        return '警告';
      case 'error':
        return '错误';
      case 'offline':
        return '离线';
      default:
        return '未知';
    }
  }, [currentState, isConnected]);

  const pageKey = React.useMemo(() => {
    const pathname = location.pathname || '/';
    if (pathname.startsWith('/monitoring')) return 'monitoring';
    if (pathname.startsWith('/planning')) return 'planning';
    if (pathname.startsWith('/tuning')) return 'tuning';
    if (pathname.startsWith('/settings')) return 'settings';
    return 'dashboard';
  }, [location.pathname]);

  const formattedUpdate = React.useMemo(() => {
    if (!lastUpdate) return '从未';
    try {
      return formatDistanceToNow(new Date(lastUpdate), { addSuffix: true, locale: zhCN });
    } catch {
      return '未知';
    }
  }, [lastUpdate]);

  return (
    <Box
      sx={{
        height: 32,
        bgcolor: 'background.paper',
        borderTop: 1,
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 1.5,
        flexShrink: 0,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1,
          py: 0.25,
          borderRadius: 999,
          border: 1,
          borderColor: 'divider',
        }}
      >
        <Box sx={{ width: 8, height: 8, borderRadius: 999, bgcolor: statusColor }} />
        <Typography sx={{ fontSize: 12, fontWeight: 700, lineHeight: 1.2 }}>
          {statusText}
        </Typography>
      </Box>

      <Box sx={{ flexGrow: 1 }} />

      <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'text.secondary' }}>
        当前页面: {pageKey} · 更新: {formattedUpdate}
      </Typography>
    </Box>
  );
};

export default StatusBar;
