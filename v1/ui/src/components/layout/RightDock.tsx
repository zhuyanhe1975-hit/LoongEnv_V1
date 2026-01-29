import React from 'react';
import { Box, Card, CardContent, CardHeader, LinearProgress, Typography } from '@mui/material';
import Robot3DViewer from '../common/Robot3DViewer';
import { useAppSelector } from '../../store';

const RightDock: React.FC = () => {
  const { backendStatus, backendProgress, backendResults } = useAppSelector((state) => state.tuning);
  const { isConnected } = useAppSelector((state) => state.robot);

  const showProgress = backendStatus === 'running';
  const improvement = backendResults?.overallImprovement;

  return (
    <Box
      sx={{
        width: { xs: '100%', md: '50vw' },
        minWidth: { md: 520 },
        maxWidth: { md: 960 },
        flexShrink: 0,
        borderLeft: 1,
        borderColor: 'divider',
        bgcolor: 'background.default',
        p: 1.5,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        height: '100%',
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      <Box sx={{ flexGrow: 1, minHeight: 0 }}>
        <Robot3DViewer height="100%" />
      </Box>

      <Card sx={{ flexShrink: 0 }}>
        <CardHeader title="运行状态" subheader="调优结果常驻显示" />
        <CardContent>
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
            连接: {isConnected ? '已连接' : '未连接'}
          </Typography>

          <Box sx={{ mt: 1.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 800 }}>
              参数优化: {backendStatus === 'running' ? '运行中' : backendStatus === 'completed' ? '已完成' : backendStatus === 'error' ? '错误' : '空闲'}
            </Typography>
            {showProgress && (
              <>
                <LinearProgress variant="determinate" value={backendProgress} sx={{ mt: 1 }} />
                <Typography variant="caption" color="text.secondary">
                  {backendProgress}%
                </Typography>
              </>
            )}
            {typeof improvement === 'number' && backendStatus !== 'running' && (
              <Typography variant="body2" sx={{ mt: 1, fontWeight: 800 }}>
                性能提升: {Number(improvement).toFixed(2)}%
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default RightDock;
