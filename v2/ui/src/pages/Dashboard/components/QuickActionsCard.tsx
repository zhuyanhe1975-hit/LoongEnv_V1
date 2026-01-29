import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Button,
  Grid,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Pause,
  Settings,
  Refresh,
  Report,
  Cable,
} from '@mui/icons-material';
import { useAppSelector } from '../../../store';
import ConnectionManager from '../../../components/common/ConnectionManager';

const QuickActionsCard: React.FC = () => {
  const { isConnected } = useAppSelector((state) => state.robot);
  const [connectionManagerOpen, setConnectionManagerOpen] = useState(false);

  const handleAction = (action: string) => {
    console.log(`执行操作: ${action}`);
    // TODO: 实现具体的操作逻辑
  };

  return (
    <Card>
      <CardHeader title="快速操作" />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color={isConnected ? "success" : "primary"}
              startIcon={<Cable />}
              onClick={() => setConnectionManagerOpen(true)}
            >
              {isConnected ? '已连接' : '连接机器人'}
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="primary"
              startIcon={<PlayArrow />}
              onClick={() => handleAction('start')}
              disabled={!isConnected}
            >
              启动
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="secondary"
              startIcon={<Stop />}
              onClick={() => handleAction('stop')}
              disabled={!isConnected}
            >
              停止
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<Pause />}
              onClick={() => handleAction('pause')}
              disabled={!isConnected}
            >
              暂停
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<Refresh />}
              onClick={() => handleAction('reset')}
              disabled={!isConnected}
            >
              重置
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<Settings />}
              onClick={() => handleAction('settings')}
            >
              设置
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Button
              fullWidth
              variant="contained"
              color="error"
              startIcon={<Report />}
              onClick={() => handleAction('emergency')}
            >
              急停
            </Button>
          </Grid>
        </Grid>
      </CardContent>

      {/* Connection Manager Dialog */}
      <ConnectionManager
        open={connectionManagerOpen}
        onClose={() => setConnectionManagerOpen(false)}
      />
    </Card>
  );
};

export default QuickActionsCard;