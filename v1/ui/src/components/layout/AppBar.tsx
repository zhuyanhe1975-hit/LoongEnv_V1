import React from 'react';
import {
  AppBar as MuiAppBar,
  Toolbar,
  Typography,
  Box,
  Button,
} from '@mui/material';
import {
  Cable as ConnectionIcon,
} from '@mui/icons-material';

import { useAppSelector } from '../../store';
import ConnectionManager from '../common/ConnectionManager';

const AppBar: React.FC = () => {
  const { isConnected } = useAppSelector((state) => state.robot);
  const [connectionManagerOpen, setConnectionManagerOpen] = React.useState(false);

  return (
    <MuiAppBar
      position="fixed"
      sx={{
        zIndex: 1201,
        backgroundColor: 'background.paper',
        color: 'text.primary',
        borderBottom: 1,
        borderColor: 'divider',
      }}
    >
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          机器人运动控制系统
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
            {isConnected ? '已连接' : '未连接'}
          </Typography>

          <Button
            variant="outlined"
            color="inherit"
            startIcon={<ConnectionIcon />}
            onClick={() => setConnectionManagerOpen(true)}
            sx={{
              borderColor: 'divider',
              color: 'text.primary',
              '&:hover': { borderColor: 'divider', bgcolor: 'rgba(255, 255, 255, 0.04)' },
            }}
          >
            连接管理
          </Button>
        </Box>
      </Toolbar>

      {/* Connection Manager Dialog */}
      <ConnectionManager
        open={connectionManagerOpen}
        onClose={() => setConnectionManagerOpen(false)}
      />
    </MuiAppBar>
  );
};

export default AppBar;
