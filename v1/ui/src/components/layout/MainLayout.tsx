import React from 'react';
import { Box } from '@mui/material';

import AppBar from './AppBar';
import Sidebar from './Sidebar';
import StatusBar from './StatusBar';
import RightDock from './RightDock';
import TuningStatusPoller from '../common/TuningStatusPoller';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <Box sx={{ height: '100vh', overflow: 'hidden', bgcolor: 'background.default' }}>
      <AppBar />
      <TuningStatusPoller />

      <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)', mt: '64px' }}>
        <Sidebar />

        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <Box sx={{ flexGrow: 1, display: 'flex', minHeight: 0 }}>
            <Box component="main" sx={{ flexGrow: 1, overflow: 'auto', p: 1, minWidth: 0 }}>
              {children}
            </Box>

            <RightDock />
          </Box>

          <StatusBar />
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;
