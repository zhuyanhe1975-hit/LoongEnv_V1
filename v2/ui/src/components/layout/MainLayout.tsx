import React from 'react';
import { Box } from '@mui/material';
import { useAppSelector } from '../../store';

import AppBar from './AppBar';
import Sidebar from './Sidebar';
import StatusBar from './StatusBar';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const sidebarOpen = useAppSelector((state) => state.ui.sidebarOpen);

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* App Bar */}
      <AppBar />
      
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          marginLeft: sidebarOpen ? '280px' : '80px',
          marginTop: '64px',
          transition: 'margin 225ms cubic-bezier(0.4, 0, 0.6, 1) 0ms',
          overflow: 'hidden',
        }}
      >
        {/* Page Content */}
        <Box
          sx={{
            flexGrow: 1,
            padding: 3,
            overflow: 'auto',
            backgroundColor: 'background.default',
          }}
        >
          {children}
        </Box>
        
        {/* Status Bar */}
        <StatusBar />
      </Box>
    </Box>
  );
};

export default MainLayout;