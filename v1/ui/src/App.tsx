import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';

import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard/Dashboard';
import MonitoringPage from './pages/Monitoring/MonitoringPage';
import PlanningPage from './pages/Planning/PlanningPage';
import TuningPage from './pages/Tuning/TuningPage';
import SettingsPage from './pages/Settings/SettingsPage';
import NotificationProvider from './components/common/NotificationProvider';

const App: React.FC = () => {
  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <NotificationProvider>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/monitoring/*" element={<MonitoringPage />} />
            <Route path="/planning" element={<PlanningPage />} />
            <Route path="/tuning" element={<TuningPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </MainLayout>
      </NotificationProvider>
    </Box>
  );
};

export default App;