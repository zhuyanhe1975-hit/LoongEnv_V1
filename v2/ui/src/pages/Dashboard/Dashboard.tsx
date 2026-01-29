import React from 'react';
import { Grid, Typography, Box } from '@mui/material';

import SystemOverviewCard from './components/SystemOverviewCard';
import QuickActionsCard from './components/QuickActionsCard';
import PerformanceChart from './components/PerformanceChart';
import RecentTasksList from './components/RecentTasksList';
import RobotStatusCard from './components/RobotStatusCard';
import SafetyStatusCard from './components/SafetyStatusCard';
import Robot3DViewer from '../../components/common/Robot3DViewer';
import DiagnosticViewer from '../../components/common/DiagnosticViewer';

const Dashboard: React.FC = () => {
  return (
    <Box>
      {/* Page Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          仪表板
        </Typography>
        <Typography variant="body1" color="text.secondary">
          机器人运动控制系统概览
        </Typography>
      </Box>

      {/* Dashboard Grid */}
      <Grid container spacing={3}>
        {/* Top Row - System Overview and 3D Model */}
        <Grid item xs={12} lg={8}>
          <SystemOverviewCard />
        </Grid>
        <Grid item xs={12} lg={4}>
          <Robot3DViewer />
        </Grid>

        {/* Second Row - Robot Status and Safety */}
        <Grid item xs={12} md={6}>
          <RobotStatusCard />
        </Grid>
        <Grid item xs={12} md={6}>
          <SafetyStatusCard />
        </Grid>

        {/* Third Row - Performance and Actions */}
        <Grid item xs={12} md={8}>
          <PerformanceChart />
        </Grid>
        <Grid item xs={12} md={4}>
          <QuickActionsCard />
        </Grid>

        {/* Fourth Row - Recent Tasks */}
        <Grid item xs={12}>
          <RecentTasksList />
        </Grid>

        {/* Diagnostic Viewer - Temporary for debugging */}
        <Grid item xs={12}>
          <DiagnosticViewer />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;