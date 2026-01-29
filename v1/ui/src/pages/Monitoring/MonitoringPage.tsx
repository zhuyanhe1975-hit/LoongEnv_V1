import React from 'react';
import { Box } from '@mui/material';

import IndustrialDashboard from '../../components/dashboard/IndustrialDashboard';

const MonitoringPage: React.FC = () => {
  return (
    <Box sx={{ height: '100%', p: 2 }}>
      <IndustrialDashboard />
    </Box>
  );
};

export default MonitoringPage;
