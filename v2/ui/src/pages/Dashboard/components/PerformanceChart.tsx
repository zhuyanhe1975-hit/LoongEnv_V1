import React from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Box,
  Typography,
  useTheme,
} from '@mui/material';

const PerformanceChart: React.FC = () => {
  const theme = useTheme();

  // Mock data for demonstration
  const performanceData = {
    cpu: 45,
    memory: 62,
    network: 28,
    disk: 15,
  };

  return (
    <Card>
      <CardHeader 
        title="系统性能" 
        subheader="实时性能监控"
      />
      <CardContent>
        <Box sx={{ height: 300 }}>
          {/* Placeholder for actual chart */}
          <Box
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              bgcolor: theme.palette.grey[100],
              borderRadius: 1,
              border: `1px dashed ${theme.palette.grey[300]}`,
            }}
          >
            <Typography variant="h6" color="text.secondary" gutterBottom>
              性能图表
            </Typography>
            <Typography variant="body2" color="text.secondary" align="center">
              CPU: {performanceData.cpu}% | 内存: {performanceData.memory}%
              <br />
              网络: {performanceData.network}% | 磁盘: {performanceData.disk}%
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
              实时图表将在此处显示
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart;