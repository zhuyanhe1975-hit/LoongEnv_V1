import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Chip,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  Timeline,
  Speed,
  Tune,
} from '@mui/icons-material';

const PlanningPage: React.FC = () => {
  const [algorithm, setAlgorithm] = useState('topp');
  const [isPlanning, setIsPlanning] = useState(false);

  const algorithms = [
    { value: 'topp', label: 'TOPP算法' },
    { value: 'rrt', label: 'RRT路径规划' },
    { value: 'quintic', label: '五次多项式' },
    { value: 'spline', label: '样条插值' },
  ];

  const trajectoryParams = {
    maxVelocity: 2.0,
    maxAcceleration: 1.5,
    maxJerk: 3.0,
    smoothness: 0.8,
  };

  const handleStartPlanning = () => {
    setIsPlanning(true);
    // Simulate planning process
    setTimeout(() => {
      setIsPlanning(false);
    }, 3000);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        轨迹规划
      </Typography>

      <Grid container spacing={3}>
        {/* Planning Configuration */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              规划配置
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>规划算法</InputLabel>
              <Select
                value={algorithm}
                label="规划算法"
                onChange={(e) => setAlgorithm(e.target.value)}
              >
                {algorithms.map((alg) => (
                  <MenuItem key={alg.value} value={alg.value}>
                    {alg.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="最大速度 (m/s)"
              type="number"
              defaultValue={trajectoryParams.maxVelocity}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="最大加速度 (m/s²)"
              type="number"
              defaultValue={trajectoryParams.maxAcceleration}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="最大加加速度 (m/s³)"
              type="number"
              defaultValue={trajectoryParams.maxJerk}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="平滑度"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.1 }}
              defaultValue={trajectoryParams.smoothness}
              sx={{ mb: 3 }}
            />

            <Button
              fullWidth
              variant="contained"
              startIcon={isPlanning ? <Stop /> : <PlayArrow />}
              onClick={handleStartPlanning}
              disabled={isPlanning}
              color={isPlanning ? 'secondary' : 'primary'}
            >
              {isPlanning ? '规划中...' : '开始规划'}
            </Button>
          </Paper>
        </Grid>

        {/* Planning Results */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              规划结果
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Timeline sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                    <Typography variant="h6">轨迹长度</Typography>
                    <Typography variant="h4" color="primary">
                      2.45m
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Speed sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                    <Typography variant="h6">执行时间</Typography>
                    <Typography variant="h4" color="secondary">
                      3.2s
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Tune sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                    <Typography variant="h6">平滑度</Typography>
                    <Typography variant="h4" color="warning.main">
                      0.85
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={6} md={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Settings sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                    <Typography variant="h6">状态</Typography>
                    <Chip 
                      label="已完成" 
                      color="success" 
                      variant="filled"
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>

          {/* Trajectory Visualization */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              轨迹可视化
            </Typography>
            <Box 
              sx={{ 
                height: 400, 
                bgcolor: 'grey.100', 
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Typography color="text.secondary">
                3D轨迹可视化将在此处显示
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Waypoints */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              路径点管理
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Button variant="outlined" sx={{ mr: 1 }}>
                添加路径点
              </Button>
              <Button variant="outlined" sx={{ mr: 1 }}>
                导入路径
              </Button>
              <Button variant="outlined">
                导出路径
              </Button>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Typography variant="body2" color="text.secondary">
              当前路径点: 起始点 → 中间点1 → 中间点2 → 目标点
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PlanningPage;