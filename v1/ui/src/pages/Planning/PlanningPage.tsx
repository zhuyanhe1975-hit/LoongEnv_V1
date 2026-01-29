import React, { useMemo, useState } from 'react';
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
  LinearProgress,
  Snackbar,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  Timeline,
  Speed,
  Tune,
  Add,
  FileUpload,
  FileDownload,
} from '@mui/icons-material';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip, CartesianGrid, Legend } from 'recharts';
import { backendService } from '../../services/backendService';

const PlanningPage: React.FC = () => {
  const [algorithm, setAlgorithm] = useState('topp');
  const [isPlanning, setIsPlanning] = useState(false);
  const [planningProgress, setPlanningProgress] = useState(0);
  const [trajectory, setTrajectory] = useState<Array<{ time: number; position: number[] }> | null>(null);
  const [waypoints, setWaypoints] = useState([
    { position: [0, 0, 0, 0, 0, 0], time: 0 },
    { position: [0.2, -0.2, 0.3, 0.1, 0.0, 0.0], time: 2.0 },
    { position: [0.4, 0.1, 0.1, -0.2, 0.1, 0.0], time: 4.0 },
  ]);
  const [snackbar, setSnackbar] = useState({ open: false, message: '' });

  const algorithms = [
    { value: 'topp', label: 'TOPP算法' },
    { value: 'rrt', label: 'RRT路径规划' },
    { value: 'quintic', label: '五次多项式' },
    { value: 'spline', label: '样条插值' },
  ];

  const [trajectoryParams, setTrajectoryParams] = useState({
    maxVelocity: 2.0,
    maxAcceleration: 1.5,
    maxJerk: 3.0,
    smoothness: 0.8,
  });

  const chartData = useMemo(() => {
    if (!trajectory) return [];
    return trajectory.map((p) => ({
      time: Number(p.time.toFixed(3)),
      j1: p.position?.[0] ?? 0,
      j2: p.position?.[1] ?? 0,
      j3: p.position?.[2] ?? 0,
    }));
  }, [trajectory]);

  const handleStartPlanning = async () => {
    setIsPlanning(true);
    setPlanningProgress(0);
    setTrajectory(null);

    let timer: number | undefined;
    try {
      timer = window.setInterval(() => {
        setPlanningProgress((p) => Math.min(95, p + 7));
      }, 200);

      const result = await backendService.planTrajectory({
        waypoints: waypoints,
        optimizeTime: algorithm === 'topp',
        trajectoryParams: trajectoryParams,
      });

      setTrajectory(result?.trajectory || []);
      setPlanningProgress(100);
      setSnackbar({ open: true, message: '轨迹规划完成' });
    } catch (error) {
      console.error('Trajectory planning failed:', error);
      setPlanningProgress(0);
      setSnackbar({ open: true, message: '轨迹规划失败' });
    } finally {
      if (timer) window.clearInterval(timer);
      setIsPlanning(false);
    }
  };

  const handleAddWaypoint = () => {
    const newWaypoint = {
      position: [0, 0, 0, 0, 0, 0],
      time: waypoints.length > 0 ? waypoints[waypoints.length - 1].time + 2.0 : 0,
    };
    setWaypoints([...waypoints, newWaypoint]);
    setSnackbar({ open: true, message: '路径点已添加' });
  };

  const handleImportPath = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e: any) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:5006/api/trajectory/import', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          if (data.waypoints && data.waypoints.length > 0) {
            setWaypoints(data.waypoints);
            setSnackbar({ open: true, message: '路径已导入' });
          }
        } else {
          setSnackbar({ open: true, message: '导入路径失败' });
        }
      } catch (error) {
        console.error('Failed to import path:', error);
        setSnackbar({ open: true, message: '导入路径失败' });
      }
    };
    input.click();
  };

  const handleExportPath = async () => {
    try {
      const response = await fetch('http://localhost:5006/api/trajectory/export');
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'trajectory.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        setSnackbar({ open: true, message: '路径已导出' });
      } else {
        setSnackbar({ open: true, message: '导出路径失败' });
      }
    } catch (error) {
      console.error('Failed to export path:', error);
      setSnackbar({ open: true, message: '导出路径失败' });
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 800 }}>
        轨迹规划
      </Typography>

      <Grid container spacing={2}>
        {/* Planning Configuration */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, border: 1, borderColor: 'divider', bgcolor: 'background.paper' }} elevation={0}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
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
              label="最大速度 (rad/s)"
              type="number"
              value={trajectoryParams.maxVelocity}
              onChange={(e) => setTrajectoryParams(prev => ({ ...prev, maxVelocity: parseFloat(e.target.value) || 0 }))}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="最大加速度 (rad/s²)"
              type="number"
              value={trajectoryParams.maxAcceleration}
              onChange={(e) => setTrajectoryParams(prev => ({ ...prev, maxAcceleration: parseFloat(e.target.value) || 0 }))}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="最大加加速度 (rad/s³)"
              type="number"
              value={trajectoryParams.maxJerk}
              onChange={(e) => setTrajectoryParams(prev => ({ ...prev, maxJerk: parseFloat(e.target.value) || 0 }))}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="平滑度"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.1 }}
              value={trajectoryParams.smoothness}
              onChange={(e) => setTrajectoryParams(prev => ({ ...prev, smoothness: parseFloat(e.target.value) || 0 }))}
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

            {isPlanning && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
                  规划进度
                </Typography>
                <LinearProgress variant="determinate" value={planningProgress} sx={{ mt: 1 }} />
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Planning Results */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, mb: 2, border: 1, borderColor: 'divider', bgcolor: 'background.paper' }} elevation={0}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
              规划结果
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Timeline sx={{ fontSize: 28, color: 'primary.main', mb: 1 }} />
                    <Typography variant="h6">轨迹长度</Typography>
                    <Typography variant="h4" color="primary">
                      2.45m
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Speed sx={{ fontSize: 28, color: 'secondary.main', mb: 1 }} />
                    <Typography variant="h6">执行时间</Typography>
                    <Typography variant="h4" color="secondary">
                      3.2s
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Tune sx={{ fontSize: 28, color: 'warning.main', mb: 1 }} />
                    <Typography variant="h6">平滑度</Typography>
                    <Typography variant="h4" color="warning.main">
                      0.85
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Settings sx={{ fontSize: 28, color: 'success.main', mb: 1 }} />
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
          <Paper sx={{ p: 2, border: 1, borderColor: 'divider', bgcolor: 'background.paper' }} elevation={0}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
              轨迹可视化
            </Typography>
            <Box sx={{ height: 260 }}>
              {!trajectory || trajectory.length === 0 ? (
                <Box
                  sx={{
                    height: '100%',
                    bgcolor: 'rgba(255, 255, 255, 0.04)',
                    borderRadius: 2,
                    border: '1px dashed',
                    borderColor: 'divider',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography color="text.secondary">规划结果将在此处可视化</Typography>
                </Box>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="time" tick={{ fill: '#B3B3B3', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#B3B3B3', fontSize: 12 }} />
                    <ReTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="j1" stroke="#2196F3" dot={false} name="J1" />
                    <Line type="monotone" dataKey="j2" stroke="#FF9800" dot={false} name="J2" />
                    <Line type="monotone" dataKey="j3" stroke="#4CAF50" dot={false} name="J3" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Waypoints */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, border: 1, borderColor: 'divider', bgcolor: 'background.paper' }} elevation={0}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
              路径点管理
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Button 
                variant="outlined" 
                startIcon={<Add />}
                sx={{ mr: 1 }}
                onClick={handleAddWaypoint}
              >
                添加路径点
              </Button>
              <Button 
                variant="outlined" 
                startIcon={<FileUpload />}
                sx={{ mr: 1 }}
                onClick={handleImportPath}
              >
                导入路径
              </Button>
              <Button 
                variant="outlined"
                startIcon={<FileDownload />}
                onClick={handleExportPath}
              >
                导出路径
              </Button>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Typography variant="body2" color="text.secondary">
              当前路径点数量: {waypoints.length}
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
    </Box>
  );
};

export default PlanningPage;
