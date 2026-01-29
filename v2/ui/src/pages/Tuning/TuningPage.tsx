import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Slider,
  TextField,
  Button,
  Card,
  CardContent,
  FormControlLabel,
  Switch,
  Tabs,
  Tab,
  Alert,
  LinearProgress,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Save,
  Restore,
  Analytics,
  CheckCircle,
  Error,
  Warning,
  TrendingUp,
  Assessment,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import { 
  setTuningParameters, 
  setTuningStatus, 
  setTuningResults,
  updateTuningProgress 
} from '../../store/slices/tuningSlice';
import { backendService, TuningConfig } from '../../services/backendService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tuning-tabpanel-${index}`}
      aria-labelledby={`tuning-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const TuningPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { parameters, status, results, progress } = useAppSelector((state) => state.tuning);
  
  const [tabValue, setTabValue] = useState(0);
  const [tuningConfig, setTuningConfig] = useState<TuningConfig>({
    method: 'differential_evolution',
    maxIterations: 50,
    tolerance: 1e-6,
    populationSize: 15,
    parameterTypes: ['control_gains', 'trajectory_params'],
    performanceWeights: {
      trackingAccuracy: 0.4,
      settlingTime: 0.2,
      overshoot: 0.15,
      energyEfficiency: 0.1,
      vibrationSuppression: 0.1,
      safetyMargin: 0.05,
    },
  });
  
  const [controlGains, setControlGains] = useState({
    kp: [200, 200, 200, 200, 200, 200],
    ki: [20, 20, 20, 20, 20, 20],
    kd: [15, 15, 15, 15, 15, 15],
  });

  // 轮询调优状态
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (status === 'running') {
      interval = setInterval(async () => {
        try {
          const tuningStatus = await backendService.getTuningStatus();
          dispatch(updateTuningProgress(tuningStatus.progress));
          
          if (!tuningStatus.running && tuningStatus.results) {
            dispatch(setTuningStatus('completed'));
            dispatch(setTuningResults(tuningStatus.results));
          }
        } catch (error) {
          console.error('Failed to get tuning status:', error);
          dispatch(setTuningStatus('error'));
        }
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [status, dispatch]);

  // 加载当前控制增益
  useEffect(() => {
    const loadControlGains = async () => {
      try {
        const gains = await backendService.getControlGains();
        setControlGains({
          kp: gains.kp,
          ki: gains.ki,
          kd: gains.kd,
        });
      } catch (error) {
        console.error('Failed to load control gains:', error);
      }
    };
    
    loadControlGains();
  }, []);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleStartTuning = async () => {
    try {
      dispatch(setTuningStatus('running'));
      dispatch(updateTuningProgress(0));
      await backendService.startParameterTuning(tuningConfig);
    } catch (error) {
      console.error('Failed to start tuning:', error);
      dispatch(setTuningStatus('error'));
    }
  };

  const handleStopTuning = async () => {
    try {
      await backendService.stopParameterTuning();
      dispatch(setTuningStatus('idle'));
      dispatch(updateTuningProgress(0));
    } catch (error) {
      console.error('Failed to stop tuning:', error);
    }
  };

  const handleApplyGains = async () => {
    try {
      await backendService.setControlGains(controlGains);
      // 显示成功消息
    } catch (error) {
      console.error('Failed to apply gains:', error);
    }
  };

  const handleConfigChange = (key: keyof TuningConfig, value: any) => {
    setTuningConfig(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleWeightChange = (key: string, value: number) => {
    setTuningConfig(prev => ({
      ...prev,
      performanceWeights: {
        ...prev.performanceWeights,
        [key]: value,
      },
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'running':
        return <TrendingUp color="primary" />;
      default:
        return <Warning color="warning" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        参数调优
      </Typography>

      {/* 调优状态卡片 */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {getStatusIcon(status)}
            <Typography variant="h6" sx={{ ml: 1 }}>
              调优状态: {status === 'running' ? '运行中' : status === 'completed' ? '已完成' : status === 'error' ? '错误' : '空闲'}
            </Typography>
          </Box>
          
          {status === 'running' && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                进度: {progress}%
              </Typography>
              <LinearProgress variant="determinate" value={progress} />
            </Box>
          )}
          
          {results && (
            <Alert severity={results.success ? 'success' : 'error'} sx={{ mb: 2 }}>
              {results.success 
                ? `调优完成！总体性能提升: ${results.overallImprovement?.toFixed(2)}%`
                : `调优失败: ${results.error}`
              }
            </Alert>
          )}
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={handleStartTuning}
              disabled={status === 'running'}
            >
              开始调优
            </Button>
            <Button
              variant="outlined"
              startIcon={<Stop />}
              onClick={handleStopTuning}
              disabled={status !== 'running'}
            >
              停止调优
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="调优配置" />
          <Tab label="控制增益" />
          <Tab label="性能权重" />
          <Tab label="调优结果" />
        </Tabs>

        {/* 调优配置 */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    优化算法配置
                  </Typography>
                  
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>优化方法</InputLabel>
                    <Select
                      value={tuningConfig.method}
                      onChange={(e) => handleConfigChange('method', e.target.value)}
                    >
                      <MenuItem value="differential_evolution">差分进化算法</MenuItem>
                      <MenuItem value="gradient_descent">梯度下降</MenuItem>
                      <MenuItem value="basin_hopping">盆地跳跃算法</MenuItem>
                    </Select>
                  </FormControl>

                  <TextField
                    fullWidth
                    label="最大迭代次数"
                    type="number"
                    value={tuningConfig.maxIterations}
                    onChange={(e) => handleConfigChange('maxIterations', parseInt(e.target.value))}
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="收敛容差"
                    type="number"
                    value={tuningConfig.tolerance}
                    onChange={(e) => handleConfigChange('tolerance', parseFloat(e.target.value))}
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="种群大小"
                    type="number"
                    value={tuningConfig.populationSize}
                    onChange={(e) => handleConfigChange('populationSize', parseInt(e.target.value))}
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    参数类型选择
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={tuningConfig.parameterTypes.includes('control_gains')}
                        onChange={(e) => {
                          const types = e.target.checked 
                            ? [...tuningConfig.parameterTypes, 'control_gains']
                            : tuningConfig.parameterTypes.filter(t => t !== 'control_gains');
                          handleConfigChange('parameterTypes', types);
                        }}
                      />
                    }
                    label="控制增益参数"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={tuningConfig.parameterTypes.includes('trajectory_params')}
                        onChange={(e) => {
                          const types = e.target.checked 
                            ? [...tuningConfig.parameterTypes, 'trajectory_params']
                            : tuningConfig.parameterTypes.filter(t => t !== 'trajectory_params');
                          handleConfigChange('parameterTypes', types);
                        }}
                      />
                    }
                    label="轨迹规划参数"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={tuningConfig.parameterTypes.includes('vibration_params')}
                        onChange={(e) => {
                          const types = e.target.checked 
                            ? [...tuningConfig.parameterTypes, 'vibration_params']
                            : tuningConfig.parameterTypes.filter(t => t !== 'vibration_params');
                          handleConfigChange('parameterTypes', types);
                        }}
                      />
                    }
                    label="抑振参数"
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* 控制增益 */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    PID控制增益
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {[0, 1, 2, 3, 4, 5].map((joint) => (
                      <Grid item xs={12} md={4} key={joint}>
                        <Typography variant="subtitle2" gutterBottom>
                          关节 {joint + 1}
                        </Typography>
                        
                        <TextField
                          fullWidth
                          label="Kp"
                          type="number"
                          value={controlGains.kp[joint]}
                          onChange={(e) => {
                            const newKp = [...controlGains.kp];
                            newKp[joint] = parseFloat(e.target.value) || 0;
                            setControlGains(prev => ({ ...prev, kp: newKp }));
                          }}
                          size="small"
                          sx={{ mb: 1 }}
                        />
                        
                        <TextField
                          fullWidth
                          label="Ki"
                          type="number"
                          value={controlGains.ki[joint]}
                          onChange={(e) => {
                            const newKi = [...controlGains.ki];
                            newKi[joint] = parseFloat(e.target.value) || 0;
                            setControlGains(prev => ({ ...prev, ki: newKi }));
                          }}
                          size="small"
                          sx={{ mb: 1 }}
                        />
                        
                        <TextField
                          fullWidth
                          label="Kd"
                          type="number"
                          value={controlGains.kd[joint]}
                          onChange={(e) => {
                            const newKd = [...controlGains.kd];
                            newKd[joint] = parseFloat(e.target.value) || 0;
                            setControlGains(prev => ({ ...prev, kd: newKd }));
                          }}
                          size="small"
                        />
                      </Grid>
                    ))}
                  </Grid>
                  
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      onClick={handleApplyGains}
                      startIcon={<Save />}
                    >
                      应用增益
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* 性能权重 */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    性能指标权重
                  </Typography>
                  
                  {Object.entries(tuningConfig.performanceWeights).map(([key, value]) => (
                    <Box key={key} sx={{ mb: 3 }}>
                      <Typography gutterBottom>
                        {key === 'trackingAccuracy' ? '跟踪精度' :
                         key === 'settlingTime' ? '稳定时间' :
                         key === 'overshoot' ? '超调量' :
                         key === 'energyEfficiency' ? '能效' :
                         key === 'vibrationSuppression' ? '抑振' :
                         '安全裕度'}: {value.toFixed(2)}
                      </Typography>
                      <Slider
                        value={value}
                        onChange={(_, newValue) => handleWeightChange(key, newValue as number)}
                        min={0}
                        max={1}
                        step={0.05}
                        valueLabelDisplay="auto"
                      />
                    </Box>
                  ))}
                  
                  <Alert severity="info">
                    权重总和: {Object.values(tuningConfig.performanceWeights).reduce((a, b) => a + b, 0).toFixed(2)}
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* 调优结果 */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            {results && results.success && (
              <>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        调优结果概览
                      </Typography>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="primary">
                              {results.overallImprovement?.toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              总体性能提升
                            </Typography>
                          </Paper>
                        </Grid>
                        
                        {Object.entries(results.results || {}).map(([paramType, result]: [string, any]) => (
                          <Grid item xs={12} md={4} key={paramType}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Chip 
                                icon={result.success ? <CheckCircle /> : <Error />}
                                label={result.success ? '成功' : '失败'}
                                color={result.success ? 'success' : 'error'}
                                sx={{ mb: 1 }}
                              />
                              <Typography variant="body2">
                                {paramType === 'control_gains' ? '控制增益' :
                                 paramType === 'trajectory_params' ? '轨迹参数' :
                                 '抑振参数'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                性能: {result.bestPerformance?.toFixed(4)}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
                
                {results.recommendations && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          优化建议
                        </Typography>
                        <List>
                          {results.recommendations.map((recommendation: string, index: number) => (
                            <ListItem key={index}>
                              <ListItemIcon>
                                <Assessment />
                              </ListItemIcon>
                              <ListItemText primary={recommendation} />
                            </ListItem>
                          ))}
                        </List>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </>
            )}
            
            {!results && (
              <Grid item xs={12}>
                <Alert severity="info">
                  暂无调优结果。请先运行参数调优。
                </Alert>
              </Grid>
            )}
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default TuningPage;