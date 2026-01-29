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
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Save,
  CheckCircle,
  Error as ErrorIcon,
  Warning as WarningIcon,
  TrendingUp,
  Assessment,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import { 
  setBackendTuningState,
  setBackendTuningResults,
  resetBackendTuning
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
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
}

const TuningPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { backendStatus, backendResults, backendProgress, error } = useAppSelector((state) => state.tuning);
  
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

  const [originalGains, setOriginalGains] = useState<any>(null);
  const [comparisonResults, setComparisonResults] = useState<any>(null);
  const [isComparing, setIsComparing] = useState(false);

  // 轮询调优状态
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined;
    
    if (backendStatus === 'running') {
      interval = setInterval(async () => {
        try {
          const tuningStatus = await backendService.getTuningStatus();
          dispatch(setBackendTuningState({ status: tuningStatus.running ? 'running' : 'idle', progress: tuningStatus.progress }));
          
          if (!tuningStatus.running && tuningStatus.results) {
            dispatch(setBackendTuningResults(tuningStatus.results));
          }
        } catch (error) {
          console.error('Failed to get tuning status:', error);
          dispatch(setBackendTuningState({ status: 'error', error: error instanceof Error ? error.message : String(error) }));
        }
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [backendStatus, dispatch]);

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
        // 保存原始增益用于对比
        setOriginalGains({
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
      // 保存调优前的参数
      const currentGains = await backendService.getControlGains();
      setOriginalGains({
        kp: currentGains.kp,
        ki: currentGains.ki,
        kd: currentGains.kd,
      });
      
      dispatch(resetBackendTuning());
      dispatch(setBackendTuningState({ status: 'running', progress: 0 }));
      await backendService.startParameterTuning(tuningConfig);
    } catch (error) {
      console.error('Failed to start tuning:', error);
      dispatch(setBackendTuningState({ status: 'error', error: error instanceof Error ? error.message : String(error) }));
    }
  };

  const handleStopTuning = async () => {
    try {
      await backendService.stopParameterTuning();
      dispatch(setBackendTuningState({ status: 'stopped', progress: 0 }));
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

  const handleComparePerformance = async () => {
    if (!originalGains || !backendResults?.results?.control_gains?.optimalParameters) {
      alert('请先完成参数调优');
      return;
    }

    setIsComparing(true);
    try {
      const optimizedParams = backendResults.results.control_gains.optimalParameters;
      
      const comparison = await backendService.comparePerformance(
        originalGains,
        optimizedParams
      );
      
      setComparisonResults(comparison);
    } catch (error) {
      console.error('Failed to compare performance:', error);
      alert('性能对比失败: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsComparing(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'running':
        return <TrendingUp color="primary" />;
      default:
        return <WarningIcon color="warning" />;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        参数调优
      </Typography>

      {/* 调优状态卡片 */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {getStatusIcon(backendStatus)}
            <Typography variant="h6" sx={{ ml: 1 }}>
              调优状态: {backendStatus === 'running' ? '运行中' : backendStatus === 'completed' ? '已完成' : backendStatus === 'error' ? '错误' : backendStatus === 'stopped' ? '已停止' : '空闲'}
            </Typography>
          </Box>
          
          {backendStatus === 'running' && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                进度: {backendProgress}%
              </Typography>
              <LinearProgress variant="determinate" value={backendProgress} />
            </Box>
          )}
          
          {backendResults && (
            <Alert severity={backendResults.success ? 'success' : 'error'} sx={{ mb: 2 }}>
              {backendResults.success
                ? `调优完成！总体性能提升: ${Number(backendResults.overallImprovement ?? 0).toFixed(2)}%`
                : `调优失败: ${backendResults.error}`
              }
            </Alert>
          )}

          {error && backendStatus === 'error' && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={handleStartTuning}
              disabled={backendStatus === 'running'}
            >
              开始调优
            </Button>
            <Button
              variant="outlined"
              startIcon={<Stop />}
              onClick={handleStopTuning}
              disabled={backendStatus !== 'running'}
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
          <Tab label="性能对比" />
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
                        onChange={(_event: Event, newValue: number | number[]) =>
                          handleWeightChange(key, Array.isArray(newValue) ? newValue[0] : newValue)
                        }
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
            {backendResults && backendResults.success && (
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
                              {Number(backendResults.overallImprovement ?? 0).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              总体性能提升
                            </Typography>
                          </Paper>
                        </Grid>
                        
                        {Object.entries(backendResults.results || {}).map(([paramType, result]: [string, any]) => (
                          <Grid item xs={12} md={4} key={paramType}>
                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                              <Chip 
                                icon={result.success ? <CheckCircle /> : <ErrorIcon />}
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
                
                {backendResults.recommendations && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          优化建议
                        </Typography>
                        <List>
                          {backendResults.recommendations.map((recommendation: string, index: number) => (
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
            
            {!backendResults && (
              <Grid item xs={12}>
                <Alert severity="info">
                  暂无调优结果。请先运行参数调优。
                </Alert>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        {/* 性能对比 */}
        <TabPanel value={tabValue} index={4}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      优化前后性能对比
                    </Typography>
                    <Button
                      variant="contained"
                      onClick={handleComparePerformance}
                      disabled={isComparing || !backendResults?.success || !originalGains}
                      startIcon={<Assessment />}
                    >
                      {isComparing ? '对比中...' : '执行性能对比'}
                    </Button>
                  </Box>

                  {!backendResults?.success && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      请先完成参数调优，然后点击"执行性能对比"按钮查看优化效果。
                    </Alert>
                  )}

                  {comparisonResults && (
                    <>
                      <Alert severity="success" sx={{ mb: 3 }}>
                        性能对比完成！机器人在相同轨迹下的表现对比如下：
                      </Alert>

                      <Grid container spacing={2}>
                        {/* 平均跟踪误差 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              平均跟踪误差 (rad)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.avgTrackingError.toFixed(6)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.avgTrackingError.toFixed(6)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.avgTrackingError.toFixed(1)}%`}
                              color={comparisonResults.improvements.avgTrackingError > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>

                        {/* 最大跟踪误差 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              最大跟踪误差 (rad)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.maxTrackingError.toFixed(6)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.maxTrackingError.toFixed(6)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.maxTrackingError.toFixed(1)}%`}
                              color={comparisonResults.improvements.maxTrackingError > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>

                        {/* 稳定时间 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              稳定时间 (s)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.settlingTime.toFixed(4)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.settlingTime.toFixed(4)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.settlingTime.toFixed(1)}%`}
                              color={comparisonResults.improvements.settlingTime > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>

                        {/* 超调量 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              超调量 (rad)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.overshoot.toFixed(6)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.overshoot.toFixed(6)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.overshoot.toFixed(1)}%`}
                              color={comparisonResults.improvements.overshoot > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>

                        {/* 能耗 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              能耗 (J)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.energyConsumption.toFixed(2)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.energyConsumption.toFixed(2)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.energyConsumption.toFixed(1)}%`}
                              color={comparisonResults.improvements.energyConsumption > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>

                        {/* RMS误差 */}
                        <Grid item xs={12} md={6}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                              RMS误差 (rad)
                            </Typography>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化前</Typography>
                                <Typography variant="h6">{comparisonResults.original.rmsError.toFixed(6)}</Typography>
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">优化后</Typography>
                                <Typography variant="h6" color="primary">{comparisonResults.optimized.rmsError.toFixed(6)}</Typography>
                              </Box>
                            </Box>
                            <Chip 
                              label={`改善 ${comparisonResults.improvements.rmsError.toFixed(1)}%`}
                              color={comparisonResults.improvements.rmsError > 0 ? 'success' : 'error'}
                              size="small"
                            />
                          </Paper>
                        </Grid>
                      </Grid>

                      <Box sx={{ mt: 3 }}>
                        <Alert severity="info">
                          <Typography variant="body2">
                            <strong>测试轨迹信息：</strong>
                            总时长 {comparisonResults.trajectory.totalTime.toFixed(2)}s，
                            共 {comparisonResults.trajectory.totalPoints} 个轨迹点
                          </Typography>
                        </Alert>
                      </Box>
                    </>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default TuningPage;
