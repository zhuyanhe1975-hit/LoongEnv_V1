import React, { useEffect } from 'react';
import { Row, Col, Card, Button, Select, Progress, Statistic, Typography, Space } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { useAppStore } from '@/stores/appStore';
import SimulationViewer from '@/components/SimulationViewer';

const { Title, Text } = Typography;

const SimulationPage: React.FC = () => {
  const { 
    trajectoryPoints,
    simulation,
    startSimulation,
    stopSimulation,
    updateSimulationTime,
    setSimulationSpeed,
  } = useAppStore();

  // 仿真循环
  useEffect(() => {
    let animationId: number;
    
    if (simulation.isRunning && trajectoryPoints.length > 1) {
      const animate = () => {
        const deltaTime = 0.016 * simulation.speed; // 60fps
        const newTime = Math.min(simulation.currentTime + deltaTime, simulation.totalTime);
        
        updateSimulationTime(newTime);
        
        if (newTime < simulation.totalTime) {
          animationId = requestAnimationFrame(animate);
        } else {
          stopSimulation();
        }
      };
      
      animationId = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [simulation.isRunning, simulation.speed, simulation.currentTime, simulation.totalTime, trajectoryPoints.length]);

  const handleStart = () => {
    if (trajectoryPoints.length < 2) {
      alert('请至少添加2个轨迹点');
      return;
    }
    startSimulation();
  };

  const handleStop = () => {
    stopSimulation();
  };

  const handleReset = () => {
    stopSimulation();
    updateSimulationTime(0);
  };

  const calculateMetrics = () => {
    if (trajectoryPoints.length < 2) {
      return {
        maxSpeed: 0,
        avgSpeed: 0,
        totalTime: 0,
        pathLength: 0,
      };
    }

    // 计算路径长度
    let pathLength = 0;
    for (let i = 1; i < trajectoryPoints.length; i++) {
      const prev = trajectoryPoints[i - 1].position;
      const curr = trajectoryPoints[i].position;
      const distance = Math.sqrt(
        Math.pow(curr[0] - prev[0], 2) +
        Math.pow(curr[1] - prev[1], 2) +
        Math.pow(curr[2] - prev[2], 2)
      );
      pathLength += distance;
    }

    // 简化的速度计算
    const avgSpeed = pathLength / simulation.totalTime;
    const maxSpeed = avgSpeed * 1.5; // 假设最大速度是平均速度的1.5倍

    return {
      maxSpeed: Math.round(maxSpeed),
      avgSpeed: Math.round(avgSpeed),
      totalTime: simulation.totalTime,
      pathLength: Math.round(pathLength),
    };
  };

  const metrics = calculateMetrics();
  const progress = (simulation.currentTime / simulation.totalTime) * 100;

  return (
    <div>
      <Title level={2}>仿真预览</Title>
      
      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="仿真场景" size="small">
            <div style={{ height: 400, background: '#f5f5f5' }}>
              <SimulationViewer 
                trajectoryPoints={trajectoryPoints}
                currentTime={simulation.currentTime}
                totalTime={simulation.totalTime}
                isRunning={simulation.isRunning}
              />
            </div>
            
            <div style={{ marginTop: 16 }}>
              <Progress 
                percent={progress} 
                format={() => `${simulation.currentTime.toFixed(1)}s / ${simulation.totalTime}s`}
              />
            </div>
            
            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <Space size="large">
                <Button 
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleStart}
                  disabled={simulation.isRunning || trajectoryPoints.length < 2}
                >
                  开始
                </Button>
                <Button 
                  icon={<PauseCircleOutlined />}
                  onClick={handleStop}
                  disabled={!simulation.isRunning}
                >
                  停止
                </Button>
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={handleReset}
                >
                  重置
                </Button>
              </Space>
            </div>
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="仿真控制" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>仿真速度</Text>
                <Select
                  value={simulation.speed}
                  onChange={setSimulationSpeed}
                  style={{ width: '100%', marginTop: 8 }}
                  disabled={simulation.isRunning}
                >
                  <Select.Option value={0.5}>0.5x</Select.Option>
                  <Select.Option value={1}>1x</Select.Option>
                  <Select.Option value={2}>2x</Select.Option>
                  <Select.Option value={5}>5x</Select.Option>
                </Select>
              </div>
              
              <div>
                <Text strong>状态</Text>
                <div style={{ marginTop: 8 }}>
                  <Text type={simulation.isRunning ? 'success' : 'secondary'}>
                    {simulation.isRunning ? '● 运行中' : '○ 已停止'}
                  </Text>
                </div>
              </div>
            </Space>
          </Card>
          
          <Card title="性能指标" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="最大速度"
                  value={metrics.maxSpeed}
                  suffix="mm/s"
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="平均速度"
                  value={metrics.avgSpeed}
                  suffix="mm/s"
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
            </Row>
            
            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Statistic
                  title="总时间"
                  value={metrics.totalTime}
                  suffix="s"
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="路径长度"
                  value={metrics.pathLength}
                  suffix="mm"
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
            </Row>
          </Card>
          
          <Card title="轨迹信息" size="small" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>轨迹点数量: </Text>
                <Text>{trajectoryPoints.length}</Text>
              </div>
              
              {trajectoryPoints.length < 2 && (
                <div>
                  <Text type="warning">
                    ⚠️ 需要至少2个轨迹点才能开始仿真
                  </Text>
                </div>
              )}
              
              {trajectoryPoints.length >= 2 && (
                <div>
                  <Text type="success">
                    ✓ 轨迹配置完成，可以开始仿真
                  </Text>
                </div>
              )}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SimulationPage;
