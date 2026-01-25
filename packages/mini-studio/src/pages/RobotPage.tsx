import React from 'react';
import { Row, Col, Card, Table, InputNumber, Button, Typography, Space, Select } from 'antd';
import { ReloadOutlined, CheckCircleOutlined, RobotOutlined } from '@ant-design/icons';
import { useAppStore } from '@/stores/appStore';
import { robotPresets } from '@/utils/robotPresets';
import RobotViewer from '@/components/RobotViewer';

const { Title, Text } = Typography;

const RobotPage: React.FC = () => {
  const { robot, updateDHParam, updateJointLimit, resetRobotToDefault, loadRobotPreset } = useAppStore();

  const dhColumns = [
    {
      title: '关节',
      dataIndex: 'joint',
      key: 'joint',
      width: 60,
    },
    {
      title: 'a (mm)',
      dataIndex: 'a',
      key: 'a',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateDHParam(record.joint, { a: val || 0 })}
          style={{ width: '100%' }}
        />
      ),
    },
    {
      title: 'd (mm)',
      dataIndex: 'd',
      key: 'd',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateDHParam(record.joint, { d: val || 0 })}
          style={{ width: '100%' }}
        />
      ),
    },
    {
      title: 'θ (°)',
      dataIndex: 'theta',
      key: 'theta',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateDHParam(record.joint, { theta: val || 0 })}
          style={{ width: '100%' }}
        />
      ),
    },
    {
      title: 'α (°)',
      dataIndex: 'alpha',
      key: 'alpha',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateDHParam(record.joint, { alpha: val || 0 })}
          style={{ width: '100%' }}
        />
      ),
    },
  ];

  const limitColumns = [
    {
      title: '关节',
      dataIndex: 'joint',
      key: 'joint',
      width: 60,
    },
    {
      title: '最小角度 (°)',
      dataIndex: 'min',
      key: 'min',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateJointLimit(record.joint, { min: val || -180 })}
          style={{ width: '100%' }}
        />
      ),
    },
    {
      title: '最大角度 (°)',
      dataIndex: 'max',
      key: 'max',
      render: (value: number, record: any) => (
        <InputNumber
          value={value}
          onChange={(val) => updateJointLimit(record.joint, { max: val || 180 })}
          style={{ width: '100%' }}
        />
      ),
    },
  ];

  const testForwardKinematics = () => {
    // TODO: 实现正运动学测试
    console.log('正运动学测试');
  };

  const testInverseKinematics = () => {
    // TODO: 实现逆运动学测试
    console.log('逆运动学测试');
  };

  return (
    <div>
      <Title level={2}>机械臂配置</Title>
      
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="DH参数配置" size="small">
            <Table
              columns={dhColumns}
              dataSource={robot.dhParams}
              rowKey="joint"
              pagination={false}
              size="small"
            />
            
            <div style={{ marginTop: 16, textAlign: 'right' }}>
              <Space>
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={resetRobotToDefault}
                >
                  重置默认
                </Button>
                <Select
                  placeholder="选择预设模型"
                  style={{ width: 200 }}
                  onChange={loadRobotPreset}
                >
                  {robotPresets.map(preset => (
                    <Select.Option key={preset.id} value={preset.id}>
                      <Space>
                        <RobotOutlined />
                        <div>
                          <div>{preset.name}</div>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {preset.manufacturer} | 负载{preset.workspace.payload}kg | 半径{preset.workspace.reach}mm
                          </div>
                        </div>
                      </Space>
                    </Select.Option>
                  ))}
                </Select>
              </Space>
            </div>
          </Card>
          
          <Card title="关节限制" size="small" style={{ marginTop: 16 }}>
            <Table
              columns={limitColumns}
              dataSource={robot.jointLimits}
              rowKey="joint"
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="3D机械臂模型" size="small">
            <div style={{ height: 400, background: '#f5f5f5' }}>
              <RobotViewer robot={robot} />
            </div>
            
            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <Space>
                <Button 
                  type="primary"
                  icon={<CheckCircleOutlined />}
                  onClick={testForwardKinematics}
                >
                  正运动学测试
                </Button>
                <Button 
                  icon={<CheckCircleOutlined />}
                  onClick={testInverseKinematics}
                >
                  逆运动学测试
                </Button>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>
      
      <Card title="配置说明" style={{ marginTop: 16 }}>
        <Text type="secondary">
          <strong>DH参数说明：</strong><br />
          • a: 连杆长度，沿X轴方向的距离<br />
          • d: 连杆偏移，沿Z轴方向的距离<br />
          • θ: 关节角度，绕Z轴的旋转角度<br />
          • α: 连杆扭转角，绕X轴的旋转角度<br /><br />
          
          <strong>关节限制：</strong><br />
          • 设置每个关节的运动范围，确保机械臂安全运行<br />
          • 角度单位为度，范围通常在-180°到180°之间
        </Text>
      </Card>
    </div>
  );
};

export default RobotPage;
