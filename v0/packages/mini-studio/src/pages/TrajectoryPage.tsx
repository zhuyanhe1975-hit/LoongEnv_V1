import React, { useState } from 'react';
import { Row, Col, Card, List, Button, Input, InputNumber, Typography, Space, Modal } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined, ClearOutlined } from '@ant-design/icons';
import { useAppStore } from '@/stores/appStore';
import TrajectoryViewer from '@/components/TrajectoryViewer';

const { Title, Text } = Typography;

const TrajectoryPage: React.FC = () => {
  const { 
    trajectoryPoints, 
    ui,
    addTrajectoryPoint, 
    updateTrajectoryPoint, 
    deleteTrajectoryPoint, 
    clearTrajectory,
    selectPoint 
  } = useAppStore();

  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingPoint, setEditingPoint] = useState<any>(null);
  const [pointForm, setPointForm] = useState({
    name: '',
    x: 0,
    y: 0,
    z: 500,
  });

  const handleAddPoint = () => {
    setEditingPoint(null);
    setPointForm({
      name: `P${trajectoryPoints.length}`,
      x: 0,
      y: 0,
      z: 500,
    });
    setIsModalVisible(true);
  };

  const handleEditPoint = (point: any) => {
    setEditingPoint(point);
    setPointForm({
      name: point.name,
      x: point.position[0],
      y: point.position[1],
      z: point.position[2],
    });
    setIsModalVisible(true);
  };

  const handleSavePoint = () => {
    const position: [number, number, number] = [pointForm.x, pointForm.y, pointForm.z];
    
    if (editingPoint) {
      updateTrajectoryPoint(editingPoint.id, {
        name: pointForm.name,
        position,
      });
    } else {
      addTrajectoryPoint({
        name: pointForm.name,
        position,
      });
    }
    
    setIsModalVisible(false);
  };

  const handleDeletePoint = (id: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个轨迹点吗？',
      onOk: () => deleteTrajectoryPoint(id),
    });
  };

  const handleClearTrajectory = () => {
    Modal.confirm({
      title: '确认清空',
      content: '确定要清空所有轨迹点吗？',
      onOk: clearTrajectory,
    });
  };

  return (
    <div>
      <Title level={2}>轨迹设计</Title>
      
      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card 
            title="轨迹点列表" 
            size="small"
            extra={
              <Space>
                <Button 
                  type="primary" 
                  size="small"
                  icon={<PlusOutlined />}
                  onClick={handleAddPoint}
                >
                  添加点
                </Button>
                <Button 
                  danger
                  size="small"
                  icon={<ClearOutlined />}
                  onClick={handleClearTrajectory}
                  disabled={trajectoryPoints.length === 0}
                >
                  清空
                </Button>
              </Space>
            }
          >
            <List
              size="small"
              dataSource={trajectoryPoints}
              renderItem={(point) => (
                <List.Item
                  style={{
                    background: ui.selectedPoint === point.id ? '#e6f7ff' : 'transparent',
                    padding: '8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  onClick={() => selectPoint(point.id)}
                  actions={[
                    <Button 
                      type="text" 
                      size="small"
                      icon={<EditOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEditPoint(point);
                      }}
                    />,
                    <Button 
                      type="text" 
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeletePoint(point.id);
                      }}
                    />,
                  ]}
                >
                  <List.Item.Meta
                    title={point.name}
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        X: {point.position[0]}, Y: {point.position[1]}, Z: {point.position[2]}
                      </Text>
                    }
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无轨迹点，点击"添加点"开始设计轨迹' }}
            />
          </Card>
          
          {ui.selectedPoint && (
            <Card title="点详情" size="small" style={{ marginTop: 16 }}>
              {(() => {
                const point = trajectoryPoints.find(p => p.id === ui.selectedPoint);
                if (!point) return null;
                
                return (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>名称: </Text>
                      <Text>{point.name}</Text>
                    </div>
                    <div>
                      <Text strong>位置: </Text>
                      <Text>X={point.position[0]}, Y={point.position[1]}, Z={point.position[2]}</Text>
                    </div>
                    <Button 
                      type="primary" 
                      size="small"
                      onClick={() => handleEditPoint(point)}
                    >
                      编辑
                    </Button>
                  </Space>
                );
              })()}
            </Card>
          )}
        </Col>
        
        <Col span={16}>
          <Card title="3D工作空间" size="small">
            <div style={{ height: 500, background: '#f5f5f5' }}>
              <TrajectoryViewer 
                trajectoryPoints={trajectoryPoints}
                selectedPoint={ui.selectedPoint}
                onPointSelect={selectPoint}
                onAddPoint={(position) => {
                  addTrajectoryPoint({
                    name: `P${trajectoryPoints.length}`,
                    position,
                  });
                }}
              />
            </div>
            
            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <Space>
                <Text type="secondary">点击空间添加轨迹点</Text>
                <Text type="secondary">|</Text>
                <Text type="secondary">拖拽轨迹点修改位置</Text>
                <Text type="secondary">|</Text>
                <Text type="secondary">点击轨迹点选择</Text>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>
      
      <Modal
        title={editingPoint ? '编辑轨迹点' : '添加轨迹点'}
        open={isModalVisible}
        onOk={handleSavePoint}
        onCancel={() => setIsModalVisible(false)}
        okText="保存"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>点名称</Text>
            <Input
              value={pointForm.name}
              onChange={(e) => setPointForm({ ...pointForm, name: e.target.value })}
              placeholder="请输入点名称"
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>X坐标 (mm)</Text>
            <InputNumber
              value={pointForm.x}
              onChange={(val) => setPointForm({ ...pointForm, x: val || 0 })}
              style={{ width: '100%', marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>Y坐标 (mm)</Text>
            <InputNumber
              value={pointForm.y}
              onChange={(val) => setPointForm({ ...pointForm, y: val || 0 })}
              style={{ width: '100%', marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>Z坐标 (mm)</Text>
            <InputNumber
              value={pointForm.z}
              onChange={(val) => setPointForm({ ...pointForm, z: val || 500 })}
              style={{ width: '100%', marginTop: 8 }}
            />
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default TrajectoryPage;
