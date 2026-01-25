import React, { useState } from 'react';
import { Card, Input, Button, Select, Space, Typography, Divider } from 'antd';
import { PlusOutlined, DeleteOutlined, ImportOutlined } from '@ant-design/icons';
import { useAppStore } from '@/stores/appStore';

const { Title, Text } = Typography;
const { TextArea } = Input;

const ProjectPage: React.FC = () => {
  const { currentProject, createProject, updateProject } = useAppStore();
  const [isCreating, setIsCreating] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');

  const handleCreateProject = () => {
    if (projectName.trim()) {
      createProject(projectName.trim(), projectDescription.trim());
      setProjectName('');
      setProjectDescription('');
      setIsCreating(false);
    }
  };

  const handleUpdateProject = (field: string, value: string) => {
    if (currentProject) {
      updateProject({ [field]: value });
    }
  };

  const handleImportConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const config = JSON.parse(e.target?.result as string);
          if (config.project) {
            createProject(config.project.name, config.project.description);
            // TODO: 导入机械臂和轨迹配置
          }
        } catch (error) {
          console.error('导入配置失败:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  if (!currentProject && !isCreating) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 0' }}>
        <Title level={3}>欢迎使用 Mini LoongEnv Studio</Title>
        <Text type="secondary">开始创建您的第一个机械臂项目</Text>
        <div style={{ marginTop: 32 }}>
          <Space size="large">
            <Button 
              type="primary" 
              size="large"
              icon={<PlusOutlined />}
              onClick={() => setIsCreating(true)}
            >
              新建项目
            </Button>
            <Button 
              size="large"
              icon={<ImportOutlined />}
              onClick={() => document.getElementById('import-input')?.click()}
            >
              导入项目
            </Button>
          </Space>
          <input
            id="import-input"
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleImportConfig}
          />
        </div>
      </div>
    );
  }

  if (isCreating) {
    return (
      <Card title="创建新项目" style={{ maxWidth: 600, margin: '0 auto' }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div>
            <Text strong>项目名称 *</Text>
            <Input
              placeholder="请输入项目名称"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>项目描述</Text>
            <TextArea
              placeholder="请输入项目描述"
              value={projectDescription}
              onChange={(e) => setProjectDescription(e.target.value)}
              rows={3}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>机械臂类型</Text>
            <Select
              value="industrial_6axis"
              disabled
              style={{ width: '100%', marginTop: 8 }}
            >
              <Select.Option value="industrial_6axis">6轴工业机械臂</Select.Option>
            </Select>
          </div>
          
          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setIsCreating(false)}>
                取消
              </Button>
              <Button 
                type="primary" 
                onClick={handleCreateProject}
                disabled={!projectName.trim()}
              >
                创建项目
              </Button>
            </Space>
          </div>
        </Space>
      </Card>
    );
  }

  return (
    <div>
      <Title level={2}>项目管理</Title>
      
      <Card>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div>
            <Text strong>项目名称</Text>
            <Input
              value={currentProject?.name || ''}
              onChange={(e) => handleUpdateProject('name', e.target.value)}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>项目描述</Text>
            <TextArea
              value={currentProject?.description || ''}
              onChange={(e) => handleUpdateProject('description', e.target.value)}
              rows={3}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>机械臂类型</Text>
            <Select
              value={currentProject?.robotType || 'industrial_6axis'}
              disabled
              style={{ width: '100%', marginTop: 8 }}
            >
              <Select.Option value="industrial_6axis">6轴工业机械臂</Select.Option>
            </Select>
          </div>
          
          <Divider />
          
          <div>
            <Text type="secondary">
              创建时间: {currentProject?.createdAt ? new Date(currentProject.createdAt).toLocaleString() : ''}
            </Text>
            <br />
            <Text type="secondary">
              更新时间: {currentProject?.updatedAt ? new Date(currentProject.updatedAt).toLocaleString() : ''}
            </Text>
          </div>
          
          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button 
                danger
                icon={<DeleteOutlined />}
                onClick={() => {
                  if (confirm('确定要删除当前项目吗？')) {
                    // TODO: 实现删除项目功能
                    window.location.reload();
                  }
                }}
              >
                删除项目
              </Button>
              <Button 
                icon={<ImportOutlined />}
                onClick={() => document.getElementById('import-input')?.click()}
              >
                导入配置
              </Button>
            </Space>
          </div>
        </Space>
        
        <input
          id="import-input"
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleImportConfig}
        />
      </Card>
    </div>
  );
};

export default ProjectPage;
