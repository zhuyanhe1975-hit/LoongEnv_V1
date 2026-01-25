import React from 'react';
import { Layout, Menu, Button, Typography } from 'antd';
import { 
  FolderOutlined, 
  RobotOutlined, 
  AimOutlined, 
  PlayCircleOutlined,
  ExportOutlined 
} from '@ant-design/icons';
import { useAppStore } from '@/stores/appStore';
import ProjectPage from '@/pages/ProjectPage';
import RobotPage from '@/pages/RobotPage';
import TrajectoryPage from '@/pages/TrajectoryPage';
import SimulationPage from '@/pages/SimulationPage';

const { Header, Sider, Content, Footer } = Layout;
const { Title } = Typography;

const App: React.FC = () => {
  const { ui, currentProject, setActiveTab } = useAppStore();

  const menuItems = [
    {
      key: 'project',
      icon: <FolderOutlined />,
      label: '项目',
    },
    {
      key: 'robot',
      icon: <RobotOutlined />,
      label: '机械臂',
    },
    {
      key: 'trajectory',
      icon: <AimOutlined />,
      label: '轨迹',
    },
    {
      key: 'simulation',
      icon: <PlayCircleOutlined />,
      label: '仿真',
    },
  ];

  const renderContent = () => {
    switch (ui.activeTab) {
      case 'project':
        return <ProjectPage />;
      case 'robot':
        return <RobotPage />;
      case 'trajectory':
        return <TrajectoryPage />;
      case 'simulation':
        return <SimulationPage />;
      default:
        return <ProjectPage />;
    }
  };

  const handleExport = () => {
    const data = {
      project: currentProject,
      robot: useAppStore.getState().robot,
      trajectory: useAppStore.getState().trajectoryPoints,
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentProject?.name || 'project'}_config.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        background: '#001529',
        padding: '0 24px'
      }}>
        <Title level={3} style={{ color: 'white', margin: 0 }}>
          Mini LoongEnv Studio
        </Title>
        <Button 
          type="primary" 
          icon={<ExportOutlined />}
          onClick={handleExport}
          disabled={!currentProject}
        >
          导出配置
        </Button>
      </Header>
      
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[ui.activeTab]}
            items={menuItems}
            style={{ height: '100%', borderRight: 0 }}
            onClick={({ key }) => setActiveTab(key as any)}
          />
        </Sider>
        
        <Layout style={{ padding: '0 24px 24px' }}>
          <Content
            style={{
              background: '#fff',
              padding: 24,
              margin: 0,
              minHeight: 280,
            }}
          >
            {renderContent()}
          </Content>
        </Layout>
      </Layout>
      
      <Footer style={{ textAlign: 'center' }}>
        状态: 就绪 | 当前项目: {currentProject?.name || '未选择'}
      </Footer>
    </Layout>
  );
};

export default App;
