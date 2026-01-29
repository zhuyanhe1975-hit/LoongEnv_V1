# 机器人运动控制系统 - UI实现计划

## 项目结构

```
ui/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── assets/
│       ├── icons/
│       ├── models/
│       └── images/
├── src/
│   ├── components/           # 通用组件
│   │   ├── common/          # 基础组件
│   │   ├── charts/          # 图表组件
│   │   ├── forms/           # 表单组件
│   │   └── layout/          # 布局组件
│   ├── pages/               # 页面组件
│   │   ├── Dashboard/
│   │   ├── Monitoring/
│   │   ├── Planning/
│   │   ├── Tuning/
│   │   └── Settings/
│   ├── hooks/               # 自定义Hooks
│   ├── services/            # API服务
│   ├── store/               # 状态管理
│   ├── utils/               # 工具函数
│   ├── types/               # TypeScript类型
│   ├── styles/              # 样式文件
│   └── App.tsx
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## 核心组件设计

### 1. 布局组件

#### MainLayout.tsx
```typescript
interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <AppBar />
      <Sidebar />
      <MainContent>{children}</MainContent>
      <StatusBar />
    </Box>
  );
};
```

#### Sidebar.tsx
```typescript
const menuItems = [
  { id: 'dashboard', label: '仪表板', icon: DashboardIcon, path: '/' },
  { id: 'monitoring', label: '实时监控', icon: MonitorIcon, path: '/monitoring' },
  { id: 'planning', label: '轨迹规划', icon: RouteIcon, path: '/planning' },
  { id: 'tuning', label: '参数调优', icon: TuneIcon, path: '/tuning' },
  { id: 'settings', label: '系统设置', icon: SettingsIcon, path: '/settings' },
];
```

### 2. 仪表板组件

#### Dashboard.tsx
```typescript
const Dashboard: React.FC = () => {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <SystemOverviewCard />
      </Grid>
      <Grid item xs={12} md={4}>
        <QuickActionsCard />
      </Grid>
      <Grid item xs={12} md={6}>
        <PerformanceChart />
      </Grid>
      <Grid item xs={12} md={6}>
        <RecentTasksList />
      </Grid>
    </Grid>
  );
};
```

#### SystemOverviewCard.tsx
```typescript
interface SystemStatus {
  isConnected: boolean;
  currentMode: string;
  jointPositions: number[];
  endEffectorPose: number[];
  safetyStatus: 'safe' | 'warning' | 'error';
}

const SystemOverviewCard: React.FC = () => {
  const { data: systemStatus } = useSystemStatus();
  
  return (
    <Card>
      <CardHeader title="系统状态" />
      <CardContent>
        <StatusIndicator status={systemStatus?.safetyStatus} />
        <RobotVisualization pose={systemStatus?.endEffectorPose} />
        <JointStatusGrid positions={systemStatus?.jointPositions} />
      </CardContent>
    </Card>
  );
};
```

### 3. 实时监控组件

#### MonitoringPage.tsx
```typescript
const MonitoringPage: React.FC = () => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} lg={8}>
        <Robot3DViewer />
      </Grid>
      <Grid item xs={12} lg={4}>
        <Stack spacing={2}>
          <JointMonitorCard />
          <ForceMonitorCard />
          <CollisionStatusCard />
        </Stack>
      </Grid>
      <Grid item xs={12}>
        <RealTimeChartsPanel />
      </Grid>
    </Grid>
  );
};
```

#### Robot3DViewer.tsx
```typescript
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const Robot3DViewer: React.FC = () => {
  const { robotModel } = useRobotModel();
  
  return (
    <Card sx={{ height: 500 }}>
      <CardContent sx={{ height: '100%', p: 0 }}>
        <Canvas camera={{ position: [2, 2, 2] }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          <RobotModel model={robotModel} />
          <OrbitControls />
        </Canvas>
      </CardContent>
    </Card>
  );
};
```

### 4. 轨迹规划组件

#### TrajectoryPlanningPage.tsx
```typescript
const TrajectoryPlanningPage: React.FC = () => {
  const [selectedTrajectory, setSelectedTrajectory] = useState<Trajectory | null>(null);
  
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={8}>
        <TrajectoryVisualization trajectory={selectedTrajectory} />
      </Grid>
      <Grid item xs={12} md={4}>
        <Stack spacing={2}>
          <TrajectoryLibrary onSelect={setSelectedTrajectory} />
          <TrajectoryEditor trajectory={selectedTrajectory} />
          <SimulationControls />
        </Stack>
      </Grid>
    </Grid>
  );
};
```

### 5. 参数调优组件

#### ParameterTuningPage.tsx
```typescript
const ParameterTuningPage: React.FC = () => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <ParameterConfigPanel />
      </Grid>
      <Grid item xs={12} md={6}>
        <OptimizationProgress />
      </Grid>
      <Grid item xs={12}>
        <TuningResultsChart />
      </Grid>
    </Grid>
  );
};
```

## 状态管理

### Redux Store 结构
```typescript
interface RootState {
  robot: RobotState;
  ui: UIState;
  monitoring: MonitoringState;
  planning: PlanningState;
  tuning: TuningState;
}

interface RobotState {
  isConnected: boolean;
  currentPose: number[];
  jointPositions: number[];
  safetyStatus: SafetyStatus;
  operationMode: OperationMode;
}

interface UIState {
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  notifications: Notification[];
}
```

### API服务层
```typescript
class RobotControlAPI {
  private baseURL = 'http://localhost:8000/api';
  
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await fetch(`${this.baseURL}/status`);
    return response.json();
  }
  
  async sendTrajectory(trajectory: Trajectory): Promise<void> {
    await fetch(`${this.baseURL}/trajectory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trajectory),
    });
  }
  
  async startParameterTuning(config: TuningConfig): Promise<string> {
    const response = await fetch(`${this.baseURL}/tuning/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const { taskId } = await response.json();
    return taskId;
  }
}
```

## 实时通信

### WebSocket连接
```typescript
class RobotWebSocketService {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Function[]> = new Map();
  
  connect(url: string) {
    this.ws = new WebSocket(url);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit(data.type, data.payload);
    };
  }
  
  subscribe(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }
  
  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event) || [];
    callbacks.forEach(callback => callback(data));
  }
}
```

### 自定义Hooks
```typescript
const useRealTimeData = (endpoint: string) => {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const ws = new WebSocketService();
    ws.connect(`ws://localhost:8000/ws/${endpoint}`);
    
    ws.subscribe('data', (newData) => {
      setData(newData);
      setIsLoading(false);
    });
    
    return () => ws.disconnect();
  }, [endpoint]);
  
  return { data, isLoading };
};

const useSystemStatus = () => {
  return useRealTimeData('system-status');
};

const useJointPositions = () => {
  return useRealTimeData('joint-positions');
};
```

## 图表组件

### 实时图表
```typescript
interface RealTimeChartProps {
  title: string;
  dataKey: string;
  maxDataPoints?: number;
  yAxisLabel?: string;
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({
  title,
  dataKey,
  maxDataPoints = 100,
  yAxisLabel
}) => {
  const [data, setData] = useState<ChartDataPoint[]>([]);
  
  useEffect(() => {
    const ws = new WebSocketService();
    ws.subscribe(dataKey, (newPoint: ChartDataPoint) => {
      setData(prev => {
        const updated = [...prev, newPoint];
        return updated.slice(-maxDataPoints);
      });
    });
  }, [dataKey, maxDataPoints]);
  
  return (
    <Card>
      <CardHeader title={title} />
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <XAxis dataKey="timestamp" />
            <YAxis label={{ value: yAxisLabel, angle: -90 }} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#2196F3" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
```

## 主题系统

### 主题配置
```typescript
const createAppTheme = (mode: 'light' | 'dark') => {
  return createTheme({
    palette: {
      mode,
      primary: {
        main: '#2196F3',
      },
      secondary: {
        main: '#FF9800',
      },
      background: {
        default: mode === 'dark' ? '#121212' : '#f5f5f5',
        paper: mode === 'dark' ? '#1E1E1E' : '#ffffff',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: mode === 'dark' 
              ? '0 4px 12px rgba(0,0,0,0.3)'
              : '0 2px 8px rgba(0,0,0,0.1)',
          },
        },
      },
    },
  });
};
```

## 开发工具配置

### package.json
```json
{
  "name": "robot-control-ui",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@mui/material": "^5.14.0",
    "@mui/icons-material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "@reduxjs/toolkit": "^1.9.0",
    "react-redux": "^8.1.0",
    "@tanstack/react-query": "^4.29.0",
    "react-router-dom": "^6.14.0",
    "@react-three/fiber": "^8.13.0",
    "@react-three/drei": "^9.77.0",
    "three": "^0.154.0",
    "recharts": "^2.7.0",
    "framer-motion": "^10.12.0",
    "socket.io-client": "^4.7.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/three": "^0.154.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^4.4.0",
    "eslint": "^8.45.0",
    "prettier": "^3.0.0"
  }
}
```

### vite.config.ts
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
```

这个实现计划提供了完整的现代UI架构，包括组件设计、状态管理、实时通信和开发配置。接下来我可以帮你实现具体的组件代码。