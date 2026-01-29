# Web界面

基于React + TypeScript构建的现代化机器人控制Web界面。

## 技术栈

### 前端
- **React 18**: UI框架
- **TypeScript**: 类型安全
- **Material-UI (MUI)**: UI组件库
- **Redux Toolkit**: 状态管理
- **Three.js**: 3D可视化
- **Vite**: 构建工具

### 后端
- **Flask**: Web框架
- **Flask-CORS**: 跨域支持
- **Python 3.12**: 运行环境

## 项目结构

```
ui/
├── src/                    # 前端源代码
│   ├── components/        # React组件
│   │   ├── common/       # 通用组件
│   │   ├── dashboard/    # 仪表盘组件
│   │   └── layout/       # 布局组件
│   ├── pages/            # 页面组件
│   │   ├── Dashboard/    # 仪表盘页面
│   │   ├── Planning/     # 轨迹规划页面
│   │   ├── Monitoring/   # 实时监控页面
│   │   ├── Tuning/       # 参数调优页面
│   │   └── Settings/     # 系统设置页面
│   ├── services/         # API服务
│   ├── store/            # Redux状态管理
│   ├── styles/           # 样式和主题
│   ├── types/            # TypeScript类型定义
│   ├── utils/            # 工具函数
│   ├── App.tsx           # 应用入口
│   └── main.tsx          # 主文件
├── public/               # 静态资源
│   └── models/          # 3D模型文件
├── dist/                # 构建输出
├── backend_api.py       # Flask后端API
├── package.json         # 依赖配置
├── tsconfig.json        # TypeScript配置
├── vite.config.ts       # Vite配置
└── index.html           # HTML模板
```

## 快速开始

### 安装依赖

```bash
cd ui
npm install
```

### 开发模式

```bash
# 启动前端开发服务器
npm run dev

# 启动后端API服务
python backend_api.py
```

访问 http://localhost:5173

### 生产构建

```bash
npm run build
```

构建输出在 `dist/` 目录。

## 功能模块

### 1. 仪表盘 (Dashboard)

**路径**: `/`

**功能**:
- 系统状态总览
- 3D机器人可视化
- 实时关节状态
- 性能指标图表
- 快速操作面板

**组件**:
- `IndustrialDashboard.tsx`: 主仪表盘
- `Enhanced3DViewer.tsx`: 3D可视化
- `RobotStatusCard.tsx`: 机器人状态卡片
- `PerformanceChart.tsx`: 性能图表

### 2. 轨迹规划 (Planning)

**路径**: `/planning`

**功能**:
- 路径点编辑
- 轨迹参数配置
- 轨迹预览
- 导入/导出轨迹
- 时间优化

**API端点**:
- `POST /api/trajectory/plan`: 规划轨迹
- `GET /api/trajectory/current`: 获取当前轨迹
- `POST /api/trajectory/waypoints/add`: 添加路径点

### 3. 实时监控 (Monitoring)

**路径**: `/monitoring`

**功能**:
- 实时关节状态
- 力矩监控
- 误差跟踪
- 性能指标
- 历史数据图表

**API端点**:
- `GET /api/robot/status`: 获取机器人状态
- `GET /api/monitoring/performance`: 获取性能指标

### 4. 参数调优 (Tuning)

**路径**: `/tuning`

**功能**:
- 优化算法配置
- 参数类型选择
- 性能权重设置
- 实时进度显示
- 优化结果展示
- 性能对比分析

**API端点**:
- `POST /api/tuning/start`: 启动调优
- `GET /api/tuning/status`: 获取调优状态
- `POST /api/tuning/stop`: 停止调优
- `POST /api/tuning/compare`: 性能对比

### 5. 系统设置 (Settings)

**路径**: `/settings`

**功能**:
- 主题切换（亮色/暗色）
- 语言设置
- 控制参数配置
- 系统参数管理
- 配置导入/导出

**API端点**:
- `GET /api/settings`: 获取设置
- `POST /api/settings`: 保存设置
- `POST /api/settings/reset`: 恢复默认

## API文档

### 后端API (backend_api.py)

Flask后端提供RESTful API接口。

#### 健康检查

```
GET /api/health
```

#### 机器人状态

```
GET /api/robot/status
GET /api/robot/specs
```

#### 轨迹规划

```
POST /api/trajectory/plan
GET /api/trajectory/current
POST /api/trajectory/waypoints/add
POST /api/trajectory/import
GET /api/trajectory/export
```

#### 参数调优

```
POST /api/tuning/start
GET /api/tuning/status
POST /api/tuning/stop
POST /api/tuning/compare
```

#### 控制增益

```
GET /api/control/gains
POST /api/control/gains
```

#### 系统设置

```
GET /api/settings
POST /api/settings
POST /api/settings/reset
GET /api/settings/export
POST /api/settings/import
```

## 状态管理

使用Redux Toolkit管理全局状态。

### Store结构

```typescript
{
  robot: {
    status: RobotStatus,
    specs: RobotSpecs,
    isConnected: boolean
  },
  planning: {
    waypoints: Waypoint[],
    trajectory: TrajectoryPoint[],
    params: TrajectoryParams
  },
  monitoring: {
    currentState: RobotState,
    performanceMetrics: PerformanceMetrics
  },
  tuning: {
    isRunning: boolean,
    progress: number,
    results: TuningResults
  },
  ui: {
    theme: 'light' | 'dark',
    language: string,
    notifications: Notification[]
  }
}
```

### Slices

- `robotSlice.ts`: 机器人状态
- `planningSlice.ts`: 轨迹规划
- `monitoringSlice.ts`: 实时监控
- `tuningSlice.ts`: 参数调优
- `uiSlice.ts`: UI状态

## 样式和主题

### 主题配置

两套主题：亮色和暗色。

**文件**: `src/styles/theme.ts`, `src/styles/industrialTheme.ts`

**特点**:
- 工业风格设计
- 高对比度
- 响应式布局
- 无障碍支持

### 自定义主题

```typescript
import { createTheme } from '@mui/material/styles';

const customTheme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#dc004e' }
  }
});
```

## 3D可视化

使用Three.js和React Three Fiber实现3D机器人可视化。

### 功能

- URDF模型加载
- STL网格渲染
- 实时关节更新
- 相机控制
- 光照和阴影

### 组件

- `Robot3DViewer.tsx`: 3D查看器
- `URDFRobotModel.tsx`: URDF模型加载
- `STLRobotModel.tsx`: STL模型加载
- `STLLoader.tsx`: STL文件加载器

## 开发指南

### 添加新页面

1. 在 `src/pages/` 创建页面组件
2. 在 `App.tsx` 添加路由
3. 在导航菜单添加链接
4. 创建对应的Redux slice（如需要）

### 添加新API

1. 在 `backend_api.py` 添加路由
2. 在 `src/services/backendService.ts` 添加方法
3. 在组件中调用API
4. 更新类型定义

### 样式规范

- 使用MUI的sx prop
- 遵循主题变量
- 响应式设计
- 无障碍标准

## 测试

```bash
# 运行测试
npm test

# 类型检查
npm run type-check

# Lint检查
npm run lint
```

## 构建优化

- 代码分割
- 懒加载
- Tree shaking
- 压缩和混淆

## 部署

### 开发环境

```bash
npm run dev
```

### 生产环境

```bash
# 构建
npm run build

# 预览
npm run preview
```

### Docker部署

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

## 常见问题

### Q: 3D模型不显示？

A: 检查：
1. 模型文件路径是否正确
2. WebGL是否支持
3. 浏览器控制台错误

### Q: API请求失败？

A: 确认：
1. 后端服务是否运行
2. CORS配置是否正确
3. API端点是否正确

### Q: 构建失败？

A: 尝试：
1. 删除 `node_modules` 重新安装
2. 清除构建缓存
3. 检查TypeScript错误

## 相关文档

- [UI设计文档](../docs/ui/)
- [API文档](../docs/reports/)
- [项目README](../README.md)

## 更新日期

2026-01-29
