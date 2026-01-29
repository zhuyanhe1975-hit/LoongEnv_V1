# 机器人控制界面

基于React的现代化机器人运动控制系统用户界面。

## 🚀 快速开始

界面现已完全可用！访问地址：
- **http://localhost:3001/** (如果3000端口被占用)
- **http://localhost:3000/** (默认端口)

## ✅ 当前状态

- ✅ 开发服务器运行成功
- ✅ 所有TypeScript编译错误已修复
- ✅ 所有页面组件已实现
- ✅ Redux状态管理已正确配置
- ✅ 现代深色主题已应用
- ✅ 响应式布局正常工作

## 🎯 功能特性

- **实时仪表板** - 系统概览与状态卡片
- **监控页面** - CPU、内存、温度监控
- **规划页面** - 轨迹规划与算法选择
- **调优页面** - PID参数和动力学调优
- **设置页面** - 系统配置管理
- **现代界面** - Material Design 3.0工业美学设计
- **深色主题** - 专业控制界面样式
- **响应式设计** - 适配不同屏幕尺寸

## 🛠 技术栈

- React 18 + TypeScript
- Material-UI (MUI) v5
- Redux Toolkit 状态管理
- React Router 路由导航
- Vite 开发构建工具
- Three.js 3D可视化

## 📋 可用命令

```bash
# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 类型检查
npm run type-check

# 代码检查
npm run lint
npm run lint:fix
```

## 📁 项目结构

```
src/
├── components/          # 可复用UI组件
│   ├── common/         # 通用组件 (连接状态、通知提供器)
│   └── layout/         # 布局组件 (应用栏、侧边栏、状态栏)
├── pages/              # 页面组件
│   ├── Dashboard/      # 仪表板与系统概览
│   ├── Monitoring/     # 实时系统监控
│   ├── Planning/       # 轨迹规划界面
│   ├── Tuning/         # 参数调优标签页
│   └── Settings/       # 系统配置
├── store/              # Redux状态存储
│   └── slices/         # Redux切片 (机器人、界面、监控、规划、调优)
├── styles/             # 主题配置
├── types/              # TypeScript类型定义
└── utils/              # 工具函数
```

## 🔧 配置说明

- 后端API: `http://localhost:8000` (在 `vite.config.ts` 中配置)
- WebSocket: `ws://localhost:8000/ws`
- 开发端口: 3001 (如果3000被占用则自动选择)

## 🎨 已实现的UI组件

### 仪表板
- SystemOverviewCard - 主要系统指标
- QuickActionsCard - 机器人控制按钮
- PerformanceChart - 系统性能可视化
- RecentTasksList - 任务历史与状态
- RobotStatusCard - 实时机器人状态
- SafetyStatusCard - 安全系统监控

### 页面
- **监控** - 系统指标、警报、实时图表
- **规划** - 算法选择、轨迹参数、结果
- **调优** - PID控制器、动力学、安全参数
- **设置** - 主题、语言、通知、系统信息

## 🔗 后续步骤

1. **后端集成** - 连接到您的Python机器人控制系统
2. **真实数据** - 用实际系统数据替换模拟数据
3. **WebSocket连接** - 实现实时数据流传输
4. **3D可视化** - 添加机器人模型和轨迹可视化
5. **Pencil集成** - 安装Pencil MCP工具进行高级UI设计

## 🐛 故障排除

如果遇到问题：

1. **3000端口被占用**: 服务器将自动使用3001端口
2. **构建错误**: 运行 `npm run type-check` 识别TypeScript问题
3. **依赖缺失**: 运行 `npm install` 确保所有包已安装
4. **404错误**: 确保 `index.html` 在根目录中 (不在 `public/` 中)

界面现已完全可用，所有组件均已实现并正常工作！