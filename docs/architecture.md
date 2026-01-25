# LoongEnv-Studio 技术架构设计

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoongEnv-Studio 架构                         │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │
│ │   前端应用       │ │   API网关       │ │   后端服务       │     │
│ │                 │ │                 │ │                 │     │
│ │ React + TS      │◄┤ FastAPI        │◄┤ 业务逻辑        │     │
│ │ Ant Design      │ │ 认证授权        │ │ 数据处理        │     │
│ │ Three.js        │ │ 请求路由        │ │ 文件管理        │     │
│ │ Zustand         │ │ 负载均衡        │ │ 计算服务        │     │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘     │
│                                                                 │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │
│ │   数据存储       │ │   缓存层        │ │   文件存储       │     │
│ │                 │ │                 │ │                 │     │
│ │ PostgreSQL      │ │ Redis           │ │ MinIO           │     │
│ │ 项目数据        │ │ 会话缓存        │ │ 3D模型文件      │     │
│ │ 配置参数        │ │ 计算结果        │ │ 项目文件        │     │
│ │ 用户信息        │ │ 临时数据        │ │ 导出文件        │     │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 前端架构设计

### 2.1 项目结构
```
packages/studio/
├── public/                 # 静态资源
├── src/
│   ├── components/         # 通用组件
│   │   ├── ui/            # 基础UI组件
│   │   ├── 3d/            # 3D相关组件
│   │   └── forms/         # 表单组件
│   ├── pages/             # 页面组件
│   │   ├── project/       # 项目管理
│   │   ├── task/          # 任务设计
│   │   ├── robot/         # 机械臂建模
│   │   ├── control/       # 控制配置
│   │   └── simulation/    # 仿真验证
│   ├── hooks/             # 自定义Hooks
│   ├── stores/            # 状态管理
│   ├── services/          # API服务
│   ├── utils/             # 工具函数
│   ├── types/             # 类型定义
│   └── constants/         # 常量定义
├── package.json
└── vite.config.ts
```

### 2.2 核心技术栈
```typescript
// 主要依赖
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "antd": "^5.12.0",
    "@ant-design/pro-components": "^2.6.0",
    "three": "^0.158.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei": "^9.88.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.0.0",
    "react-router-dom": "^6.18.0",
    "axios": "^1.6.0",
    "lodash-es": "^4.17.21"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.1.0",
    "eslint": "^8.53.0",
    "@typescript-eslint/eslint-plugin": "^6.10.0",
    "prettier": "^3.0.0",
    "vitest": "^0.34.0",
    "@testing-library/react": "^13.4.0"
  }
}
```

### 2.3 状态管理架构
```typescript
// stores/index.ts
export interface AppState {
  // 项目状态
  project: {
    current: Project | null;
    list: Project[];
    loading: boolean;
  };
  
  // 任务设计状态
  task: {
    workspace: Workspace;
    trajectory: TrajectoryPoint[];
    constraints: Constraints;
  };
  
  // 机械臂状态
  robot: {
    dhParams: DHParameter[];
    dynamics: DynamicsParameter[];
    limits: JointLimits[];
    model: RobotModel | null;
  };
  
  // 控制配置状态
  control: {
    controllers: Controller[];
    parameters: ControlParameters;
    safety: SafetySettings;
  };
  
  // 仿真状态
  simulation: {
    running: boolean;
    time: number;
    data: SimulationData[];
    performance: PerformanceMetrics;
  };
}
```

## 3. 后端架构设计

### 3.1 项目结构
```
packages/studio-api/
├── app/
│   ├── api/               # API路由
│   │   ├── v1/
│   │   │   ├── projects/  # 项目管理API
│   │   │   ├── tasks/     # 任务设计API
│   │   │   ├── robots/    # 机械臂建模API
│   │   │   ├── control/   # 控制配置API
│   │   │   └── simulation/ # 仿真验证API
│   │   └── deps.py        # 依赖注入
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   ├── security.py    # 安全认证
│   │   └── database.py    # 数据库连接
│   ├── models/            # 数据模型
│   ├── schemas/           # Pydantic模式
│   ├── services/          # 业务逻辑
│   └── utils/             # 工具函数
├── requirements.txt
└── main.py
```

### 3.2 核心依赖
```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.0
pydantic-settings==2.0.3
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
numpy==1.24.3
scipy==1.11.4
```

### 3.3 数据模型设计
```python
# models/project.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000))
    robot_type = Column(String(50), nullable=False)
    task_type = Column(String(50), nullable=False)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联关系
    tasks = relationship("Task", back_populates="project")
    robots = relationship("Robot", back_populates="project")
    simulations = relationship("Simulation", back_populates="project")

class Robot(Base):
    __tablename__ = "robots"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    name = Column(String(255), nullable=False)
    dh_parameters = Column(JSON)  # DH参数
    dynamics_parameters = Column(JSON)  # 动力学参数
    joint_limits = Column(JSON)  # 关节限制
    model_file_path = Column(String(500))  # 3D模型文件路径
    
    project = relationship("Project", back_populates="robots")
```

## 4. API设计规范

### 4.1 RESTful API设计
```python
# api/v1/projects.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("/", response_model=List[ProjectResponse])
async def get_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取项目列表"""
    return project_service.get_projects(db, skip=skip, limit=limit)

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db)
):
    """创建新项目"""
    return project_service.create_project(db, project)

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: Session = Depends(get_db)
):
    """获取项目详情"""
    project = project_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project: ProjectUpdate,
    db: Session = Depends(get_db)
):
    """更新项目"""
    return project_service.update_project(db, project_id, project)
```

### 4.2 WebSocket实时通信
```python
# api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/ws/simulation/{project_id}")
async def simulation_websocket(websocket: WebSocket, project_id: int):
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理仿真控制命令
            if message["type"] == "start_simulation":
                await simulation_service.start_simulation(project_id)
            elif message["type"] == "stop_simulation":
                await simulation_service.stop_simulation(project_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## 5. 3D渲染架构

### 5.1 Three.js集成
```typescript
// components/3d/RobotViewer.tsx
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Axes } from '@react-three/drei';

export const RobotViewer: React.FC<RobotViewerProps> = ({
  robot,
  trajectory,
  onTrajectoryChange
}) => {
  return (
    <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
      {/* 环境设置 */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      
      {/* 控制器 */}
      <OrbitControls enablePan enableZoom enableRotate />
      
      {/* 坐标系和网格 */}
      <Grid args={[10, 10]} />
      <Axes scale={[1, 1, 1]} />
      
      {/* 机械臂模型 */}
      <RobotModel robot={robot} />
      
      {/* 轨迹显示 */}
      <TrajectoryPath 
        points={trajectory} 
        onChange={onTrajectoryChange}
      />
      
      {/* 工作空间边界 */}
      <WorkspaceBounds bounds={robot.workspace} />
    </Canvas>
  );
};
```

### 5.2 性能优化策略
```typescript
// utils/3d-optimization.ts
export class RenderOptimizer {
  private static instance: RenderOptimizer;
  private lodManager: LODManager;
  private instanceManager: InstanceManager;
  
  // LOD (Level of Detail) 管理
  setupLOD(object: Object3D, distances: number[]) {
    const lod = new LOD();
    distances.forEach((distance, index) => {
      const mesh = this.createLODMesh(object, index);
      lod.addLevel(mesh, distance);
    });
    return lod;
  }
  
  // 实例化渲染
  createInstancedMesh(geometry: BufferGeometry, material: Material, count: number) {
    const instancedMesh = new InstancedMesh(geometry, material, count);
    return instancedMesh;
  }
  
  // 视锥剔除
  frustumCulling(camera: Camera, objects: Object3D[]) {
    const frustum = new Frustum();
    frustum.setFromProjectionMatrix(camera.projectionMatrix);
    
    return objects.filter(obj => {
      return frustum.intersectsObject(obj);
    });
  }
}
```

## 6. 数据库设计

### 6.1 核心表结构
```sql
-- 项目表
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    robot_type VARCHAR(50) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 机械臂表
CREATE TABLE robots (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    dh_parameters JSONB NOT NULL,
    dynamics_parameters JSONB,
    joint_limits JSONB NOT NULL,
    model_file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 任务表
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    workspace JSONB NOT NULL,
    trajectory JSONB NOT NULL,
    constraints JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 控制配置表
CREATE TABLE control_configs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    controller_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    safety_settings JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 仿真结果表
CREATE TABLE simulation_results (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    simulation_time REAL NOT NULL,
    joint_angles JSONB NOT NULL,
    joint_velocities JSONB,
    joint_torques JSONB,
    end_effector_pose JSONB NOT NULL,
    performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 索引优化
```sql
-- 性能优化索引
CREATE INDEX idx_projects_robot_type ON projects(robot_type);
CREATE INDEX idx_projects_task_type ON projects(task_type);
CREATE INDEX idx_robots_project_id ON robots(project_id);
CREATE INDEX idx_tasks_project_id ON tasks(project_id);
CREATE INDEX idx_simulation_results_project_id ON simulation_results(project_id);
CREATE INDEX idx_simulation_results_time ON simulation_results(simulation_time);

-- JSONB字段索引
CREATE INDEX idx_projects_config ON projects USING GIN(config);
CREATE INDEX idx_robots_dh_params ON robots USING GIN(dh_parameters);
CREATE INDEX idx_tasks_trajectory ON tasks USING GIN(trajectory);
```

这个技术架构设计为LoongEnv-Studio提供了完整的技术实现方案。需要我继续设计开发环境配置和构建流程吗？
