# LoongEnv-Studio 开发环境配置

## 1. 开发环境要求

### 系统要求
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Node.js**: >= 18.0.0
- **Python**: >= 3.9.0
- **PostgreSQL**: >= 13.0
- **Redis**: >= 6.0
- **Git**: >= 2.30.0

### 开发工具
- **IDE**: VS Code (推荐) / WebStorm / PyCharm
- **浏览器**: Chrome >= 90 / Firefox >= 88
- **API测试**: Postman / Insomnia
- **数据库管理**: pgAdmin / DBeaver

## 2. 项目初始化

### 2.1 创建Studio前端项目
```bash
# 进入packages目录
cd packages

# 创建React项目
npm create vite@latest studio -- --template react-ts
cd studio

# 安装依赖
npm install antd @ant-design/pro-components
npm install three @react-three/fiber @react-three/drei
npm install zustand @tanstack/react-query
npm install react-router-dom axios
npm install @types/three lodash-es @types/lodash-es

# 开发依赖
npm install -D @types/node
npm install -D eslint-config-prettier eslint-plugin-react-hooks
npm install -D @testing-library/jest-dom @testing-library/user-event
npm install -D vitest jsdom
```

### 2.2 创建后端API项目
```bash
# 创建Python虚拟环境
cd packages
python -m venv studio-api
cd studio-api

# 激活虚拟环境 (Linux/Mac)
source bin/activate
# 激活虚拟环境 (Windows)
# Scripts\activate

# 安装依赖
pip install fastapi[all] uvicorn[standard]
pip install sqlalchemy alembic psycopg2-binary
pip install redis pydantic pydantic-settings
pip install python-multipart python-jose[cryptography]
pip install passlib[bcrypt] numpy scipy
pip install pytest pytest-asyncio httpx
```

## 3. 项目配置文件

### 3.1 前端配置 (vite.config.ts)
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@services': path.resolve(__dirname, './src/services'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@types': path.resolve(__dirname, './src/types'),
    },
  },
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
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          antd: ['antd', '@ant-design/pro-components'],
          three: ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
  },
});
```

### 3.2 TypeScript配置 (tsconfig.json)
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@pages/*": ["src/pages/*"],
      "@hooks/*": ["src/hooks/*"],
      "@stores/*": ["src/stores/*"],
      "@services/*": ["src/services/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### 3.3 ESLint配置 (.eslintrc.js)
```javascript
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
    'prettier',
  ],
  ignorePatterns: ['dist', '.eslintrc.js'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
  },
};
```

### 3.4 后端配置 (app/core/config.py)
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "LoongEnv Studio API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 数据库配置
    DATABASE_URL: str = "postgresql://user:password@localhost/loongenv_studio"
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379"
    
    # JWT配置
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 文件存储配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "loongenv-studio"
    
    # CORS配置
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 4. Docker配置

### 4.1 前端Dockerfile
```dockerfile
# packages/studio/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 4.2 后端Dockerfile
```dockerfile
# packages/studio-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  # 前端服务
  studio-frontend:
    build:
      context: ./packages/studio
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - studio-api
    environment:
      - NODE_ENV=production

  # 后端API服务
  studio-api:
    build:
      context: ./packages/studio-api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - minio
    environment:
      - DATABASE_URL=postgresql://loongenv:password@postgres:5432/loongenv_studio
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
    volumes:
      - ./data/uploads:/app/uploads

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=loongenv_studio
      - POSTGRES_USER=loongenv
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

## 5. 开发脚本

### 5.1 根目录package.json脚本
```json
{
  "scripts": {
    "dev": "concurrently \"npm run dev:api\" \"npm run dev:frontend\"",
    "dev:frontend": "cd packages/studio && npm run dev",
    "dev:api": "cd packages/studio-api && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000",
    "build": "npm run build:frontend && npm run build:api",
    "build:frontend": "cd packages/studio && npm run build",
    "build:api": "cd packages/studio-api && python -m py_compile main.py",
    "test": "npm run test:frontend && npm run test:api",
    "test:frontend": "cd packages/studio && npm run test",
    "test:api": "cd packages/studio-api && python -m pytest",
    "lint": "npm run lint:frontend && npm run lint:api",
    "lint:frontend": "cd packages/studio && npm run lint",
    "lint:api": "cd packages/studio-api && python -m flake8 .",
    "docker:up": "docker-compose up -d",
    "docker:down": "docker-compose down",
    "docker:build": "docker-compose build"
  }
}
```

### 5.2 数据库初始化脚本
```sql
-- scripts/init.sql
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入默认管理员用户
INSERT INTO users (username, email, hashed_password) VALUES 
('admin', 'admin@loongenv.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW');

-- 创建项目相关表
-- (之前在architecture.md中定义的表结构)
```

## 6. CI/CD配置

### 6.1 GitHub Actions工作流
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: packages/studio/package-lock.json
      
      - name: Install dependencies
        run: cd packages/studio && npm ci
      
      - name: Run tests
        run: cd packages/studio && npm run test
      
      - name: Run linting
        run: cd packages/studio && npm run lint
      
      - name: Build
        run: cd packages/studio && npm run build

  test-backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd packages/studio-api
          pip install -r requirements.txt
          pip install pytest pytest-asyncio httpx
      
      - name: Run tests
        run: cd packages/studio-api && python -m pytest
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db

  deploy:
    needs: [test-frontend, test-backend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and push Docker images
        run: |
          docker build -t loongenv/studio-frontend:latest packages/studio
          docker build -t loongenv/studio-api:latest packages/studio-api
          # Push to registry (需要配置Docker Hub secrets)
```

这个开发环境配置提供了完整的开发、测试和部署流程。需要我继续创建具体的代码实现吗？
