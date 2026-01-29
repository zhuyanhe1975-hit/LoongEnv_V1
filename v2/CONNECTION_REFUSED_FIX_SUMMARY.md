# 连接拒绝错误修复总结

## 问题描述
前端显示错误：`Failed to load resource: net::ERR_CONNECTION_REFUSED` 当尝试访问 `/api/robot/specs` 端点时。

## 根本原因
Vite开发服务器的代理配置指向了错误的后端端口。

### 问题分析过程

1. **后端API正常工作**：
   ```bash
   curl http://localhost:5003/api/robot/specs
   # ✅ 返回正确的机器人规格数据
   ```

2. **前端服务正常启动**：
   - React应用运行在 http://localhost:3001
   - 无TypeScript编译错误

3. **发现Vite代理配置问题**：
   - 检查 `ui/vite.config.ts` 发现代理指向 `localhost:8000`
   - 而实际后端运行在 `localhost:5003`

## 解决方案

### 1. 更新Vite代理配置
**文件**: `ui/vite.config.ts`

**修改前**:
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}
```

**修改后**:
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:5003',
    changeOrigin: true,
  },
}
```

### 2. 简化前端API配置
**文件**: `ui/src/services/backendService.ts`

**修改前**:
```typescript
const API_BASE_URL = 'http://localhost:5003/api';
```

**修改后**:
```typescript
const API_BASE_URL = '/api';
```

### 3. 重启开发服务器
重启Vite开发服务器以应用配置更改。

## 验证结果

### 通过Vite代理的API测试
```bash
# 健康检查
curl http://localhost:3001/api/health
# ✅ 返回: {"backend_available": true, "status": "healthy"}

# 机器人规格
curl http://localhost:3001/api/robot/specs
# ✅ 返回: 完整的ER15-1400机器人规格

# 机器人状态
curl http://localhost:3001/api/robot/status
# ✅ 返回: 实时机器人状态数据
```

### 后端日志确认
```
127.0.0.1 - - [28/Jan/2026 15:42:06] "GET /api/robot/specs HTTP/1.1" 200 -
127.0.0.1 - - [28/Jan/2026 15:42:10] "GET /api/health HTTP/1.1" 200 -
127.0.0.1 - - [28/Jan/2026 15:42:15] "GET /api/robot/status HTTP/1.1" 200 -
```

## 技术说明

### Vite代理工作原理
- Vite开发服务器在端口3001运行
- 所有 `/api/*` 请求被代理到 `http://localhost:5003`
- 前端代码使用相对路径 `/api`，由Vite自动转发

### 优势
1. **开发环境简化**: 前端不需要处理CORS问题
2. **配置集中**: 所有代理配置在Vite配置文件中
3. **生产环境兼容**: 相对路径在生产环境中也能正常工作

## 当前系统架构

```
浏览器 → http://localhost:3001 (Vite Dev Server)
                ↓ (代理 /api/*)
        http://localhost:5003 (Flask Backend)
```

### 服务端口分配
- **前端开发服务器**: http://localhost:3001
- **后端Flask API**: http://localhost:5003
- **API访问**: http://localhost:3001/api/* (通过代理)

## 修改文件清单

### 修改的文件
1. `ui/vite.config.ts` - 更新代理目标端口
2. `ui/src/services/backendService.ts` - 使用相对API路径

### 重启的服务
- Vite开发服务器 (应用配置更改)

## 测试状态

✅ 前端可以正常访问所有后端API端点
✅ 代理配置正确转发请求
✅ 后端正常接收和响应请求
✅ 无CORS错误
✅ 无连接拒绝错误

## 总结

连接拒绝错误已完全解决。问题的根本原因是Vite代理配置指向了错误的后端端口。通过更新代理配置并重启开发服务器，现在前端可以正常与后端通信，所有API端点都可以正常访问。