# Pencil (MCP) 安装与使用

本仓库已下载 Pencil Linux 桌面版（AppImage）到：

- `tools/pencil/Pencil-linux-x86_64.AppImage`

## 运行（Linux）

```bash
./tools/pencil/Pencil-linux-x86_64.AppImage
```

> 备注：AppImage 在部分发行版上可能需要安装 `libfuse2`（或等价的 FUSE2 兼容包）才能启动。

## 使用 MCP（给 Codex/IDE 用）

Pencil 的 MCP Server 会在你打开 Pencil 时自动启动（本地运行，无需额外配置）。 citeturn0search1

在 Codex CLI 中验证连接：

1. 先启动 Pencil
2. 在终端打开 Codex CLI
3. 运行 `/mcp`，列表里应出现 Pencil citeturn0search1turn0search0

如果你希望我“用 Pencil MCP 设计 UI”，需要你在同一台机器上把 Pencil 跑起来并保持打开，然后我才能通过 MCP 工具读写 `.pen` 文件来生成/修改设计。 citeturn0search0

