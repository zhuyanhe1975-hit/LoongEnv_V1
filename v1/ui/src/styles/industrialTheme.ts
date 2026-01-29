import { createTheme, Theme } from '@mui/material/styles';

// UI/UX Pro Max 推荐的工业控制系统设计系统
export const createIndustrialTheme = (mode: 'light' | 'dark'): Theme => {
  const isDark = mode === 'dark';

  return createTheme({
    palette: {
      mode,
      // UI/UX Pro Max 推荐：活力配色方案 - 蓝色主调 + 橙色强调
      primary: {
        main: '#2563EB', // 信任蓝 - 专业可靠
        light: '#3B82F6',
        dark: '#1D4ED8',
        contrastText: '#ffffff',
      },
      secondary: {
        main: '#6366F1', // 靛蓝 - 科技感
        light: '#818CF8',
        dark: '#4F46E5',
        contrastText: '#ffffff',
      },
      // 工业安全色彩标准 - 增强对比度
      error: {
        main: '#EF4444', // 鲜红 - 更醒目的错误提示
        light: '#F87171',
        dark: '#DC2626',
      },
      warning: {
        main: '#F59E0B', // 琥珀黄 - 保持警告色
        light: '#FBBF24',
        dark: '#D97706',
      },
      success: {
        main: '#10B981', // 翠绿 - 更鲜艳的成功色
        light: '#34D399',
        dark: '#059669',
      },
      info: {
        main: '#06B6D4', // 青蓝 - 更亮的信息色
        light: '#22D3EE',
        dark: '#0891B2',
      },
      background: {
        default: isDark ? '#0F172A' : '#F8FAFC', // 深蓝灰/浅灰白
        paper: isDark ? '#1E293B' : '#FFFFFF',
      },
      text: {
        primary: isDark ? '#F8FAFC' : '#1E293B',
        secondary: isDark ? '#94A3B8' : '#64748B',
        disabled: isDark ? '#475569' : '#94A3B8',
      },
      divider: isDark ? '#334155' : '#E2E8F0',
      // 扩展机器人状态色彩 - 更鲜艳的状态指示
      robotStatus: {
        safe: '#10B981',      // 翠绿
        warning: '#F59E0B',   // 琥珀黄
        error: '#EF4444',     // 鲜红
        offline: '#6B7280',   // 中性灰
      },
    },
    // UI/UX Pro Max 推荐：Fira Code/Fira Sans 字体组合
    typography: {
      fontFamily: '"Fira Sans", "Inter", system-ui, -apple-system, sans-serif',
      // 代码和数据显示使用等宽字体
      h1: {
        fontFamily: '"Fira Code", "JetBrains Mono", monospace',
        fontSize: '2.5rem',
        fontWeight: 600,
        lineHeight: 1.2,
        letterSpacing: '-0.02em',
      },
      h2: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '2rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h3: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '1.75rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h4: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '1.5rem',
        fontWeight: 700,
        lineHeight: 1.4,
      },
      h5: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '1.25rem',
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h6: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '1rem',
        fontWeight: 600,
        lineHeight: 1.5,
      },
      body1: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '0.875rem',
        lineHeight: 1.5,
      },
      body2: {
        fontFamily: '"Fira Sans", sans-serif',
        fontSize: '0.75rem',
        lineHeight: 1.43,
      },
      // 数据显示专用样式
      caption: {
        fontFamily: '"Fira Code", monospace',
        fontSize: '0.75rem',
        lineHeight: 1.4,
        letterSpacing: '0.02em',
      },
      button: {
        fontFamily: '"Fira Sans", sans-serif',
        textTransform: 'none',
        fontWeight: 600,
        letterSpacing: '0.01em',
      },
    },
    shape: {
      borderRadius: 8, // 更工业化的直角设计
    },
    components: {
      // 工业风格卡片
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            boxShadow: isDark 
              ? '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            border: isDark ? '1px solid #334155' : '1px solid #E2E8F0',
            backgroundImage: 'none',
            transition: 'all 0.2s ease-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: isDark
                ? '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)'
                : '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            },
          },
        },
      },
      // 工业风格按钮
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            textTransform: 'none',
            fontWeight: 600,
            padding: '10px 20px',
            cursor: 'pointer', // UI/UX Pro Max 要求
            transition: 'all 0.2s ease-out', // UI/UX Pro Max 推荐时长
          },
          contained: {
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 4px 8px rgba(37, 99, 235, 0.25)', // 蓝色阴影
              transform: 'translateY(-1px)',
            },
            '&:active': {
              transform: 'translateY(0)',
            },
          },
          outlined: {
            borderWidth: 2,
            '&:hover': {
              borderWidth: 2,
              backgroundColor: isDark ? 'rgba(37, 99, 235, 0.1)' : 'rgba(37, 99, 235, 0.05)',
            },
          },
        },
      },
      // 数据密集型表格
      MuiTableCell: {
        styleOverrides: {
          root: {
            fontFamily: '"Fira Code", monospace',
            fontSize: '0.75rem',
            padding: '8px 12px',
            borderBottom: isDark ? '1px solid #334155' : '1px solid #E2E8F0',
          },
          head: {
            fontFamily: '"Fira Sans", sans-serif',
            fontWeight: 600,
            backgroundColor: isDark ? '#334155' : '#F1F5F9',
            color: isDark ? '#F1F5F9' : '#334155',
          },
        },
      },
      // 工业风格输入框
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 6,
              backgroundColor: isDark ? 'rgba(30, 41, 59, 0.5)' : 'rgba(248, 250, 252, 0.8)',
              fontFamily: '"Fira Code", monospace',
              '& fieldset': {
                borderColor: isDark ? '#475569' : '#CBD5E1',
                borderWidth: 2,
              },
              '&:hover fieldset': {
                borderColor: isDark ? '#2563EB' : '#3B82F6',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#2563EB',
                borderWidth: 2,
                boxShadow: '0 0 0 3px rgba(37, 99, 235, 0.1)',
              },
            },
          },
        },
      },
      // 状态指示器
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 4,
            fontFamily: '"Fira Sans", sans-serif',
            fontWeight: 600,
            fontSize: '0.75rem',
            cursor: 'pointer', // UI/UX Pro Max 要求
          },
        },
      },
      // 导航栏
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDark ? '#1E293B' : '#FFFFFF',
            color: isDark ? '#F1F5F9' : '#334155',
            boxShadow: isDark
              ? '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            borderBottom: isDark ? '1px solid #334155' : '1px solid #E2E8F0',
          },
        },
      },
      // 侧边栏
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: isDark ? '#1E293B' : '#F8FAFC',
            borderRight: isDark ? '1px solid #334155' : '1px solid #E2E8F0',
          },
        },
      },
      // 列表项
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            margin: '2px 8px',
            cursor: 'pointer', // UI/UX Pro Max 要求
            transition: 'all 0.2s ease-out', // UI/UX Pro Max 推荐
            '&.Mui-selected': {
              backgroundColor: '#2563EB',
              color: '#FFFFFF',
              '&:hover': {
                backgroundColor: '#1D4ED8',
              },
            },
            '&:hover': {
              backgroundColor: isDark ? 'rgba(37, 99, 235, 0.1)' : 'rgba(37, 99, 235, 0.05)',
            },
          },
        },
      },
    },
  });
};

// 导出Google Fonts链接
export const INDUSTRIAL_FONTS_URL = 'https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap';