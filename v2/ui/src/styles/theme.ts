import { createTheme, Theme } from '@mui/material/styles';

declare module '@mui/material/styles' {
  interface Palette {
    robotStatus: {
      safe: string;
      warning: string;
      error: string;
      offline: string;
    };
  }

  interface PaletteOptions {
    robotStatus?: {
      safe: string;
      warning: string;
      error: string;
      offline: string;
    };
  }
}

export const createAppTheme = (mode: 'light' | 'dark'): Theme => {
  const isDark = mode === 'dark';

  return createTheme({
    palette: {
      mode,
      primary: {
        main: '#2196F3',
        light: '#64B5F6',
        dark: '#1976D2',
        contrastText: '#ffffff',
      },
      secondary: {
        main: '#FF9800',
        light: '#FFB74D',
        dark: '#F57C00',
        contrastText: '#ffffff',
      },
      error: {
        main: '#F44336',
        light: '#EF5350',
        dark: '#D32F2F',
      },
      warning: {
        main: '#FFC107',
        light: '#FFEB3B',
        dark: '#FF8F00',
      },
      success: {
        main: '#4CAF50',
        light: '#81C784',
        dark: '#388E3C',
      },
      background: {
        default: isDark ? '#121212' : '#f5f5f5',
        paper: isDark ? '#1E1E1E' : '#ffffff',
      },
      text: {
        primary: isDark ? '#ffffff' : '#212121',
        secondary: isDark ? '#b3b3b3' : '#757575',
        disabled: isDark ? '#666666' : '#bdbdbd',
      },
      divider: isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
      robotStatus: {
        safe: '#4CAF50',
        warning: '#FFC107',
        error: '#F44336',
        offline: '#9E9E9E',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 600,
        lineHeight: 1.2,
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 500,
        lineHeight: 1.4,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 500,
        lineHeight: 1.4,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 500,
        lineHeight: 1.5,
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.5,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.43,
      },
      button: {
        textTransform: 'none',
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarWidth: 'thin',
            scrollbarColor: isDark ? '#424242 #1E1E1E' : '#c1c1c1 #f1f1f1',
            '&::-webkit-scrollbar': {
              width: 8,
            },
            '&::-webkit-scrollbar-track': {
              background: isDark ? '#1E1E1E' : '#f1f1f1',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: isDark ? '#424242' : '#c1c1c1',
              borderRadius: 4,
              '&:hover': {
                backgroundColor: isDark ? '#616161' : '#a8a8a8',
              },
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: isDark
              ? '0 4px 12px rgba(0, 0, 0, 0.3)'
              : '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: isDark ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
            backgroundImage: 'none',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 500,
            padding: '8px 16px',
          },
          contained: {
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            },
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
              backgroundColor: isDark ? 'rgba(255, 255, 255, 0.05)' : 'transparent',
              '& fieldset': {
                borderColor: isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)',
              },
              '&:hover fieldset': {
                borderColor: isDark ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)',
              },
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDark ? '#1E1E1E' : '#ffffff',
            color: isDark ? '#ffffff' : '#212121',
            boxShadow: isDark
              ? '0 1px 3px rgba(0, 0, 0, 0.3)'
              : '0 1px 3px rgba(0, 0, 0, 0.1)',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: isDark ? '#1E1E1E' : '#ffffff',
            borderRight: isDark
              ? '1px solid rgba(255, 255, 255, 0.1)'
              : '1px solid rgba(0, 0, 0, 0.1)',
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            margin: '2px 8px',
            '&.Mui-selected': {
              backgroundColor: isDark
                ? 'rgba(33, 150, 243, 0.2)'
                : 'rgba(33, 150, 243, 0.1)',
              '&:hover': {
                backgroundColor: isDark
                  ? 'rgba(33, 150, 243, 0.3)'
                  : 'rgba(33, 150, 243, 0.15)',
              },
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
          },
        },
      },
    },
  });
};