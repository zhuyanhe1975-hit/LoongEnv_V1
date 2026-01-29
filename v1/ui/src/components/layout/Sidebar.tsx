import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItemButton,
  Typography,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Visibility as MonitorIcon,
  Route as RouteIcon,
  Tune as TuneIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import { useAppDispatch } from '../../store';
import { setCurrentPage } from '../../store/slices/uiSlice';

type NavKey = 'dashboard' | 'monitoring' | 'planning' | 'tuning' | 'settings';

interface NavItem {
  key: NavKey;
  label: string;
  path: string;
  icon: React.ReactElement;
}

const navItems: NavItem[] = [
  { key: 'dashboard', label: '模型加载', path: '/', icon: <DashboardIcon sx={{ fontSize: 16 }} /> },
  { key: 'monitoring', label: '实时监控', path: '/monitoring', icon: <MonitorIcon sx={{ fontSize: 16 }} /> },
  { key: 'planning', label: '轨迹规划', path: '/planning', icon: <RouteIcon sx={{ fontSize: 16 }} /> },
  { key: 'tuning', label: '参数调优', path: '/tuning', icon: <TuneIcon sx={{ fontSize: 16 }} /> },
  { key: 'settings', label: '系统设置', path: '/settings', icon: <SettingsIcon sx={{ fontSize: 16 }} /> },
];

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useAppDispatch();

  const selectedKey: NavKey = React.useMemo(() => {
    const pathname = location.pathname || '/';
    if (pathname.startsWith('/monitoring')) return 'monitoring';
    if (pathname.startsWith('/planning')) return 'planning';
    if (pathname.startsWith('/tuning')) return 'tuning';
    if (pathname.startsWith('/settings')) return 'settings';
    return 'dashboard';
  }, [location.pathname]);

  React.useEffect(() => {
    dispatch(setCurrentPage(selectedKey));
  }, [dispatch, selectedKey]);

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 220,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 220,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: (theme) => `1px solid ${theme.palette.divider}`,
          overflowX: 'hidden',
          top: 64,
          height: 'calc(100% - 64px)',
        },
      }}
    >
      <Box sx={{ p: 1.5, display: 'flex', alignItems: 'center', gap: 1.25 }}>
        <Box sx={{ width: 24, height: 24, borderRadius: 1.25, bgcolor: 'primary.main' }} />
        <Box>
          <Typography sx={{ fontSize: 15, fontWeight: 800, lineHeight: 1.2 }}>
            RobotControl
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
            v1.0.0
          </Typography>
        </Box>
      </Box>

      <Box sx={{ px: 1 }}>
        <List disablePadding>
          {navItems.map((item) => {
            const selected = item.key === selectedKey;

            return (
              <ListItemButton
                key={item.key}
                selected={selected}
                onClick={() => {
                  navigate(item.path);
                }}
                sx={{
                  mx: 1,
                  my: 0.5,
                  gap: 1,
                  py: 1,
                  px: 1.25,
                  borderRadius: 1.75,
                  '&.Mui-selected': {
                    bgcolor: 'primary.main',
                    '&:hover': { bgcolor: 'primary.dark' },
                  },
                }}
              >
                <Box
                  sx={{
                    width: 18,
                    height: 18,
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: selected ? 'rgba(255,255,255,0.20)' : 'rgba(255,255,255,0.10)',
                    color: selected ? 'common.white' : 'text.secondary',
                    flexShrink: 0,
                  }}
                >
                  {item.icon}
                </Box>

                <Typography
                  sx={{
                    fontSize: 13,
                    fontWeight: 800,
                    color: selected ? 'common.white' : 'text.secondary',
                  }}
                >
                  {item.label}
                </Typography>
              </ListItemButton>
            );
          })}
        </List>
      </Box>

      <Box sx={{ mt: 'auto', p: 1.5 }}>
        <Typography sx={{ fontSize: 13, fontWeight: 800 }}>
          User
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
          user@example.com
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
