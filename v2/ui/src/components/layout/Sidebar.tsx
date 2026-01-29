import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
  Divider,
  Collapse,
  Tooltip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Visibility as MonitorIcon,
  Route as RouteIcon,
  Tune as TuneIcon,
  Settings as SettingsIcon,
  SmartToy as RobotIcon,
  ExpandLess,
  ExpandMore,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';

import { useAppDispatch, useAppSelector } from '../../store';
import { setCurrentPage } from '../../store/slices/uiSlice';

interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactElement;
  path: string;
  children?: MenuItem[];
}

const menuItems: MenuItem[] = [
  {
    id: 'dashboard',
    label: '仪表板',
    icon: <DashboardIcon />,
    path: '/',
  },
  {
    id: 'monitoring',
    label: '实时监控',
    icon: <MonitorIcon />,
    path: '/monitoring',
    children: [
      {
        id: 'robot-status',
        label: '机器人状态',
        icon: <RobotIcon />,
        path: '/monitoring/robot',
      },
      {
        id: 'performance',
        label: '性能监控',
        icon: <SpeedIcon />,
        path: '/monitoring/performance',
      },
      {
        id: 'safety',
        label: '安全监控',
        icon: <SecurityIcon />,
        path: '/monitoring/safety',
      },
    ],
  },
  {
    id: 'planning',
    label: '轨迹规划',
    icon: <RouteIcon />,
    path: '/planning',
    children: [
      {
        id: 'trajectory-editor',
        label: '轨迹编辑器',
        icon: <TimelineIcon />,
        path: '/planning/editor',
      },
      {
        id: 'trajectory-library',
        label: '轨迹库',
        icon: <RouteIcon />,
        path: '/planning/library',
      },
    ],
  },
  {
    id: 'tuning',
    label: '参数调优',
    icon: <TuneIcon />,
    path: '/tuning',
  },
  {
    id: 'settings',
    label: '系统设置',
    icon: <SettingsIcon />,
    path: '/settings',
  },
];

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useAppDispatch();
  const sidebarOpen = useAppSelector((state) => state.ui.sidebarOpen);
  
  const [expandedItems, setExpandedItems] = React.useState<string[]>(['monitoring', 'planning']);

  const handleItemClick = (item: MenuItem) => {
    if (item.children) {
      // Toggle expansion for items with children
      setExpandedItems(prev => 
        prev.includes(item.id) 
          ? prev.filter(id => id !== item.id)
          : [...prev, item.id]
      );
    } else {
      // Navigate to the page
      navigate(item.path);
      dispatch(setCurrentPage(item.id));
    }
  };

  const isSelected = (path: string) => {
    return location.pathname === path;
  };

  const renderMenuItem = (item: MenuItem, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.id);
    const selected = isSelected(item.path);

    return (
      <React.Fragment key={item.id}>
        <Tooltip title={!sidebarOpen ? item.label : ''} placement="right">
          <ListItemButton
            selected={selected}
            onClick={() => handleItemClick(item)}
            sx={{
              pl: 2 + level * 2,
              minHeight: 48,
              borderRadius: 1,
              mx: 1,
              mb: 0.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: sidebarOpen ? 40 : 'auto',
                color: selected ? 'primary.main' : 'inherit',
              }}
            >
              {item.icon}
            </ListItemIcon>
            {sidebarOpen && (
              <>
                <ListItemText
                  primary={item.label}
                  sx={{
                    color: selected ? 'primary.main' : 'inherit',
                  }}
                />
                {hasChildren && (
                  isExpanded ? <ExpandLess /> : <ExpandMore />
                )}
              </>
            )}
          </ListItemButton>
        </Tooltip>
        
        {hasChildren && sidebarOpen && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => renderMenuItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: sidebarOpen ? 280 : 80,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: sidebarOpen ? 280 : 80,
          boxSizing: 'border-box',
          transition: (theme) =>
            theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          overflowX: 'hidden',
          mt: '64px', // AppBar height
        },
      }}
    >
      {/* Logo/Brand Section */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: sidebarOpen ? 'flex-start' : 'center',
        }}
      >
        <RobotIcon sx={{ color: 'primary.main', fontSize: 32 }} />
        {sidebarOpen && (
          <Box sx={{ ml: 2 }}>
            <Typography variant="h6" color="primary" fontWeight="bold">
              RobotControl
            </Typography>
            <Typography variant="caption" color="text.secondary">
              v1.0.0
            </Typography>
          </Box>
        )}
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <Box sx={{ flexGrow: 1, py: 1 }}>
        <List>
          {menuItems.map(item => renderMenuItem(item))}
        </List>
      </Box>

      {/* Footer */}
      {sidebarOpen && (
        <>
          <Divider />
          <Box sx={{ p: 2 }}>
            <Typography variant="caption" color="text.secondary" align="center" display="block">
              © 2024 Robot Control System
            </Typography>
          </Box>
        </>
      )}
    </Drawer>
  );
};

export default Sidebar;