import React from 'react';
import {
  AppBar as MuiAppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Box,
  Tooltip,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Button,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  AccountCircle as AccountIcon,
  Cable as ConnectionIcon,
} from '@mui/icons-material';

import { useAppDispatch, useAppSelector } from '../../store';
import { toggleSidebar, toggleTheme } from '../../store/slices/uiSlice';
import ConnectionStatus from '../common/ConnectionStatus';
import ConnectionManager from '../common/ConnectionManager';

const AppBar: React.FC = () => {
  const dispatch = useAppDispatch();
  const { theme, notifications } = useAppSelector((state) => state.ui);
  const { isConnected } = useAppSelector((state) => state.robot);
  
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [notificationAnchor, setNotificationAnchor] = React.useState<null | HTMLElement>(null);
  const [connectionManagerOpen, setConnectionManagerOpen] = React.useState(false);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchor(null);
  };

  const handleToggleSidebar = () => {
    dispatch(toggleSidebar());
  };

  const handleToggleTheme = () => {
    dispatch(toggleTheme());
  };

  return (
    <MuiAppBar
      position="fixed"
      sx={{
        zIndex: 1201, // theme.zIndex.drawer + 1
        backgroundColor: 'background.paper',
        color: 'text.primary',
        boxShadow: 1,
      }}
    >
      <Toolbar>
        {/* Menu Button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="toggle sidebar"
          onClick={handleToggleSidebar}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        {/* Title */}
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          机器人运动控制系统
        </Typography>

        {/* Connection Status */}
        <ConnectionStatus isConnected={isConnected} />

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Connection Manager */}
          <Tooltip title="连接管理">
            <Button
              color="inherit"
              startIcon={<ConnectionIcon />}
              onClick={() => setConnectionManagerOpen(true)}
              sx={{ textTransform: 'none' }}
            >
              {isConnected ? '已连接' : '未连接'}
            </Button>
          </Tooltip>
          {/* Theme Toggle */}
          <Tooltip title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}>
            <IconButton color="inherit" onClick={handleToggleTheme}>
              {theme === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>

          {/* Notifications */}
          <Tooltip title="通知">
            <IconButton color="inherit" onClick={handleNotificationOpen}>
              <Badge badgeContent={unreadCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* Settings */}
          <Tooltip title="设置">
            <IconButton color="inherit">
              <SettingsIcon />
            </IconButton>
          </Tooltip>

          {/* User Menu */}
          <Tooltip title="用户菜单">
            <IconButton color="inherit" onClick={handleMenuOpen}>
              <Avatar sx={{ width: 32, height: 32 }}>
                <AccountIcon />
              </Avatar>
            </IconButton>
          </Tooltip>
        </Box>

        {/* User Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          <MenuItem onClick={handleMenuClose}>个人资料</MenuItem>
          <MenuItem onClick={handleMenuClose}>账户设置</MenuItem>
          <Divider />
          <MenuItem onClick={handleMenuClose}>退出登录</MenuItem>
        </Menu>

        {/* Notification Menu */}
        <Menu
          anchorEl={notificationAnchor}
          open={Boolean(notificationAnchor)}
          onClose={handleNotificationClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          PaperProps={{
            sx: { width: 320, maxHeight: 400 },
          }}
        >
          {notifications.length === 0 ? (
            <MenuItem disabled>
              <Typography variant="body2" color="text.secondary">
                暂无通知
              </Typography>
            </MenuItem>
          ) : (
            notifications.slice(0, 5).map((notification) => (
              <MenuItem key={notification.id} onClick={handleNotificationClose}>
                <Box>
                  <Typography variant="subtitle2" color={notification.read ? 'text.secondary' : 'text.primary'}>
                    {notification.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {notification.message}
                  </Typography>
                </Box>
              </MenuItem>
            ))
          )}
          {notifications.length > 5 && (
            <>
              <Divider />
              <MenuItem onClick={handleNotificationClose}>
                <Typography variant="body2" color="primary">
                  查看全部通知
                </Typography>
              </MenuItem>
            </>
          )}
        </Menu>
      </Toolbar>

      {/* Connection Manager Dialog */}
      <ConnectionManager
        open={connectionManagerOpen}
        onClose={() => setConnectionManagerOpen(false)}
      />
    </MuiAppBar>
  );
};

export default AppBar;