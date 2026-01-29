import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  Card,
  CardContent,
  CardHeader,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Chip,
  Snackbar,
} from '@mui/material';
import {
  Save,
  Restore,
  Download,
  Upload,
  Security,
  Notifications,
  Palette,
} from '@mui/icons-material';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState({
    theme: 'dark',
    language: 'zh-CN',
    notifications: true,
    autoSave: true,
    debugMode: false,
    safetyMode: true,
    updateInterval: 100,
    logLevel: 'info',
  });

  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'info',
  });

  // 加载设置
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch('http://localhost:5006/api/settings');
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
      showSnackbar('加载设置失败', 'error');
    }
  };

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info' = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleSettingChange = (key: keyof typeof settings) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = event.target.type === 'checkbox' 
      ? event.target.checked 
      : event.target.value;
    
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSelectChange = (key: keyof typeof settings) => (
    event: any
  ) => {
    setSettings(prev => ({
      ...prev,
      [key]: event.target.value
    }));
  };

  const handleSaveSettings = async () => {
    try {
      const response = await fetch('http://localhost:5006/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        showSnackbar('设置已保存', 'success');
      } else {
        showSnackbar('保存设置失败', 'error');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      showSnackbar('保存设置失败', 'error');
    }
  };

  const handleResetSettings = async () => {
    try {
      const response = await fetch('http://localhost:5006/api/settings/reset', {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        setSettings(data.settings);
        showSnackbar('设置已恢复为默认值', 'success');
      } else {
        showSnackbar('恢复默认设置失败', 'error');
      }
    } catch (error) {
      console.error('Failed to reset settings:', error);
      showSnackbar('恢复默认设置失败', 'error');
    }
  };

  const handleExportSettings = async () => {
    try {
      const response = await fetch('http://localhost:5006/api/settings/export');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'robot-settings.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        showSnackbar('配置已导出', 'success');
      } else {
        showSnackbar('导出配置失败', 'error');
      }
    } catch (error) {
      console.error('Failed to export settings:', error);
      showSnackbar('导出配置失败', 'error');
    }
  };

  const handleImportSettings = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e: any) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:5006/api/settings/import', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          setSettings(data.settings);
          showSnackbar('配置已导入', 'success');
        } else {
          showSnackbar('导入配置失败', 'error');
        }
      } catch (error) {
        console.error('Failed to import settings:', error);
        showSnackbar('导入配置失败', 'error');
      }
    };
    input.click();
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        系统设置
      </Typography>

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              avatar={<Palette />}
              title="界面设置"
            />
            <CardContent>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>主题</InputLabel>
                <Select
                  value={settings.theme}
                  label="主题"
                  onChange={handleSelectChange('theme')}
                >
                  <MenuItem value="light">浅色主题</MenuItem>
                  <MenuItem value="dark">深色主题</MenuItem>
                  <MenuItem value="auto">跟随系统</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>语言</InputLabel>
                <Select
                  value={settings.language}
                  label="语言"
                  onChange={handleSelectChange('language')}
                >
                  <MenuItem value="zh-CN">简体中文</MenuItem>
                  <MenuItem value="en-US">English</MenuItem>
                  <MenuItem value="ja-JP">日本語</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="更新间隔 (ms)"
                type="number"
                value={settings.updateInterval}
                onChange={handleSettingChange('updateInterval')}
                sx={{ mb: 2 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={settings.autoSave}
                    onChange={handleSettingChange('autoSave')}
                  />
                }
                label="自动保存"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              avatar={<Notifications />}
              title="通知设置"
            />
            <CardContent>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications}
                    onChange={handleSettingChange('notifications')}
                  />
                }
                label="启用通知"
                sx={{ mb: 2 }}
              />

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>日志级别</InputLabel>
                <Select
                  value={settings.logLevel}
                  label="日志级别"
                  onChange={handleSelectChange('logLevel')}
                >
                  <MenuItem value="debug">调试</MenuItem>
                  <MenuItem value="info">信息</MenuItem>
                  <MenuItem value="warning">警告</MenuItem>
                  <MenuItem value="error">错误</MenuItem>
                </Select>
              </FormControl>

              <Alert severity="info" sx={{ mb: 2 }}>
                通知设置将在下次启动时生效
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* Security Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              avatar={<Security />}
              title="安全设置"
            />
            <CardContent>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.safetyMode}
                    onChange={handleSettingChange('safetyMode')}
                  />
                }
                label="安全模式"
                sx={{ mb: 2 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={settings.debugMode}
                    onChange={handleSettingChange('debugMode')}
                  />
                }
                label="调试模式"
                sx={{ mb: 2 }}
              />

              <Alert severity="warning">
                调试模式会降低系统安全性，仅在开发时使用
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* System Information */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title="系统信息"
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  版本号
                </Typography>
                <Chip label="v1.0.0" size="small" />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  构建时间
                </Typography>
                <Typography variant="body1">
                  2024-01-28 12:00:00
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  运行时间
                </Typography>
                <Typography variant="body1">
                  2小时 35分钟
                </Typography>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="body2" color="text.secondary">
                许可证: MIT License
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Configuration Management */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="配置管理" />
            <CardContent>
            <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                startIcon={<Save />}
                onClick={handleSaveSettings}
              >
                保存设置
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Restore />}
                onClick={handleResetSettings}
              >
                恢复默认
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Download />}
                onClick={handleExportSettings}
              >
                导出配置
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Upload />}
                onClick={handleImportSettings}
              >
                导入配置
              </Button>
            </Box>

            <Alert severity="info">
              配置文件保存在: ~/.robot-control/config.json
            </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
    </Box>
  );
};

export default SettingsPage;
