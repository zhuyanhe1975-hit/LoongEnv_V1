import React from 'react';
import {
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  CardHeader,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';

import { useAppDispatch, useAppSelector } from '../../store';
import { setRobotModel } from '../../store/slices/robotSlice';

const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { modelUrdfUrl, modelName } = useAppSelector((state) => state.robot);
  const previousBlobUrlRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    return () => {
      if (previousBlobUrlRef.current) {
        URL.revokeObjectURL(previousBlobUrlRef.current);
        previousBlobUrlRef.current = null;
      }
    };
  }, []);

  const handleSelectBuiltIn = (value: string) => {
    dispatch(setRobotModel({ urdfUrl: value }));
  };

  const handleUploadUrdf = (file: File | null) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const name = file.name.replace(/\.urdf$/i, '');
    if (previousBlobUrlRef.current) {
      URL.revokeObjectURL(previousBlobUrlRef.current);
    }
    previousBlobUrlRef.current = url;
    dispatch(setRobotModel({ urdfUrl: url, name }));
  };

  return (
    <Box>
      {/* Page Header */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="h5" component="h1" gutterBottom sx={{ fontWeight: 800 }}>
          模型加载
        </Typography>
        <Typography variant="body1" color="text.secondary">
          选择或导入机器人模型（URDF）并查看必要信息
        </Typography>
      </Box>

      <Card sx={{ maxWidth: 720 }}>
        <CardHeader title="模型源" subheader="右侧 3D 视图将常驻显示" />
        <CardContent>
          <FormControl fullWidth>
            <InputLabel>内置模型</InputLabel>
            <Select
              label="内置模型"
              value={modelUrdfUrl.startsWith('blob:') ? '/models/ER15-1400.urdf' : modelUrdfUrl}
              onChange={(e) => handleSelectBuiltIn(String(e.target.value))}
            >
              <MenuItem value="/models/ER15-1400.urdf">ER15-1400</MenuItem>
            </Select>
          </FormControl>

          <Divider sx={{ my: 2 }} />

          <Typography variant="body2" color="text.secondary">
            当前模型
          </Typography>
          <Typography sx={{ fontWeight: 800 }}>
            {modelName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {modelUrdfUrl}
          </Typography>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            说明：导入的 URDF 需要其引用的网格文件可被前端访问（例如放在 `ui/public/models/`）。
          </Typography>
        </CardContent>
        <CardActions sx={{ px: 2, pb: 2 }}>
          <Button component="label" variant="outlined" startIcon={<UploadFileIcon />}>
            导入 URDF
            <input
              hidden
              type="file"
              accept=".urdf,application/xml,text/xml"
              onChange={(e) => handleUploadUrdf(e.target.files?.[0] || null)}
            />
          </Button>
        </CardActions>
      </Card>
    </Box>
  );
};

export default Dashboard;
