import type { DHParameter, JointLimit, MiniRobot } from '@/types';

// 机械臂预设模型库
export interface RobotPreset {
  id: string;
  name: string;
  description: string;
  manufacturer: string;
  dhParams: DHParameter[];
  jointLimits: JointLimit[];
  workspace: {
    reach: number; // 最大工作半径 (mm)
    payload: number; // 负载能力 (kg)
  };
}

// 标准6轴工业机械臂 (默认)
const standardIndustrial6Axis: RobotPreset = {
  id: 'standard_6axis',
  name: '标准6轴工业机械臂',
  description: '通用6轴工业机械臂模型',
  manufacturer: '通用',
  dhParams: [
    { joint: 1, a: 0, d: 0, theta: 0, alpha: 90 },
    { joint: 2, a: 420, d: 0, theta: -90, alpha: 0 },
    { joint: 3, a: 400, d: 0, theta: 0, alpha: 90 },
    { joint: 4, a: 0, d: 380, theta: 0, alpha: -90 },
    { joint: 5, a: 0, d: 0, theta: 0, alpha: 90 },
    { joint: 6, a: 0, d: 65, theta: 0, alpha: 0 },
  ],
  jointLimits: [
    { joint: 1, min: -180, max: 180 },
    { joint: 2, min: -90, max: 135 },
    { joint: 3, min: -180, max: 70 },
    { joint: 4, min: -180, max: 180 },
    { joint: 5, min: -120, max: 120 },
    { joint: 6, min: -360, max: 360 },
  ],
  workspace: {
    reach: 1200,
    payload: 10,
  },
};

// ER15-1400 机械臂 (基于URDF文件解析)
const er15_1400: RobotPreset = {
  id: 'er15_1400',
  name: 'ER15-1400',
  description: '15kg负载，1400mm工作半径的工业机械臂',
  manufacturer: 'EFORT',
  dhParams: [
    // 基于URDF文件的关节变换解析得出的DH参数
    { joint: 1, a: 0, d: 430, theta: 0, alpha: 0 },        // base_link -> link_1
    { joint: 2, a: 180, d: 0, theta: -90, alpha: 90 },     // link_1 -> link_2  
    { joint: 3, a: 580, d: 0, theta: 0, alpha: 0 },        // link_2 -> link_3
    { joint: 4, a: 160, d: 640, theta: 0, alpha: -90 },    // link_3 -> link_4
    { joint: 5, a: 0, d: 0, theta: 0, alpha: 90 },         // link_4 -> link_5
    { joint: 6, a: 0, d: 116, theta: 0, alpha: 0 },        // link_5 -> link_6
  ],
  jointLimits: [
    // 从URDF文件中提取的关节限制 (弧度转换为度)
    { joint: 1, min: -170, max: 170 },    // -2.967 ~ 2.967 rad
    { joint: 2, min: -160, max: 90 },     // -2.7925 ~ 1.5708 rad  
    { joint: 3, min: -85, max: 175 },     // -1.4835 ~ 3.0543 rad
    { joint: 4, min: -190, max: 190 },    // -3.316 ~ 3.316 rad
    { joint: 5, min: -130, max: 130 },    // -2.2689 ~ 2.2689 rad
    { joint: 6, min: -360, max: 360 },    // -6.2832 ~ 6.2832 rad
  ],
  workspace: {
    reach: 1400,
    payload: 15,
  },
};

// 机械臂预设库
export const robotPresets: RobotPreset[] = [
  standardIndustrial6Axis,
  er15_1400,
];

// 根据ID获取预设
export const getRobotPreset = (id: string): RobotPreset | undefined => {
  return robotPresets.find(preset => preset.id === id);
};

// 将预设转换为MiniRobot格式
export const presetToRobot = (preset: RobotPreset): MiniRobot => {
  return {
    dhParams: preset.dhParams,
    jointLimits: preset.jointLimits,
  };
};
