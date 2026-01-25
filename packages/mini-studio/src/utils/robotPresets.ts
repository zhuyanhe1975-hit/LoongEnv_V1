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

// ER15-1400 机械臂 (基于ER15-1400-fulldyn-local.urdf文件解析)
const er15_1400: RobotPreset = {
  id: 'er15_1400',
  name: 'ER15-1400',
  description: '15kg负载，1400mm工作半径的工业机械臂',
  manufacturer: 'EFORT',
  dhParams: [
    // 基于URDF关节变换的精确DH参数
    { joint: 1, a: 0, d: 430, theta: 0, alpha: 0 },        // joint_1: xyz="0 0 0.43"
    { joint: 2, a: 180, d: 0, theta: -90, alpha: 90 },     // joint_2: xyz="0.18 0 0" rpy="1.57 -1.57 0"
    { joint: 3, a: 580, d: 0, theta: 0, alpha: 0 },        // joint_3: xyz="0.58 0 0"
    { joint: 4, a: 160, d: 640, theta: 0, alpha: -90 },    // joint_4: xyz="0.16 -0.64 0" rpy="-1.57 0 3.14"
    { joint: 5, a: 0, d: 0, theta: 0, alpha: 90 },         // joint_5: xyz="0 0 0" rpy="-1.57 0 3.14"
    { joint: 6, a: 0, d: 116, theta: 0, alpha: 0 },        // joint_6: xyz="0 -0.116 0" rpy="1.57 0 0"
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

// 机械臂预设库 (仅包含ER15-1400)
export const robotPresets: RobotPreset[] = [
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
