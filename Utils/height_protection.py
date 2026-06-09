from WVRENV_PHD.basic.ObstacleModel import NeedtoAvoidObstacle, ray_terrain_intersection, get_elevation
from ctypes import *
import os
import sys
import platform
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time

from WVRENV_PHD.utils.GNCData import wgs84ToNED

sensor_pargs = [10, 3000, 1, 10]

def attitude_trace(fc_data, delta_pos):
    '''姿态追踪法子函数，'''

    vel = fc_data.fGroundSpeed
    phi = np.deg2rad(fc_data.fRollAngle)  # 滚转角
    pitch = np.deg2rad(fc_data.fPitchAngle)  # 俯仰角
    yaw = np.deg2rad(fc_data.fYawAngle)  # 偏航角

    q_p_v = math.atan2(1 * -delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))  # 计算制导所需要的视线角，速度角
    q_p_l = math.atan2(delta_pos[1], delta_pos[0])

    a_vp_traj = np.zeros(3)  # 计算速度
    a_vp_traj[0] = 0
    a_vp_traj[1] = 2 * vel * math.sin(q_p_l - yaw)
    a_vp_traj[2] = -2 * vel * math.sin(q_p_v - pitch)

    rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
    R_trajtovel = rtv.as_matrix()  # 弹道系转速度系

    # ----------------------------------这里应该是航迹角而不是姿态角----------------------------
    rev = R.from_euler(
        'ZYX',
        [fc_data.fPathYawAngle, fc_data.fPathPitchAngle, np.rad2deg(phi)],
        degrees=True,
    )
    R_eartovel = rev.as_matrix()  # 惯性系转速度系

    g = np.zeros(3)
    g[2] = 9.8
    a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

    load = -a_vp_vel[2] / g[2]
    phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

    if load > 0:
        load = load / 9
    else:
        load = load / 3

    action = []
    action.append(1)
    action.append(load)
    action.append(phi_dot / 180)
    action.append(0)

    return action

def abs_height_protection(fighter, target, control_input, abs_height = 5700):

    # 高度大于绝对高度时，不进行高度保护   地形最高 5664
    if fighter.fc_data.fAltitude > abs_height:
        return control_input
    # 高度小于绝对高度时，进行姿态追踪
    else:
        delta_pos = [1000 * np.cos(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), 1000 * np.sin(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), - 1000 * np.sin(np.deg2rad(10))]
        return attitude_trace(fighter.fc_data, delta_pos)


def rel_height_protection(fighter, target, control_input,rel_height = 800):

    # _, _, terrain_alt, _ = ray_terrain_intersection(fighter, sensor_pargs[0], sensor_pargs[1], sensor_pargs[2], sensor_pargs[3])

    terrain_alt = get_elevation(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude)
    if terrain_alt == None:
        terrain_alt = 0
    # 高度大于相对高度时，不进行高度保护
    if fighter.fc_data.fAltitude - terrain_alt > rel_height:
        return control_input
    # 高度小于相对高度时，进行姿态追踪
    else:
        delta_pos = [1000 * np.cos(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), 1000 * np.sin(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), - 1000 * np.sin(np.deg2rad(10))]
        return attitude_trace(fighter.fc_data, delta_pos)


class Comb_glorithm():
    def __init__(self):
        self.flag_AO = 0
        self.cycle_AO = 300
        self.flag_control = 0
        self.need_ao = 0.00001


    def reset(self):
        self.flag_AO = 0
        self.cycle_AO = 300
        self.flag_control = 0
        self.need_ao = 0.00001

    def step(self, fighter, target, control_input):

        delta_pos = [1000 * np.cos(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), 1000 * np.sin(np.deg2rad(fighter.fc_data.fPathYawAngle)) * np.cos(np.deg2rad(10)), - 1000 * np.sin(np.deg2rad(10))]
        self.need_ao = NeedtoAvoidObstacle(fighter)
        #未触发避障
        if self.need_ao <= 0.001 and self.flag_AO <= 0.001:
            self.flag_control = 0
        #第一次进入避障
        if self.need_ao > 0.001:
            self.flag_AO = 1
            self.flag_control = 1
        #避障指令减弱
        if self.need_ao <= 0.001 and self.flag_AO >= 0.00001:
            # 如果离地太近，延长渐弱周期
            terrain_alt = get_elevation(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude)
            if fighter.fc_data.fAltitude - terrain_alt < 100:
                self.cycle_AO = self.cycle_AO + 10
            self.flag_AO = max(0.0, self.flag_AO - 10 / self.cycle_AO)
            if self.flag_AO <= 0.0:
                self.flag_control = 0
            else:
                self.flag_control = 2

        #完全响应空战指令
        if self.flag_control == 0:
            return control_input

        #完全响应避障指令
        elif self.flag_control == 1:
            return attitude_trace(fighter.fc_data, delta_pos)
        #融合指令
        elif self.flag_control == 2:
            action_ao = np.array(attitude_trace(fighter.fc_data, delta_pos))
            control_input = np.array(control_input)
            action_mix = self.flag_AO * action_ao + (1 - self.flag_AO) * control_input
            return action_mix.tolist()
        else:
            return control_input


def normalize(vec, eps=1e-6):
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < eps:
        return [0.0, 0.0, 0.0], 0.0
    return [v / norm for v in vec], norm


def collision_avoidance(env, sim_in_list):
    collision_dist = 500.0  # 防撞触发距离
    escape_dist = 300.0  # 防撞时朝目标点拉开的距离

    red_pos = env.world.fighters[0].state.ned_Pos
    blue_pos = env.world.fighters[1].state.ned_Pos

    # red相对blue的位置矢量
    delta_pos = [
        red_pos[0] - blue_pos[0],
        red_pos[1] - blue_pos[1],
        red_pos[2] - blue_pos[2]
    ]

    distance = math.sqrt(
        delta_pos[0] * delta_pos[0] +
        delta_pos[1] * delta_pos[1] +
        delta_pos[2] * delta_pos[2]
    )

    if distance < collision_dist:
        # 1) 主分离方向：沿两机连线反向分开
        rel_dir, _ = normalize(delta_pos)

        # 2) 横向错开方向：在水平面内取一个垂直方向，避免“纯后退”导致再次相遇
        horiz_vec = [delta_pos[0], delta_pos[1], 0.0]
        horiz_dir, horiz_norm = normalize(horiz_vec)

        if horiz_norm < 1e-6:
            # 如果几乎在同一垂直线上，给一个默认横向方向
            lateral_dir = [1.0, 0.0, 0.0]
        else:
            # 水平面内左法向量
            lateral_dir = [-horiz_dir[1], horiz_dir[0], 0.0]
    red_escape_dir = [
        0.80 * rel_dir[0] + 0.50 * lateral_dir[0],
        0.80 * rel_dir[1] + 0.50 * lateral_dir[1],
        0.30 * rel_dir[2] - 0.25
    ]

    blue_escape_dir = [
        -0.80 * rel_dir[0] - 0.50 * lateral_dir[0],
        -0.80 * rel_dir[1] - 0.50 * lateral_dir[1],
        -0.30 * rel_dir[2] + 0.25
    ]
    red_escape_dir, _ = normalize(red_escape_dir)
    blue_escape_dir, _ = normalize(blue_escape_dir)

    # 4) 构造供 attitude_trace 使用的目标矢量
    red_target_vec = [
        red_escape_dir[0] * escape_dist,
        red_escape_dir[1] * escape_dist,
        red_escape_dir[2] * escape_dist
    ]

    blue_target_vec = [
        blue_escape_dir[0] * escape_dist,
        blue_escape_dir[1] * escape_dist,
        blue_escape_dir[2] * escape_dist
    ]

    red_control = attitude_trace(env.world.fighters[0].fc_data, red_target_vec)
    blue_control = attitude_trace(env.world.fighters[1].fc_data, blue_target_vec)

    sim_in_list[0].control_input = red_control
    sim_in_list[1].control_input = blue_control

    return sim_in_list

















