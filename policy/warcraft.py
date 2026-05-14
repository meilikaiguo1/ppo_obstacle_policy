import numpy as np
import math

from scipy.spatial.transform import Rotation as R
from WVRENV_PHD.utils.GNCData import wgs84ToNED, euler2vector

lat0 = 32.8
lon0 = 96.3
h0 = 0


class Warcraft(object):
    """建立一个关于状态机的类"""
    def __init__(self):
        """设置初始变量"""
        self.count = 0                  # 单机步长计数器
        self.count_switch = 0           # 能量过低锁死计数用的计数器

        # 储存动作选择的list
        self.mode = []
        self.mode.append(0)             # 随便输入一个数进去防止为空

        self.mode1 = 0                  # 记录当前的动作模式选择
        self.switch = False             # 能量过低锁死使用的tag
        self.switch_max = 500

        # 态势得分部分
        self.mul_score_data = []
        self.target_index_data = []
        self.max_score = 0

        # 用于执行动作时的变量参数
        self.gamma_des = 15
        self.phi_des = 0
        self.load_des = 1
        self.phi_dot_des = 0
        self.ATA = 0

        self.tag0 = False               # 当一个动作有多个阶段，使用这两个tag来标记阶段完成进度
        self.tag1 = False

    def aircraft(self, fighter, target, control_mode=0, control_p_n=0):
        """总控制函数"""
        if not self.switch:
            if control_mode == 0:
                self.state_machine(fighter, target)
            if control_mode == 1:
                self.random()
            if control_mode == 2:
                self.mode1 = control_p_n
        elif self.switch:
            self.mode1 = self.mode[-1]
            self.count_switch += 1

        action = self.choose_action(fighter, target)

        # 检查是否松开锁死按钮
        if self.count_switch >= self.switch_max:
            self.count_switch = 0
            self.switch = False

        # 检查动作是否发生变化
        if self.mode[-1] != self.mode1:
            self.tag0 = False
            self.tag1 = False
        self.mode.append(self.mode1)

        # 输出限幅
        if action[0] >= 12:
            action[0] = 12
        if action[0] <= -5:
            action[0] = -5

        self.count += 1

        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        d = (delta_pos[0] ** 2 + delta_pos[1] ** 2 + delta_pos[2] ** 2) ** 0.5  # 位置差的绝对值
        ATA = np.arccos(np.inner(delta_pos, fighter.state.ned_Vel) / np.linalg.norm(delta_pos) / np.linalg.norm(
            fighter.state.ned_Vel))  # 自身速度矢量与视线矢量的夹角
        ATA = np.rad2deg(ATA)
        self.ATA = ATA

        return action

    def policy_reset(self):

        self.count = 0                  # 单机步长计数器
        self.count_switch = 0           # 能量过低锁死计数用的计数器

        # 储存动作选择的list
        self.mode = []
        self.mode.append(0)             # 随便输入一个数进去防止为空

        self.mode1 = 0                  # 记录当前的动作模式选择
        self.switch = False             # 能量过低锁死使用的tag
        self.switch_max = 500

        # 态势得分部分
        self.mul_score_data = []
        self.target_index_data = []
        self.max_score = 0

        # 用于执行动作时的变量参数
        self.gamma_des = 15
        self.phi_des = 0
        self.load_des = 1
        self.phi_dot_des = 0
        self.ATA = 0

        self.tag0 = False               # 当一个动作有多个阶段，使用这两个tag来标记阶段完成进度
        self.tag1 = False

    def state_machine(self, fighter, target):
        """状态机子函数"""
        # —————————————————————————————————— 决策态势处理 ————————————————————————————————————
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        d = (delta_pos[0] ** 2 + delta_pos[1] ** 2 + delta_pos[2] ** 2) ** 0.5  # 位置差的绝对值
        AOA = np.arccos(np.inner(target.state.ned_Vel, fighter.state.ned_Vel) / np.linalg.norm(
            target.state.ned_Vel) / np.linalg.norm(fighter.state.ned_Vel))  # AOA表示自身与敌机速度的夹角
        AOA = np.rad2deg(AOA)
        ATA = np.arccos(np.inner(delta_pos, fighter.state.ned_Vel) / np.linalg.norm(delta_pos) / np.linalg.norm(
            fighter.state.ned_Vel))  # 自身速度矢量与视线矢量的夹角
        ATA = np.rad2deg(ATA)
        self.ATA = ATA
        # 自己的机体矢量
        vec_body = euler2vector(roll=fighter.fc_data.fRollAngle, pitch=fighter.fc_data.fPitchAngle, yaw=fighter.fc_data.fYawAngle)
        vec_body_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
        vec_body_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))
        ATA = np.arccos(np.inner(delta_pos, vec_body) / np.linalg.norm(delta_pos) / np.linalg.norm(vec_body))  # 自身速度矢量与视线矢量的夹角
        ATA = np.rad2deg(ATA)
        self.ATA = ATA

        AOT = np.arccos(
            np.inner(target.state.ned_Vel, delta_pos) / np.linalg.norm(target.state.ned_Vel) / np.linalg.norm(
                delta_pos))  # AOT表示敌机速度与敌机和自身连线的夹角
        AOT = np.rad2deg(AOT)

        g = 9.8             # 重力加速度，暂时不考虑高度影响

        delta_E = 0.5 * fighter.fc_data.fMass * (
                    fighter.fc_data.fGroundSpeed ** 2) + fighter.fc_data.fMass * g * fighter.fc_data.fAltitude  # 总能量E = 动能0.5 * mas * vel^2 + 势能mass * g * h
        delta_E_Tar = 0.5 * target.fc_data.fMass * (
                    target.fc_data.fGroundSpeed ** 2) + target.fc_data.fMass * g * target.fc_data.fAltitude

        # —————————————————————————————————— 决策态势 ————————————————————————————————————
        if d > 5000:
            self.mode1 = 2  # 1 攻击 速度追踪
        elif d <= 5000:
            self.mode1 = 1  # 2 攻击 复合追踪
            if (d <= 1100) and (ATA < 30):  # 3 攻击 姿态瞄准
                self.mode1 = 3
            elif (d <= 300) and (ATA > 60):  # 4 防御：攻击脱离
                self.gamma_des = 20
                self.mode1 = 4
            elif (d <= 1100) and (abs(ATA) > 120) and (AOT < 30):  # 5 防御 半滚倒转
                self.mode1 = 5
            elif (d < 3000) and (d > 1100) and (delta_E < 0.6 * delta_E_Tar):  # 7 防御：放弃攻击,积攒能量
                self.gamma_des = 0
                self.mode1 = 4
            elif (d <= 800) and (AOA < 25) and (60 < AOT < 120):
                self.mode1 = 7  # 7 进攻小桶滚
                self.switch = True
                self.switch_max = 900
            elif ATA > 60:  # 6 转向 高低YOYO
                self.mode1 = 6
        else:
            self.mode1 = 100

        self.mode1 = 3

        # —————————————————————————————————— 保护机制 ————————————————————————————————————————
        if ((fighter.fc_data.fAltitude < 300) or (fighter.fc_data.fGroundSpeed < 80)) and (self.count > 10):  # 攻击脱离积攒能量
            self.switch = True
            self.switch_max = 500
            self.gamma_des = 20
            self.mode1 = 4

    def random(self,):
        """随机抽取状态子函数"""
        time_gap = 1000
        if self.count % time_gap == 0:
            self.mode1 = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12])
            print('随机动作测试机模式：', self.mode1, '运行步数', self.count)

    def choose_action(self, fighter, target):
        '''将输入的模式转换为动作指令'''
        if self.mode1 == 1:  # 模式1为速度追踪
            action = self.speed_trace(fighter, target)
        elif self.mode1 == 2:  # 模式2为复合追踪
            action = self.multi_trace(fighter, target)  # 复合追踪暂时用速度追踪占位
        elif self.mode1 == 3:  # 模式3为攻击瞄准
            action = self.attitude_trace(fighter, target)
        elif self.mode1 == 4:  # 模式4为攻击脱离，动作为追踪倾角
            action = self.track_gamma(fighter, self.gamma_des)
        elif self.mode1 == 5:  # 模式5为防御，半滚倒转
            action = self.split_S(fighter)
        elif self.mode1 == 6:  # 模式6为高YOYO
            action = self.yoyo_oldversion(fighter, target)
        # elif self.mode1 == 7:  # 模式7为进攻小滚筒
        #     action = self.small_roller(fighter, target)
        elif self.mode1 == 8:  # 模式8为转弯
            self.phi_des_deg = 85
            action = self.turn(fighter, self.phi_des_deg)
        elif self.mode1 == 9:  # 模式9为筋斗
            self.load_des = 7
            self.phi_dot_des = 10
            action = self.somersault(fighter, self.load_des, self.phi_dot_des)
        # elif self.mode1 == 10:  # 模式10为防御大桶滚
        #     action = self.big_roller(fighter, target)
        # elif self.mode1 == 11:    # 模式11为螺旋上升
        #     self.gamma_des = -10
        #     self.phi_des = 75
        #     action = self.spiral(fighter, self.gamma_des, self.phi_des)
        # elif self.mode1 == 12:    # 模式12为旧的high_yoyo
        #     action = self.yoyo_oldversion(fighter, target)
        # elif self.mode1 == 13:    # 模式13为测试
        #     action = self.speedTrace_and_Barrel(fighter, target)
        # elif self.mode1 == 14:
        #     action = self.tail_speed_trace(fighter, target)
        else:  # 若均不是，则保持平飞巡航
            action = [1, 0, 100]

        return action

    def score(self, fighter, target):
        '''态势评价得分的部分'''
        # 动态因素得分
        # 角度得分
        angle_pos = wgs84_2_ned(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude,
                                target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude)  # 计算目标机与追踪机的位置差

        angle_vel = fighter.state.ned_Vel
        angle_pos_abs = (angle_pos[0] ** 2 + angle_pos[1] ** 2 + angle_pos[2] ** 2) ** 0.5
        angle_vel_abs = (angle_vel[0] ** 2 + angle_vel[1] ** 2 + angle_vel[2] ** 2) ** 0.5
        angle = math.acos((angle_pos[0] * angle_vel[0] + angle_pos[1] * angle_vel[1] + angle_pos[2] * angle_vel[2])
                          / (angle_pos_abs * angle_vel_abs))
        if angle >= 0:
            angle_score = math.exp(-angle)
        else:
            angle_score = math.exp(angle)
        # self.angle_score = 0.25 * angle_score
        angle_score = 0.25 * angle_score

        # 速度得分
        fly_vel = fighter.fc_data.fGroundSpeed
        tar_vel = target.fc_data.fGroundSpeed
        if tar_vel == 0:  # 0时刻没有速度，会报错
            tar_vel = 1
            fly_vel = 1
        if fly_vel > 1.5 * tar_vel:
            vel_score = 1
        elif 0.6 * tar_vel <= fly_vel and fly_vel <= 1.5 * tar_vel:
            vel_score = fly_vel / tar_vel - 0.5
        elif fly_vel < 0.6 * tar_vel:
            vel_score = 0.1
        # self.vel_score = 0.25 * vel_score
        vel_score = 0.25 * vel_score

        # 距离得分
        # delta_pos = fighter.ned_Pos - target.ned_Pos
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        delta_pos_abs = (delta_pos[0] ** 2 + delta_pos[1] ** 2 + delta_pos[2] ** 2) ** 0.5
        d_weapon = 1200
        if delta_pos_abs <= d_weapon:
            distance_score = 1
        else:
            distance_score = math.exp(-(delta_pos_abs - d_weapon) / d_weapon)
        # self.distance_score = 0.25 * distance_score
        distance_score = 100 * distance_score

        # 高度得分
        delta_h = fighter.fc_data.fAltitude - target.fc_data.fAltitude
        h0 = 500
        if delta_h >= 0:
            h_score = math.log((delta_h / h0) + 1)
        else:
            h_score = math.exp(delta_h / h0) - 1
        if h_score > 1:
            h_score = 1

        h_score = 0.25 * h_score

        # 动态因素总得分
        dynamic_score = angle_score + vel_score + h_score + distance_score
        # 静态因素得分
        static_score = 0.01
        # 总得分
        mul_score = dynamic_score + static_score

        return mul_score

    def task_allocation(self, fighter, world):
        '''多对多任务分配模块'''
        time_gap = 1000
        if self.count % time_gap == 0:
            self.tag_allocation = True

        if self.tag_allocation == True:             # 当tag为true时进行目标分配
            num_blue = 2
            mul_score_data = []
            # 红色机的目标选择
            if fighter.side == 1:
                if fighter.obj_index == 3:
                    score_1 = self.score(fighter, world.fighters[0])
                    mul_score_data.append(score_1)
                    score_2 = self.score(fighter, world.fighters[1])
                    mul_score_data.append(score_2)
                    max_index = np.argmax(mul_score_data)
                    target = world.fighters[max_index]
                    # 将另一架敌机作为备选目标
                    if target.obj_index == 1:
                        other = world.fighters[1]
                    elif target.obj_index == 2:
                        other = world.fighters[0]
                # 僚机选择主机的备选目标
                elif fighter.obj_index == 4:
                    if world.fighters[2].target_index == 1:
                        target = world.fighters[1]
                        other = world.fighters[0]
                    elif world.fighters[2].target_index == 2:
                        target = world.fighters[0]
                        other = world.fighters[1]
            # 蓝色机的目标选择
            elif fighter.side == 0:
                if fighter.obj_index == 1:
                    score_3 = self.score(fighter, world.fighters[2])
                    mul_score_data.append(score_3)
                    score_4 = self.score(fighter, world.fighters[3])
                    mul_score_data.append(score_4)
                    max_index = np.argmax(mul_score_data)
                    target = world.fighters[num_blue + max_index]
                    # 将另一架敌机作为备选目标
                    if target.obj_index == 3:
                        other = world.fighters[3]
                    elif target.obj_index == 4:
                        other = world.fighters[2]
                # 僚机选择主机的备选目标
                elif fighter.obj_index == 2:
                    if world.fighters[0].target_index == 3:
                        target = world.fighters[3]
                        other = world.fighters[2]
                    elif world.fighters[0].target_index == 4:
                        target = world.fighters[2]
                        other = world.fighters[3]
            # 记忆当前的选择
            self.memorytarget = target
            self.memoryother = other
            self.tag_allocation = False
            # 记录全程的选择与当前的最高分目标，用于后续的画图使用
            self.target_index_data.append(target.obj_index)
        else:
            target = self.memorytarget
            other = self.memoryother

        # 当目标死亡后切换目标
        if target.survive_info == False:
            print('I will re-select my target, old target is dead!!')
            target = self.memoryother
            other = self.memorytarget
            self.memorytarget = target
            self.memoryother = other

        # 把当前目标的编号返回到fighter里，方便友军使用
        fighter.target_index = target.obj_index

        return target

    def speed_trace(self, fighter, target):
        """速度追踪法子函数，"""

        vel = fighter.fc_data.fGroundSpeed
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        # delta_pos = target.ned_Pos - fighter.ned_Pos  # 计算目标机与追踪机的位置差
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        q_p_v = math.atan2(1 * - delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))  # 计算制导所需要的视线角，速度角
        q_p_l = math.atan2(delta_pos[1], delta_pos[0])
        q_v_v = math.atan2(-fighter.state.ned_Vel[2],
                           ((fighter.state.ned_Vel[0] ** 2 + fighter.state.ned_Vel[1] ** 2) ** 0.5))
        q_v_l = math.atan2(fighter.state.ned_Vel[1], fighter.state.ned_Vel[0])

        a_vp_traj = np.zeros(3)  # 计算速度
        a_vp_traj[0] = 0
        a_vp_traj[1] = 3 * vel * math.sin(q_p_l - q_v_l)
        a_vp_traj[2] = -3 * vel * math.sin(q_p_v - q_v_v)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系

        chi = np.deg2rad(fighter.fc_data.fPathYawAngle)
        rev = R.from_euler('ZYX', [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)], degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def multi_trace(self, fighter, target):
        """复合追踪法子函数，为虚拟点追踪与比例导引法结合使用"""

        vel = fighter.fc_data.fGroundSpeed
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角

        # delta_pos = target.ned_Pos - fighter.ned_Pos   # 计算目标机与追踪机的位置差
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        delta_vel = target.state.ned_Vel - fighter.state.ned_Vel

        q_p_v = math.atan2(1 * - delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))
        q_p_l = math.atan2(delta_pos[1], delta_pos[0])
        q_v_v = math.atan2(-fighter.state.ned_Vel[2], ((fighter.state.ned_Vel[0] ** 2 + fighter.state.ned_Vel[1] ** 2) ** 0.5))
        q_v_l = math.atan2(fighter.state.ned_Vel[1], fighter.state.ned_Vel[0])

        a_vp_traj = np.zeros(3)
        a_vp_traj[0] = 0
        a_vp_traj[1] = 1 * vel * math.sin(q_p_l - q_v_l)
        a_vp_traj[2] = -1 * vel * math.sin(q_p_v - q_v_v)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系
        R_trajtovel = R_trajtovel.T

        rev = R.from_euler('ZYX', [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)],
                           degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系
        R_eartovel = R_eartovel.T

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel, a_vp_traj) - np.matmul(R_eartovel, g)  # 求解速度系下的过载

        q_los = np.zeros(3)
        q_los[0] = 0
        q_los[1] = math.atan2(1 * delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))
        q_los[2] = math.atan2(delta_pos[1], delta_pos[0])

        dq_los = np.zeros(3)
        dq_los[0] = 0
        dq_los[1] = (delta_pos[2] * (delta_pos[0] * delta_vel[0] + delta_pos[1] * delta_vel[1]) - delta_vel[2] *(delta_pos[0] * delta_pos[0] + delta_pos[1] * delta_pos[1])) / (delta_pos[0] * delta_pos[0] +delta_pos[1] * delta_pos[1] +delta_pos[2] * delta_pos[2]) / ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5)
        dq_los[2] = np.cos(q_los[1]) * (delta_pos[0] * delta_vel[1] - delta_pos[1] * delta_vel[0]) / (delta_pos[0] * delta_pos[0]+ delta_pos[1] * delta_pos[1])

        rel = R.from_euler('ZYX', [np.rad2deg(q_los[2]), np.rad2deg(q_los[1]), np.rad2deg(q_los[0])], degrees=True)
        R_eartolos = rel.as_matrix()  # 惯性系转视线系
        R_eartolos = R_eartolos.T
        R_lostoear = np.linalg.inv(R_eartolos)

        Vr_los = np.matmul(R_eartolos, delta_vel)
        a_pn_los = 4 * np.cross(Vr_los, dq_los)
        a_pn_ear = np.matmul(R_lostoear, a_pn_los)
        a_pn_ear = a_pn_ear - g
        a_pn_vel = np.matmul(R_eartovel, a_pn_ear)

        k_guidance = 0.5  # 复合制导的系数（0-1取值，0为速度追踪法，1为比例导引法）
        a_final = (1 - k_guidance) * a_vp_vel + k_guidance * a_pn_vel

        load = -a_final[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_final[1], -a_final[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def attitude_trace(self, fighter, target):
        """姿态追踪法子函数，"""

        vel = fighter.fc_data.fGroundSpeed
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        yaw = np.deg2rad(fighter.fc_data.fYawAngle)  # 偏航角

        # delta_pos = target.ned_Pos - fighter.ned_Pos  # 计算目标机与追踪机的位置差
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        q_p_v = math.atan2(1 * -delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))  # 计算制导所需要的视线角，速度角
        q_p_l = math.atan2(delta_pos[1], delta_pos[0])

        a_vp_traj = np.zeros(3)  # 计算速度
        a_vp_traj[0] = 0
        a_vp_traj[1] = 2 * vel * math.sin(q_p_l - yaw)
        a_vp_traj[2] = -2 * vel * math.sin(q_p_v - pitch)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系

        chi = np.deg2rad(fighter.fc_data.fPathYawAngle)
        rev = R.from_euler('ZYX',
                           [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)],
                           degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def track_gamma(self, fighter, gamma_des=20):
        """跟踪指定航迹倾角,采用较简单的初始方法"""
        vel = fighter.fc_data.fGroundSpeed
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        g = 9.8

        phi_des = 0
        phi_dot = 6 * np.rad2deg(phi_des - phi)
        gamma_des = np.deg2rad(gamma_des)
        load = 1 * vel * np.sin(gamma_des - gamma) / 9.81 + np.cos(gamma)

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def split_S(self, fighter):
        """半滚倒转"""
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角

        a_vp_traj = np.zeros(3)  # 计算速度
        a_vp_traj[0] = 0
        a_vp_traj[1] = 1
        a_vp_traj[2] = 10 * 9.81

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系

        rev = R.from_euler('ZYX', [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)], degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        if phi_dot > 300:
            phi_dot = 300
        elif phi_dot < -300:
            phi_dot = -300

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def yoyo(self, fighter, target):
        '''高/低yoyo动作'''
        vel = fighter.fc_data.fGroundSpeed  # 地速
        vel_air = fighter.fc_data.fIndicatedAirSpeed  # 指示空速
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        g = 9.8
        point_speed = 0.75 * 340  # 拐点速度

        # point = target.ned_Pos.copy()  # 复制目标机的坐标
        delta_vel = vel - point_speed  # 实际表速与目标速度之差
        # delta_pos = point - fighter.ned_Pos
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        if delta_vel >= 0:  # 计算需要多少的倾角
            gamma_des = np.deg2rad(0.6 * delta_vel)
        else:
            gamma_des = np.deg2rad(0.3 * delta_vel)
        if gamma_des <= -30:  # 对倾角进行一个幅的限
            gamma_des = -30

        if (abs(vel - (point_speed)) < 20):
            self.tag0 = True
        if self.tag0 == True:
            q_p_v = math.atan2(delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))
        else:
            q_p_v = gamma_des

        # q_p_v = math.atan2(-delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))     #
        # q_p_v = np.deg2rad(30)
        q_p_l = math.atan2(delta_pos[1], delta_pos[0])
        q_v_v = math.atan2(-fighter.state.ned_Vel[2],
                           ((fighter.state.ned_Vel[0] ** 2 + fighter.state.ned_Vel[1] ** 2) ** 0.5))
        q_v_l = math.atan2(fighter.state.ned_Vel[1], fighter.state.ned_Vel[0])

        a_vp_traj = np.zeros(3)
        a_vp_traj[0] = 0
        a_vp_traj[1] = 2 * vel * math.sin(q_p_l - q_v_l)
        a_vp_traj[2] = 2 * -vel * math.sin(q_p_v - q_v_v)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系
        rev = R.from_euler('ZYX',
                           [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)],
                           degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def yoyo_oldversion(self, fighter, target):
        """高yoyo动作"""
        vel = fighter.fc_data.fGroundSpeed
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        g = 9.8
        point_speed = 0.75 * 340

        delta_pos0 = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        delta_pos0_abs = (delta_pos0[0] ** 2 + delta_pos0[1] ** 2 + delta_pos0[2] ** 2) ** 0.5

        point = [target.fc_data.fLongitude, target.fc_data.fLatitude,
                                target.fc_data.fAltitude]
        Hc = 0.1 * (vel ** 2) / (2 * g)  # 计算期望拉起的高度Hc

        if vel > point_speed:
            point[2] = point[2] + Hc  # 计算修正过后的虚拟点的坐标，若高于拐点速度则为高yoyo（减速），反之则为低YOYO
        else:
            point[2] = point[2] - Hc

        if (abs(fighter.fc_data.fIndicatedAirSpeed - (point_speed)) < 35):
            self.tag0 = True

        if self.tag0 == True:
            point = [target.fc_data.fLongitude, target.fc_data.fLatitude,
                     target.fc_data.fAltitude]
        else:
            pass

        # delta_pos = point - fighter.ned_Pos
        delta_pos = wgs84_2_ned(point[1], point[0], point[2],
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude)  # 计算目标机与追踪机的位置差

        q_p_v = math.atan2(-delta_pos[2], ((delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5))
        q_p_l = math.atan2(delta_pos[1], delta_pos[0])
        q_v_v = math.atan2(-fighter.state.ned_Vel[2],
                           ((fighter.state.ned_Vel[0] ** 2 + fighter.state.ned_Vel[1] ** 2) ** 0.5))
        q_v_l = math.atan2(fighter.state.ned_Vel[1], fighter.state.ned_Vel[0])

        a_vp_traj = np.zeros(3)
        a_vp_traj[0] = 0
        a_vp_traj[1] = 2 * vel * math.sin(q_p_l - q_v_l)
        a_vp_traj[2] = 2 * -vel * math.sin(q_p_v - q_v_v)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系
        rev = R.from_euler('ZYX',
                           [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)],
                           degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def turn(self, fighter, phi_des_deg):
        '''转弯'''
        vel = fighter.fc_data.fGroundSpeed
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)               # 航迹倾角
        phi = np.deg2rad(fighter.fc_data.fRollAngle)     # 滚转角

        gamma_des = 0
        phi_des = phi_des_deg / 180 * np.pi
        phi_dot = 8 * np.rad2deg(phi_des - phi)
        load = 3 * vel * np.sin(gamma_des - gamma) / 9.81 + np.cos(gamma)

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def somersault(self, fighter, load_des, phi_des):
        '''筋斗'''
        phi = np.deg2rad(fighter.fc_data.fRollAngle)     # 滚转角
        load = load_des
        phi_des = phi_des / 180 * np.pi

        phi_dot = 8 * np.rad2deg(phi_des - phi)

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def spiral(self, fighter, gamma_des, phi_des):
        '''螺旋上升'''
        vel = fighter.fc_data.fGroundSpeed
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        g = 9.8

        gamma_des = np.deg2rad(gamma_des)
        phi_des = np.deg2rad(phi_des)

        phi_dot = 8 * np.rad2deg(phi_des - phi)
        load = 1 * vel * np.sin(gamma_des - gamma) / g + np.cos(gamma)

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action

    def energy(self, fighter, target):
        '''1000m内的积攒能量动作'''
        vel = fighter.fc_data.fGroundSpeed  # 地速
        vel_air = fighter.fc_data.fIndicatedAirSpeed  # 指示空速
        phi = np.deg2rad(fighter.fc_data.fRollAngle)  # 滚转角
        gamma = np.deg2rad(fighter.fc_data.fPathPitchAngle)  # 航迹倾角
        pitch = np.deg2rad(fighter.fc_data.fPitchAngle)  # 俯仰角
        g = 9.8
        point_speed = 0.75 * 340  # 拐点速度

        # point = target.ned_Pos.copy()  # 复制目标机的坐标
        # delta_vel = vel - point_speed  # 实际表速与目标速度之差
        # delta_pos = point - fighter.ned_Pos
        delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                target.fc_data.fAltitude,
                                fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                fighter.fc_data.fAltitude
                                )  # 计算目标机与追踪机的位置差

        gamma_des = np.deg2rad(20)
        q_p_v = gamma_des

        q_p_l = math.atan2(delta_pos[1], delta_pos[0])
        q_v_v = math.atan2(-fighter.state.ned_Vel[2],
                           ((fighter.state.ned_Vel[0] ** 2 + fighter.state.ned_Vel[1] ** 2) ** 0.5))
        q_v_l = math.atan2(fighter.state.ned_Vel[1], fighter.state.ned_Vel[0])

        a_vp_traj = np.zeros(3)
        a_vp_traj[0] = 0
        a_vp_traj[1] = 1 * vel * math.sin(q_p_l - q_v_l)
        a_vp_traj[2] = 1 * -vel * math.sin(q_p_v - q_v_v)

        rtv = R.from_euler('X', np.rad2deg(phi), degrees=True)
        R_trajtovel = rtv.as_matrix()  # 弹道系转速度系
        rev = R.from_euler('ZYX',
                           [fighter.fc_data.fPathYawAngle, fighter.fc_data.fPathPitchAngle, np.rad2deg(phi)],
                           degrees=True)
        R_eartovel = rev.as_matrix()  # 惯性系转速度系

        g = np.zeros(3)
        g[2] = 9.8
        a_vp_vel = np.matmul(R_trajtovel.T, a_vp_traj) - np.matmul(R_eartovel.T, g)  # 求解速度系下的过载

        load = -a_vp_vel[2] / g[2]
        phi_dot = np.rad2deg(math.atan2(a_vp_vel[1], -a_vp_vel[2]))

        action = []
        action.append(load)
        action.append(phi_dot)
        action.append(100)

        return action



def wgs84_2_ecef(lat, lon, h):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)
    lamb = np.deg2rad(lat)
    phi = np.deg2rad(lon)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)
    x = (h + N) * np.cos(lamb) * np.cos(phi)
    y = (h + N) * np.cos(lamb) * np.sin(phi)
    z = (h + (1 - e_sq) * N) * np.sin(lamb)
    r = np.zeros(3)
    r[0] = x
    r[1] = y
    r[2] = z
    return r


def ecef_2_ned(x, y, z, lat, lng, height):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)
    lamb = np.deg2rad(lat)
    phi = np.deg2rad(lng)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    x0 = (height + N) * np.cos(lamb) * np.cos(phi)
    y0 = (height + N) * np.cos(lamb) * np.sin(phi)
    z0 = (height + (1 - e_sq) * N) * np.sin(lamb)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    t = -np.cos(phi) * xd - np.sin(phi) * yd
    East = -np.sin(phi) * xd + np.cos(phi) * yd
    North = t * np.sin(lamb) + np.cos(lamb) * zd
    Up = np.cos(lamb) * np.cos(phi) * xd + np.cos(lamb) * np.sin(phi) * yd + np.sin(lamb) * zd
    r = np.zeros(3)
    r[0] = North
    r[1] = East
    r[2] = -Up
    return r


def wgs84_2_ned(lat, lon, h, lat0, lon0, h0):
    r_ecef = wgs84_2_ecef(lat, lon, h)
    ned_pos = ecef_2_ned(r_ecef[0], r_ecef[1], r_ecef[2], lat0, lon0, h0)
    return ned_pos
