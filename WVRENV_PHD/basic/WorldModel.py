import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import math
import copy
from ctypes import *
from WVRENV_PHD.basic.DataResolve import OutMsg, MultiInitMsg, MultiOutMsg, MultiErrorMsg, CombatMsg, \
    fighters_max_num, ControlMode, AircraftState, MissileInitial, MissileDataIn, MissileDataOut, ControlAction
from WVRENV_PHD.utils.GNCData import wgs84ToNED, ned_to_wgs84, euler2vector, angle_2vec
from WVRENV_PHD.SimArg import missile_str, fighter_str

Control_Mode_Dict = {}
Control_Mode_Dict[ControlMode(0)] = 0
Control_Mode_Dict[ControlMode(1)] = 1
Control_Mode_Dict[ControlMode(2)] = 2
Control_Mode_Dict[ControlMode(3)] = 3
Control_Mode_Dict[ControlMode(4)] = 4

# 导弹的DLL调用
# 加载近距导弹模型
Missile_lib = cdll.LoadLibrary(missile_str)
# 对Dll内的函数进行声明
Missile_lib.create_ptr.restype = c_void_p
Missile_lib.Initial.argtypes = [c_void_p, MissileInitial]
Missile_lib.Update.argtypes = [c_void_p, MissileDataIn, MissileDataOut]
Missile_lib.Update.restype = MissileDataOut


class World(object):
    """与仿真对抗的世界环境相关的类"""
    def __init__(self, initial_data):
        self.train_flag = initial_data.train_flag
        # 世界中的战斗机相关属性
        self.fighters = []
        self.num_blue = initial_data.num_blue
        self.num_red = initial_data.num_red
        self.missile_without_radar = initial_data.missile_without_radar

        # 世界属性
        self.dt = initial_data.dt  # 仿真终端步长
        self.dim_p = 3  # 仿真维度
        # 仿真初始位置
        self.originLatitude = 0
        self.originLongitude = 0
        self.originAltitude = 0

        # 战斗机机间通信
        self.comm_delay = initial_data.comm_delay
        self.red_fighters_comm_buffer = [[] for _ in range(self.num_red)]
        self.blue_fighters_comm_buffer = [[] for _ in range(self.num_blue)]

        # 所有战斗机的一次性初始化
        self.fighters_modes = np.zeros(shape=(4, fighters_max_num), dtype=int)
        self.all_init = MultiInitMsg()  # 这是一个c类型的结构体，里面的每个量为单架飞机的所有初始量
        self.init_list = []
        for k in range(fighters_max_num):
            self.init_list.append(eval('self.all_init.initData_' + str(
                k)))  # 用eval将字符串对应的名字的变量转换成该变量对应的值，则将C类型的结构体self.all_init的内容传递给了self.init_list
        # 所有战斗机可以输出的数据，与初始化类似
        self.all_out = MultiOutMsg()
        self.out_list = []
        for k in range(fighters_max_num):
            self.out_list.append(eval('self.all_out.outData_' + str(k)))
        # 所有战斗机报错的返回数据
        self.all_err = MultiErrorMsg()
        self.errs_list = []
        for k in range(fighters_max_num):
            self.errs_list.append(eval('self.all_err.outerr_' + str(k)))
        # 所有战斗机的存活信息
        self.survive_list = np.zeros(fighters_max_num, bool)
        for j in range(fighters_max_num):
            self.survive_list[j] = True

        # 所有战斗机的输入控制
        self.action_u_0 = np.ones(fighters_max_num)
        self.action_u_1 = np.zeros(fighters_max_num)
        self.action_u_2 = np.zeros(fighters_max_num)
        self.action_u_3 = np.zeros(fighters_max_num)
        # actions of all fighters in world
        self.action_u = np.zeros(shape=(4, fighters_max_num), dtype=float)

        # 创建仿真模型指针
        self.lib = cdll.LoadLibrary(fighter_str)
        # self.lib.MultiInit.argtypes = (c_int, MultiInitMsg)  # argtypes用于告诉Python解释器外来函数的输入参数的类型,
        self.lib.MultiInit.argtypes = (c_int, c_void_p)
        self.lib.MultiOutput.argtypes = (c_int, c_void_p)
        self.lib.UpdateMode0.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_bool * fighters_max_num, c_double * fighters_max_num,
                                    c_double * fighters_max_num, c_double * fighters_max_num, c_double * fighters_max_num)
        self.lib.UpdateMode1.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_bool * fighters_max_num, c_double * fighters_max_num,
                                    c_double * fighters_max_num, c_double * fighters_max_num, c_bool)
        self.lib.UpdateMode2.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_bool * fighters_max_num, c_double * fighters_max_num,
                                    c_double * fighters_max_num, c_double * fighters_max_num, c_double * fighters_max_num,
                                    c_bool)
        self.lib.UpdateMode3.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_bool * fighters_max_num, c_double * fighters_max_num,
                                    c_double * fighters_max_num, c_bool, c_double * fighters_max_num,
                                    c_double * fighters_max_num, c_bool)

        self.lib.GunLockWho.argtypes = (c_int, c_int, c_int, MultiOutMsg, c_bool * fighters_max_num)
        self.lib.angle.argtypes = (c_double, c_double, c_double, c_double, c_double, c_double)
        self.lib.angle.restype = c_double  # restype告诉Python解释器外来函数的返回参数的类型

        self.lib.Update.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_bool * fighters_max_num, c_int * fighters_max_num,
                               c_int * fighters_max_num, c_int * fighters_max_num, c_int * fighters_max_num,
                               c_double * fighters_max_num, c_double * fighters_max_num, c_double * fighters_max_num,
                               c_double * fighters_max_num)

        self.lib.make_fighter()

    def initialize(self):
        """变量初始化"""
        self.lib.MultiInit(len(self.fighters), byref(self.all_init))
        self.lib.MultiOutput(len(self.fighters), byref(self.all_out))  # byref 返回（x）的地址

        # 各种相关的变量初始化
        for i, fighter in enumerate(self.fighters):
            self.get_key_state()  # 获取state信息
            fighter.fc_data = self.out_list[i]  # 从DLL里获取的数据信息
            fighter.combat_data = CombatMsg()  # 战斗相关属性
            fighter.action = ControlAction()
            self.survive_list[i] = fighter.combat_data.survive_info
        self.all_err = MultiErrorMsg()
        self.errs_list = []
        for k in range(fighters_max_num):
            self.errs_list.append(eval('self.all_err.outerr_' + str(k)))

        # 把每架飞机复活
        for fighter in self.fighters:
            fighter.combat_data.survive_info = True
            # 红外信号重置
            fighter.combat_data.infrared_flag = True
            # 坠毁时间重重
            fighter.combat_data.death_time = 0
            # 把飞机的导弹发射计数器归零
            fighter.missiles_count = 0

        # 初始化/重置所有导弹
        # self.missile_reset()
        for i, fighter in enumerate(self.fighters):
            # 初始化/重置近距弹
            for j, missile in enumerate(fighter.missiles):
                missile.missile_reset()

        # 重置所有雷达
        for fighter in self.fighters:
            fighter.sensors.sensor_reset()

        # 重置通信
        self.red_fighters_comm_buffer = [[] for _ in range(self.num_red)]
        self.blue_fighters_comm_buffer = [[] for _ in range(self.num_blue)]
        for fighter in self.fighters:
            fighter.comunication_send = None
            fighter.comunication_delayed_send = None
            if fighter.side == 0:
                fighter.comunication_recv = [None for _ in range(self.num_red - 1)]
            else:
                fighter.comunication_recv = [None for _ in range(self.num_blue - 1)]
        self.get_key_state()

    def step(self):
        # ################## 回合更新 ###################
        # 机体运动学更新
        self.update_fighter()

        # 更新战斗机通信链路
        for i, fighter in enumerate(self.fighters[0: self.num_red]):
            if fighter.combat_data.survive_info:
                self.red_fighters_comm_buffer[i].append(copy.deepcopy(fighter.comunication_send))
            else:
                self.red_fighters_comm_buffer[i].append(None)
        for i, fighter in enumerate(self.fighters[self.num_red:]):
            if fighter.combat_data.survive_info:
                self.blue_fighters_comm_buffer[i].append(copy.deepcopy(fighter.comunication_send))
            else:
                self.blue_fighters_comm_buffer[i].append(None)

        for i, fighter in enumerate(self.fighters[0: self.num_red]):
            if len(self.red_fighters_comm_buffer[i]) == self.comm_delay:
                fighter.comunication_delayed_send = self.red_fighters_comm_buffer[i][0]
                self.red_fighters_comm_buffer[i] = self.red_fighters_comm_buffer[i][1:]

        for i, fighter in enumerate(self.fighters[self.num_red:]):
            if len(self.blue_fighters_comm_buffer[i]) == self.comm_delay:
                fighter.comunication_delayed_send = self.blue_fighters_comm_buffer[i][0]
                self.blue_fighters_comm_buffer[i] = self.blue_fighters_comm_buffer[i][1:]

        for i, fighter in enumerate(self.fighters):
            if fighter.side == 0:
                f_count = 0
                for friend in self.fighters[0: self.num_red]:
                    if friend.index == fighter.index:
                        pass
                    else:
                        fighter.comunication_recv[f_count] = friend.comunication_delayed_send
                        f_count += 1
            else:
                f_count = 0
                for friend in self.fighters[self.num_red: ]:
                    if friend.index == fighter.index:
                        pass
                    else:
                        fighter.comunication_recv[f_count] = friend.comunication_delayed_send
                        f_count += 1
        # 导弹DLL更新
        self.update_missile()

        # 雷达探测更新
        self.update_radar()

        # 战斗计算更新
        self.update_combat()

    def update_fighter(self):
        survive_array = np.ctypeslib.as_ctypes(self.survive_list)

        # 不同动作组合，使用一个，注释掉其他
        # 0525 动作组合 1: 连续法向过载 + 连续滚转速率 （不注释）
        for fighter in self.fighters:
            for j in range(4):
                if abs(fighter.action.u[j]) > 1:
                    fighter.action.u[j] = np.sign(fighter.action.u[j])

        for i, fighter in enumerate(self.fighters):
            if fighter.control_mode is ControlMode(4):
                self.fighters_modes[0][i] = Control_Mode_Dict[fighter.control_mode]
                for k in range(3):
                    self.fighters_modes[k + 1][i] = fighter.autopilot_mode[k]
            else:
                self.fighters_modes[0][i] = Control_Mode_Dict[fighter.control_mode]

        for i, fighter in enumerate(self.fighters):
            for j in range(4):
                self.action_u[j][i] = fighter.action.u[j]

        base_mode_array = np.ctypeslib.as_ctypes(self.fighters_modes[0])
        lon_mode_array = np.ctypeslib.as_ctypes(self.fighters_modes[1])
        lat_mode_array = np.ctypeslib.as_ctypes(self.fighters_modes[2])
        thrust_mode_array = np.ctypeslib.as_ctypes(self.fighters_modes[3])

        # 法向过载按DLL格式重新归一化
        for i, fighter in enumerate(self.fighters):
            self.action_u[0][i] *= 100
            self.action_u[0][i] = (self.action_u[0][i] - 50) / 50
            self.action_u[1][i] *= 9 if self.action_u[1][i] >= 0 else 3
            self.action_u[1][i] = (self.action_u[1][i] - 3) / 6

        cmd_1_array = np.ctypeslib.as_ctypes(self.action_u[1]) # 法向过载
        cmd_2_array = np.ctypeslib.as_ctypes(self.action_u[2]) # 滚转速率
        cmd_3_array = np.ctypeslib.as_ctypes(self.action_u[0]) # 油门
        cmd_4_array = np.ctypeslib.as_ctypes(self.action_u[3]) # 偏航

        self.lib.Update(len(self.fighters), int(self.dt / 0.01), byref(self.all_out), byref(self.all_err),
                   survive_array, base_mode_array, lon_mode_array, lat_mode_array, thrust_mode_array,
                   cmd_1_array, cmd_2_array, cmd_3_array, cmd_4_array)

        # 将更新后的信息放入fc_data里
        for i, fighter in enumerate(self.fighters):
            fighter.fc_data = self.out_list[i]

        # --------------------------对姿态角的NAN情况进行保护（把保护打在代码里！）-------------------------------------
        for i, fighter in enumerate(self.fighters):
            if np.isnan(fighter.fc_data.fPitchAngle):
                fighter.fc_data.fPitchAngle = 0
                fighter.action.fire_missile = 0
            if np.isnan(fighter.fc_data.fRollAngle):
                fighter.fc_data.fRollAngle = 0
                fighter.action.fire_missile = 0
            if np.isnan(fighter.fc_data.fYawAngle):
                fighter.fc_data.fYawAngle = 0
                fighter.action.fire_missile = 0
            if np.isnan(fighter.fc_data.fLongitude):
                fighter.fc_data.fLongitude = 0
                fighter.action.fire_missile = 0
            if np.isnan(fighter.fc_data.fLatitude):
                fighter.fc_data.fLatitude = 0
                fighter.action.fire_missile = 0
            if np.isnan(fighter.fc_data.fAltitude):
                fighter.fc_data.fAltitude = 0
                fighter.action.fire_missile = 0

        # 更新error列表
        for k in range(fighters_max_num):
            self.errs_list[k] = eval('self.all_err.outerr_' + str(k))

        # 更新fighter.state里的变量的值
        self.get_key_state()

    def update_combat(self):
        """
        战斗与损伤计算模块，包括机载武器伤害计算，超速/失速，坠地，碰撞死亡计算
        :return:
        """
        # 各个飞机在本次更新中受到的伤害,列表
        harm_list = np.zeros(len(self.fighters))

        # 伤害量更新部分
        for i, fighter in enumerate(self.fighters):
            # 循环，每架战机单独进行判断
            # 如果该架战机活着才继续判断
            if fighter.combat_data.survive_info:
                # GunLockWho 返回瞄准锥内，活着的且距离最近的敌机（0,1,2,...）
                sur_array = np.ctypeslib.as_ctypes(self.survive_list)
                fighter.action.discrete_c = self.lib.GunLockWho(i, self.num_blue, len(self.fighters), self.all_out,
                                                                sur_array)
                if fighter.action.discrete_c >= len(self.fighters):
                    # 如果找到了错误的目标，则跳出该次伤害判断
                    continue
                else:
                    if fighter.action.fire_gun and (fighter.combat_data.left_bullet > 0):
                        target = self.fighters[fighter.action.discrete_c]
                        ln, le, ld = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                                target.fc_data.fAltitude,
                                          fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                                fighter.fc_data.fAltitude)
                        dist = np.linalg.norm([ln, le, ld])
                        harm_list[fighter.action.discrete_c] += 2 * (1000 - dist) / 900
                        fighter.combat_data.left_bullet -= 1 * self.dt
                        # 单RL步累积伤害
                        fighter.combat_data.gun_harm_one_step += 1 * self.dt
                    else:
                        continue
            else:
                continue

        # 导弹命中判断部分
        for i, fighter in enumerate(self.fighters):
            # 循环，每架战机 单独进行判断
            for j, missile in enumerate(fighter.missiles):
                # 循环，每枚导弹 单独进行判断
                if (missile.state == 2) and (missile.attack == 1):      # 判断导弹是否命中+是否还有损伤能力
                    target_index = missile.target_index
                    # 根据角度对伤害值进行解算
                    # missile_ned_vel = [missile.dataout.m_vx, missile.dataout.m_vz, -missile.dataout.m_vy]
                    # angle_cos = np.inner(self.fighters[target_index - 1].state.ned_Vel,
                    #                      missile_ned_vel) / np.linalg.norm(
                    #     self.fighters[target_index - 1].state.ned_Vel) / np.linalg.norm(missile_ned_vel)
                    # if self.train_flag:
                    #     # delta_HP = 1.501
                    #     if np.random.uniform(0,1) > 0.00:
                    #         delta_HP = 3.
                    #     else:
                    #         delta_HP = 0.
                    # else:
                    #     if np.random.uniform(0,1) > 0.00:
                    #         delta_HP = 3.
                    #     else:
                    #         delta_HP = 0.
                    if target_index == fighter.index:
                        delta_HP = 0.0
                    else:
                        delta_HP = 3.0
                    # 血量扣除伤害值
                    self.fighters[target_index].combat_data.bloods = self.fighters[
                                                                         target_index].combat_data.bloods - delta_HP
                    # 该导弹的损伤能力置0
                    missile.attack = 0

        # 被击毁/死亡计算部分
        for i, fighter in enumerate(self.fighters):
            # 在本次更新前已经dead, 则跳过
            if not fighter.combat_data.survive_info:
                continue
            # 被击毁
            fighter.combat_data.bloods -= self.dt * harm_list[i]
            # 单RL步击杀、被伤
            fighter.combat_data.hurt_one_step += self.dt * harm_list[i]
            # 如果当前飞机的血量小于0（被击毁）
            if fighter.combat_data.bloods <= 0:
                for k, killer in enumerate(self.fighters):
                    # 找出凶手
                    if killer is fighter:
                        # 如果是自杀，则跳出
                        continue
                    else:
                        if (killer.action.discrete_c == i) and (killer.action.fire_gun):
                            # 如果killer的目标是当前飞机，且开火开关是开着的，则killer是真的killer
                            killer.combat_data.kill_num_one_step += 1

            # 持续失速或超速坠毁
            if ((abs(fighter.fc_data.fAttackAngle) > 30) and (fighter.fc_data.fMachNumber < 0.2)) or \
                    (abs(fighter.fc_data.fAttackAngle) > 35) or (abs(fighter.fc_data.fYawRate) > 320) or \
                    (abs(fighter.fc_data.fRollRate) > 320) or (abs(fighter.fc_data.fPitchRate) > 320):
                fighter.combat_data.err_time += self.dt
            else:
                fighter.combat_data.err_time = 0
            if (fighter.combat_data.err_time >= 10 or (np.isnan(fighter.fc_data.fAttackAngle)) or
                    (abs(fighter.fc_data.fVerticalVelocity) > 1500.)):
                fighter.combat_data.bloods = 0
                fighter.combat_data.be_effective_killed = 0

            # 坠地坠毁: 1.海拔高度小于0
            if fighter.fc_data.fAltitude <= 5:
                fighter.combat_data.bloods = 0
                fighter.combat_data.be_effective_killed = 0

            # 碰撞死亡计算
            for j, f in enumerate(self.fighters):
                if f is fighter:
                    continue
                # 在本次更新前已经dead, 则跳过
                if not f.combat_data.survive_info:
                    continue
                dist = np.linalg.norm(wgs84ToNED(f.fc_data.fLatitude, f.fc_data.fLongitude, f.fc_data.fAltitude,
                                                 fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                                 fighter.fc_data.fAltitude))
                if dist <= 5:
                    fighter.combat_data.bloods = 0
                    f.combat_data.bloods = 0
                    fighter.combat_data.be_effective_killed = 0
                    f.combat_data.be_effective_killed = 0

        # 必须最后更新生存状态
        for i, fighter in enumerate(self.fighters):
            if fighter.combat_data.bloods <= 0:
                fighter.combat_data.survive_info = False
                fighter.action.fire_gun = 0  # 若死亡，则关闭机载武器
            self.survive_list[i] = fighter.combat_data.survive_info
            if fighter.combat_data.bloods < 0:
                fighter.combat_data.bloods = 0
            if fighter.combat_data.bloods > 3:
                fighter.combat_data.bloods = 3

        # 最后的最后更新坠毁时间和红外信号
        for i, fighter in enumerate(self.fighters):
            if fighter.combat_data.survive_info:
                fighter.combat_data.death_time = 0
                fighter.combat_data.infrared_flag = True
            else:
                fighter.combat_data.death_time += self.dt
                if fighter.combat_data.death_time > 5:
                    fighter.combat_data.infrared_flag = False
                else:
                    fighter.combat_data.infrared_flag = True

    def update_radar(self):
        """
        飞机的机载雷达更新函数
        :return:
        """
        for i, fighter in enumerate(self.fighters):
            # 火控雷达更新
            fighter.sensors.fire_control_radar(fighter, self.fighters)
            # 5km内透明态势更新
            fighter.sensors.Eodas(fighter, self.fighters)
            # 雷达告警———导弹
            fighter.sensors.missile_alert(fighter, self.fighters)
            # 雷达告警——飞机
            fighter.sensors.radar_alert(fighter, self.fighters)

    def update_missile(self):
        """
        导弹更新函数，包括发射，导引头更新，动力学更新环节
        :return:
        """
        for i, fighter in enumerate(self.fighters):
            # 这里依据初始化设定，选择是否依赖雷达信息使用导弹
            for j, missile in enumerate(fighter.missiles):

                # 第一步，对导弹导引头进行更新，导引头自身确定目标，同时完成导弹目标经纬高的写入
                missile.missile_seeker_update(fighter, self.fighters)

                # 第二步，如果导弹发射需要雷达，且用户选定的目标在飞机雷达和导引头视场中，则为导弹指定该目标
                # 这里依据初始化设定，选择是否依赖雷达信息使用导弹
                if self.missile_without_radar:
                    pass
                else:
                    if missile.dataout.m_state == 0:
                        if fighter.target_index == missile.target_index:
                            # 用户指定的目标与导引头自身建立的目标一致，则直接进行发射准备
                            # 如果正在进行准备发射的导弹脱离雷达视场，则准备时间置0
                            if not (fighter.target_index in fighter.sensors.radar_list):
                                missile.launch_prepare_time = 0
                        else:
                            # 用户指定的目标是否是雷达视场中的敌机
                            if fighter.target_index in fighter.sensors.radar_list:
                                missile.target_index = fighter.target_index
                                # 改变为指定目标，导弹的锁定准备时间需要置0
                                missile.launch_prepare_time = 0
                            else:
                                # 不在雷达视场中则不为导弹目标赋值
                                # 但准备时间会一直为0，不会发射
                                missile.launch_prepare_time = 0
                    # 注： 只要确定雷达锁定某一目标，且目标在雷达和导引头视场内，则导弹准备计时会一直进行

            # 第三步，开始对导引头中的目标建立锁定，判断是否满足发射条件，最后的结果是导弹是否发射，哪枚导弹发射（用fighter.missiles_count表明轮到哪枚导弹发射）
            # 用户指定的目标没在导引头里也不会进入发射准备
            fighter.missile_launch_update(self.fighters)

            # 第四步，装填导弹dll所需的信息，调用Dll完成导弹信息更新
            for j, missile in enumerate(fighter.missiles):
                # 近距弹更新
                missile.missile_update_single(fighter, self.fighters)

    def get_key_state(self):
        for i, fighter in enumerate(self.fighters):
            ned_Pos = np.zeros(self.dim_p)
            North, East, Down = wgs84ToNED(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                           fighter.fc_data.fAltitude,)
            ned_Pos[0] = North
            ned_Pos[1] = East
            ned_Pos[2] = Down
            fighter.state.ned_Pos = ned_Pos
            fighter.state.ned_Vel = np.array([fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity,
                                              fighter.fc_data.fVerticalVelocity])
