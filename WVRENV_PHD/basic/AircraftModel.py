import numpy as np
import math
import copy
from ctypes import *
from WVRENV_PHD.basic.DataResolve import OutMsg, MultiInitMsg, MultiOutMsg, MultiErrorMsg, CombatMsg, \
    fighters_max_num, ControlMode, AircraftState, MissileInitial, MissileDataIn, MissileDataOut, ControlAction
from WVRENV_PHD.basic.SensorModel import Sensors
from WVRENV_PHD.SimArg import missile_str
from WVRENV_PHD.utils.GNCData import wgs84ToNED, ned_to_body, euler2vector, angle_2vec, ned_to_wgs84


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


class FighterAgent(object):
    def __init__(self, index, data_initial):
        # 战斗机实例整数编号(0,1,2)
        self.index = index
        self.obj_index = self.index + 1
        # 仿真模型名称，及tacview可视化机型
        self.type = "F-16"
        # 阵营 0 为红方 1 为蓝方
        self.side = 1

        # 飞行数据与态势数据（自身感知+队友共享）
        self.fc_data = OutMsg()
        # 对抗信息,与战斗，血量相关的量
        self.combat_data = CombatMsg()
        # 战斗机及武器控制
        self.action = ControlAction()
        # 储存飞机的一些基础量的类
        self.state = AircraftState()

        # 战斗机飞控模式
        self.control_mode = None

        # 战斗机机间通信链路(博士论文版)
        self.comunication_send = None
        self.comunication_delayed_send = None
        self.comunication_recv = []

        # ################################ 导弹模块 ################################
        # 战斗机导弹管理
        self.missiles = None
        # 导弹发射目标记录
        self.target_index = None  # 航电系统锁定目标记录
        self.missiles_count = 0  # 导弹发射计数
        self.missiles_max = data_initial.missiles_max  # 剩余导弹计数

        # ############################### 雷达模块 #################################
        self.sensors = Sensors(data_initial)

    def missile_launch_update(self, fighters):

        # 判断自己和目标敌机是否存活
        if self.combat_data.survive_info and fighters[self.target_index].combat_data.survive_info:
            # 判断飞机是否还载有导弹
            if self.missiles_count >= self.missiles_max:
                pass
            else:
                # 判断导弹的导引头是否截获目标
                # 只有没发射的导弹才需要发射判断，所以只需要判断下一个没发射的导弹是否探测到敌方。
                if self.missiles[self.missiles_count].target_index in self.missiles[self.missiles_count].target_list:
                    # 发射准备计时(只考虑当前准备要发射的导弹)
                    self.missiles[self.missiles_count].launch_prepare_time += 1
                    # print(f"missiles_count: {self.missiles_count}, debug: distlist: {self.missiles[self.missiles_count].target_distance_list}, tgindex: {self.missiles[self.missiles_count].target_index},tglist:{self.missiles[self.missiles_count].target_list}, "
                    #       f"pretime: {self.missiles[self.missiles_count].launch_prepare_time}, state: {self.missiles[0].state, self.missiles[1].state}")
                    # 目标是否在最小发射距离外 是否满足准备时间1s，都满足后，发射计数器+1，准备时间置0
                    target_dist_list_id = np.where(np.array(
                        self.missiles[self.missiles_count].target_list) == self.missiles[self.missiles_count].target_index)[0][0]
                    if ((self.missiles[self.missiles_count].launch_prepare_time >= 100) and
                            (self.missiles[self.missiles_count].target_distance_list[target_dist_list_id] > 500)):
                        self.missiles_count += self.action.fire_missile
                        # print(f"fire missile {self.missiles_count}, cmd: {self.action.fire_missile}")
                else:
                    # 目标脱离视场 则准备时间置0
                    self.missiles[self.missiles_count].launch_prepare_time = 0
class MissileAgent(object):
    """单个导弹的类，对单个导弹进行管理"""
    def __init__(self, index):
        # 仿真模型名称，可视化类型
        self.type = "AIM-9X"
        # 所属飞机的编号
        self.index = index
        # 导弹状态
        self.state = 0
        self.attack = 1
        self.mode = 1
        # 导弹所选目标记录
        self.target_index = None
        self.target_longitude = 0
        self.target_latitude = 0
        self.target_altitude = 0
        # 发射指令
        self.launch_num = 0
        # 发射准备时间（冷却，锁定）
        self.launch_prepare_time = 0

        # 所有导弹可以输出的数据，与飞机初始化类似
        self.datainitial = MissileInitial()
        self.datain = MissileDataIn()
        self.dataout = MissileDataOut()
        # 创建一个导弹的指针
        self.ptr = Missile_lib.create_ptr()

        # 导弹北东地
        self.ned_Pos = None

        # 导弹导引头
        self.target_list = []
        self.target_pitch_list = []
        self.target_yaw_list = []
        self.target_distance_list = []

    def missile_reset(self):
        # 基础信息重置
        self.datainitial.step_length = 0.01
        self.datainitial.k_pn = 4
        self.datainitial.missile_mode = 1
        Missile_lib.Initial(self.ptr, self.datainitial)

        # 战斗信息重置
        self.attack = 1  # 重置导弹的击中判断标志位
        self.launch_num = 0
        self.state = 0

        # 导弹导引头重置
        self.target_index = 0
        self.target_list = []
        self.target_pitch_list = []
        self.target_yaw_list = []
        self.target_distance_list = []

        # 发射准备时间重置
        self.launch_prepare_time = 0

    def missile_seeker_update(self, fighter, fighters):
        """
        导弹导引头更新函数
        :param fighter:
        :param fighters:
        :return:
        """
        # 清理列表
        self.target_list.clear()
        self.target_pitch_list.clear()
        self.target_yaw_list.clear()
        self.target_distance_list.clear()

        # 首先判断导弹自身是否处于正常工作或未发射状态中
        if self.state == 1 or self.state == 0:
            for i, enemy in enumerate(fighters):         # 循环，在所有战机中判断
                if fighter.index != enemy.index and enemy.combat_data.infrared_flag:        # 排除发出这枚导弹的飞机自身，判断敌方是否还具有红外信号
                    # 计算目标机的北东地坐标
                    target_North, target_East, target_Down = wgs84ToNED(enemy.fc_data.fLatitude,
                                                                        enemy.fc_data.fLongitude,
                                                                        enemy.fc_data.fAltitude)
                    target_NEDpos = np.array([target_North, target_East, target_Down])

                    # 计算自己的北东地坐标
                    missile_North, missile_East, missile_Down = wgs84ToNED(self.dataout.m_latitude,
                                                                           self.dataout.m_longitude,
                                                                           self.dataout.m_altitude)
                    missile_NEDpos = np.array([missile_North, missile_East, missile_Down])

                    # 首先依据角度模拟RCS，计算出导弹对这个飞机的可探测距离
                    # 视线矢量, 这里应该计算从敌机看向导弹的视线矢量
                    l_n, l_e, l_d = wgs84ToNED(self.dataout.m_latitude, self.dataout.m_longitude,
                                               self.dataout.m_altitude,
                                               enemy.fc_data.fLatitude, enemy.fc_data.fLongitude,
                                               enemy.fc_data.fAltitude)

                    # delta_pos = missile_NEDpos - target_NEDpos      # 一条由目标指向自己的矢量
                    delta_pos = np.array([l_n, l_e, l_d])      # 一条由目标指向自己的矢量

                    # 目标的姿态角
                    roll = enemy.fc_data.fRollAngle       # 注意这里取出的是导弹目标的姿态角
                    yaw = enemy.fc_data.fYawAngle
                    pitch = enemy.fc_data.fPitchAngle

                    vec_body = euler2vector(roll=roll, pitch=pitch, yaw=yaw)
                    angle = angle_2vec(vec_body, delta_pos)

                    # 依据角度模拟红外辐射特性，计算导弹对该目标的探测距离
                    # if abs(angle) <= 15:
                    #     detection_range = 1000
                    # elif abs(angle) <= 45:
                    #     detection_range = 3000
                    # elif abs(angle) <= 90:
                    #     detection_range = 6000
                    # elif abs(angle) <= 180:
                    #     detection_range = 10000
                    # else:
                    #     detection_range = 10000
                    detection_range = 10000

                    # 得到可探测距离后，再以导弹为中心，计算该目标是否处于导弹导引头范围内
                    l_n_1, l_e_1, l_d_1 = wgs84ToNED(enemy.fc_data.fLatitude, enemy.fc_data.fLongitude,
                                                     enemy.fc_data.fAltitude,
                                                     self.dataout.m_latitude, self.dataout.m_longitude,
                                                     self.dataout.m_altitude
                                               )
                    # delta_pos_1 = target_NEDpos - missile_NEDpos  # 一条由自己指向目标的矢量
                    delta_pos_1 = np.array([l_n_1, l_e_1, l_d_1])  # 一条由自己指向目标的矢量

                    # 双方距离模值
                    distance = delta_pos_1.dot(delta_pos_1) ** 0.5  # 向量点乘，求得内积，再开方

                    # 体坐标系下的视线矢量
                    roll = np.rad2deg(self.dataout.m_roll)
                    yaw = np.rad2deg(-self.dataout.m_yaw)
                    pitch = np.rad2deg(self.dataout.m_pitch)
                    vec_body = ned_to_body(delta_pos_1, yaw, pitch, roll)

                    # 视线矢量在体坐标系下的俯仰角和偏航角
                    enemy_pitch = np.rad2deg(-np.arctan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
                    enemy_yaw = np.rad2deg(np.arctan2(vec_body[1], vec_body[0]))

                    # 最后判断目标是否在导弹雷达扫描范围内
                    if (distance <= detection_range) and abs(enemy_pitch) <= 60 and abs(enemy_yaw) <= 60:  # 目标位于雷达探测范围内
                        self.target_list.append(enemy.index)
                        self.target_pitch_list.append(enemy_pitch)
                        self.target_yaw_list.append(enemy_yaw)
                        self.target_distance_list.append(distance)

        # 目标信息装填
        # 未发射时，只能装填选定目标的经纬高
        if self.state == 0:
            # 如果还没有针对某个目标开始准备发射，且视场内有目标，则对最近的目标建立锁定
            if (self.launch_prepare_time == 0) and (len(self.target_list) > 0):
                dis_min = min(self.target_distance_list)
                dis_min_index = self.target_distance_list.index(dis_min)
                self.target_index = self.target_list[dis_min_index]
            else:
                pass

            self.target_longitude = fighters[self.target_index].fc_data.fLongitude
            self.target_latitude = fighters[self.target_index].fc_data.fLatitude
            self.target_altitude = fighters[self.target_index].fc_data.fAltitude
        # 发射后，允许依据导弹的导引头探测结果切换目标
        elif self.state == 1:
            # 如果雷达范围内没有目标，则沿着原来的方向继续运动
            if len(self.target_list) == 0:
                self.target_index = 1000
            # 如果之前设定的目标仍然在导弹导引头探测范围内，则继续追踪，否则选择距离最近的目标。
            else:
                # 之前设定的目标依然在
                # 目标切换判断 （视线角几乎相同，跟踪距离更近的目标）
                if self.target_index in self.target_list:
                    target_list_id = np.where(np.array(self.target_list) == self.target_index)[0][0]
                    for tg_id, tg in enumerate(self.target_list):
                        if self.target_distance_list[tg_id] >= self.target_distance_list[target_list_id]:
                            continue
                        else:
                            if (abs(self.target_pitch_list[tg_id] - self.target_pitch_list[target_list_id]) < 1) and \
                                    (abs(self.target_yaw_list[tg_id] - self.target_yaw_list[target_list_id]) < 1):
                                self.target_index = tg
                            else:
                                continue
                    self.target_longitude = fighters[self.target_index].fc_data.fLongitude
                    self.target_latitude = fighters[self.target_index].fc_data.fLatitude
                    self.target_altitude = fighters[self.target_index].fc_data.fAltitude
                # 之前设定的目标丢失，但存在其他目标
                else:
                    # 计算其他目标距离，找到最近的
                    dis_min = min(self.target_distance_list)
                    dis_min_index = self.target_distance_list.index(dis_min)
                    self.target_index = self.target_list[dis_min_index]
                    self.target_longitude = fighters[self.target_index].fc_data.fLongitude
                    self.target_latitude = fighters[self.target_index].fc_data.fLatitude
                    self.target_altitude = fighters[self.target_index].fc_data.fAltitude

    def missile_update_single(self, fighter, fighters):

        # ############################## 开火指令解算 ##############################
        # 当开火指令为1时，解算这是第N次开火，并选择第N枚弹发射
        if self.mode == 1:
            # 判断开火指令是否为1，导弹内部发射指令是否为0
            if fighter.action.fire_missile == 1 and self.launch_num == 0:
                # 判断是否该发射这枚导弹
                if fighter.missiles_count - 1 == self.index:
                    self.launch_num = 1

                    # 将飞机的发射指令置回0
                    fighter.action.fire_missile = 0

        # ############################## 输入量数据打包 ##############################
        # 载机LLA数据
        fighter_data_list = np.array([fighter.fc_data.fLongitude, fighter.fc_data.fLatitude, fighter.fc_data.fAltitude])
        self.datain.fighter_LLA = np.ctypeslib.as_ctypes(fighter_data_list)
        # V数据
        fighter_data_list = np.array([fighter.fc_data.fNorthVelocity, -fighter.fc_data.fVerticalVelocity, fighter.fc_data.fEastVelocity])
        self.datain.fighter_v = np.ctypeslib.as_ctypes(fighter_data_list)
        # angle数据
        fighter_data_list = np.array([np.deg2rad(fighter.fc_data.fPitchAngle), - np.deg2rad(fighter.fc_data.fYawAngle),
                                      np.deg2rad(fighter.fc_data.fRollAngle)])
        self.datain.fighter_angle = np.ctypeslib.as_ctypes(fighter_data_list)
        # 目标LLA数据
        if (len(self.target_list) == 0) or (not fighters[self.target_index].combat_data.survive_info):
            # old_los_n, old_los_e, old_los_d = wgs84ToNED(self.target_latitude, self.target_longitude,
            #                                              self.target_altitude,
            #                                              self.dataout.m_latitude,
            #                                              self.dataout.m_longitude,
            #                                              self.dataout.m_altitude)
            m_North, m_East, m_Down = wgs84ToNED(self.dataout.m_latitude,
                                                 self.dataout.m_longitude,
                                                 self.dataout.m_altitude)

            # target_n = m_North + 100 * (old_los_n if abs(old_los_n) < 2 else np.sign(old_los_n) * 2)
            # target_e = m_East + 100 * (old_los_e if abs(old_los_e) < 2 else np.sign(old_los_e) * 2)
            # target_d = m_Down + 100 * (old_los_d if abs(old_los_d) < 2 else np.sign(old_los_d) * 2)

            target_n = m_North + 25 * np.sign(self.dataout.m_vx)
            target_e = m_East + 25 * np.sign(self.dataout.m_vz)
            target_d = m_Down + 0 * np.sign(-self.dataout.m_vy)

            self.datain.target_longtitude, self.datain.target_latitude, self.datain.target_altitude = (
                ned_to_wgs84([target_n, target_e, target_d]))
        else:
            self.datain.target_longtitude = self.target_longitude
            self.datain.target_latitude = self.target_latitude
            self.datain.target_altitude = self.target_altitude
        # 控制指令打包（目前只有发射）
        self.datain.launch_num = self.launch_num

        # ############################## 步长更新 ##############################
        if self.mode == 1:
            self.dataout = Missile_lib.Update(self.ptr, self.datain, self.dataout)

        # ############################## 导弹的北东地位置解算 ##############################
        self.state = self.dataout.m_state if self.dataout.m_state < 3 else 3
        self.ned_Pos = np.zeros(3)
        north, east, down = wgs84ToNED(lat=self.dataout.m_latitude, lon=self.dataout.m_longitude,
                                       h=self.dataout.m_altitude, lat0=24.8976763, lon0=160.123456, h0=0)
        self.ned_Pos[0] = north
        self.ned_Pos[1] = east
        self.ned_Pos[2] = down
