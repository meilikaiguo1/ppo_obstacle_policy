import os

from WVRENV_PHD.basic.ObstacleModel import ObstacleEncoder, get_elevation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from WVRENV_PHD.basic.WorldModel import World
from WVRENV_PHD.basic.AircraftModel import FighterAgent, MissileAgent
from WVRENV_PHD.basic.DataResolve import ControlMode
from WVRENV_PHD.utils.GNCData import ned_to_wgs84
from WVRENV_PHD.utils.data_record import record_tacview, record_csv
from WVRENV_PHD.utils.visualtcp import TcpRender
import time
from Spinup.mpi_torch_utils import mpi_fork, proc_id


class CombatEnv(object):
    """与仿真对抗的世界环境相关的类"""
    def __init__(self):
        # 环境内部记录步长和初始化量
        self.epoch = -2                 # 在demo中，每次运行reset函数时+1，而初始化会运行两次reset，所以初始值设为-2，保证在第一轮训练时该值为0
        self.time_count = 0
        self.Obstacle = ObstacleEncoder()

    def initial(self, datain, data_initial):
        """
        初始化函数
        :param datain: 所有飞机的输入控制量，需要初始化控制模式
        """
        # 初始化世界
        self.initial_data = data_initial
        self.world = World(self.initial_data)
        self.max_len = self.initial_data.len_max

        # 初始化飞机数量设定
        self.num_BlueFighter = self.initial_data.num_blue
        self.num_RedFighter = self.initial_data.num_red
        self.num_Fighter = self.num_BlueFighter + self.num_RedFighter

        # 初始化经纬高原点位置设定
        self.world.dt = data_initial.dt  #最小数据更新率(底层步长固定为0.01s)
        self.world.originLongitude = self.initial_data.originLongitude  # 仿真的经纬度原点位置，默认为0， Longitude经度
        self.world.originLatitude = self.initial_data.originLatitude  # Latitude纬度

        # ############################## 初始化战斗机 ###############################
        self.world.fighters = [FighterAgent(i, self.initial_data) for i in
                               range(self.num_Fighter)]  # 把战斗机的类存入world的fighters里（在这里才开始初始化战斗机）
        # 为每架战斗机设定基础属性
        for i, fighter in enumerate(self.world.fighters):
            fighter.control_mode = ControlMode(datain[i].control_mode)  # 操纵方式：轴向过载+体轴滚转角+油门，3个控制输入量
            fighter.side = 0 if i < self.num_RedFighter else 1  # 为每架战机设定阵营
            fighter.type = "F-16"   # 为每架战机设定仿真可视化模型

        # ########################## 初始化每架战斗机所携带的导弹 ###########################
        for j, fighter in enumerate(self.world.fighters):
            fighter.missiles = [MissileAgent(m) for m in range(self.initial_data.missiles_max)]

        # ########################### 初始化数据记录路径参数 ###########################
        self.log_tacview = self.initial_data.log_tacview  # 是否输出记录文件
        self.log_csv = self.initial_data.log_csv  # 是否输出记录文件
        localtime = time.strftime("%Y_%m_%d", time.localtime())
        video_dir = os.path.join(".\\tmp", 'train_video_' + str(localtime))
        os.makedirs(video_dir, exist_ok=True)
        self.file_dir = video_dir

        # ########################### 是否使用tcp进行可视化 ###########################
        self.tcp_use = self.initial_data.tcp_use
        if self.tcp_use:
            self.visual_tcp = TcpRender(42674)
            self.visual_tcp.send_head()

        self.reset()

    def reset(self, op_id = 0):
        """
        重置函数
        """
        self.time_count = 0
        self.epoch += 1

        if op_id == 3:
            self.world.fighters[1].control_mode = ControlMode(3)
        else:
            self.world.fighters[1].control_mode = ControlMode(0)

        # 重置每架飞机的初始位置,初始速度
        for i, fighter in enumerate(self.world.fighters):
            ned_Pos = np.array(self.initial_data.NED[i])
            orientation = self.initial_data.orientation[i]

            # 将上述重新初始化后的内容输入传递给DLL
            fLongitude, fLatitude, fAltitude = ned_to_wgs84(ned_Pos, self.world.originLatitude,
                                                            self.world.originLongitude, 0)
            self.world.init_list[i].iMediumRangeAAM = 4
            self.world.init_list[i].iShortRangeAAM = 2 # 这里可以改变飞机质量
            self.world.init_list[i].fFuelContentKg = self.initial_data.FuelKg[i]  # 初始携带的载油量
            self.world.init_list[i].fMach = self.initial_data.ma[i]  # 初始速度
            self.world.init_list[i].fStep = 0.01  # 仿真步长
            self.world.init_list[i].fLongitude = fLongitude  # 初始位置（经纬高）
            self.world.init_list[i].fLatitude = fLatitude
            self.world.init_list[i].fAltitude = fAltitude
            self.world.init_list[i].Orientation = orientation  # 初始航向



        self.world.initialize()

    def update(self,pid, datain):
        """
        回合更新函数
        :param datain: 所有飞机的输入控制量
        :return: terminal_mul: 终止代码
        """
        # ################## 外部控制输入量解包与更新 #################
        for i, fighter in enumerate(self.world.fighters):
            # 控制输入
            fighter.action.u = datain[i].control_input
            # 机炮开火指令
            fighter.action.fire_gun = datain[i].fire
            # 机载雷达锁定目标选择
            fighter.target_index = datain[i].target_index
            if fighter.target_index >= self.num_Fighter:
                fighter.target_index = fighter.index

            # 导弹开火指令
            fighter.action.fire_missile = datain[i].missile_fire

            # 通信链路发送内容 (博士论文版)
            fighter.comunication_send = datain[i].comm_send

        # ##################### 更新飞机，导弹，雷达信息 #####################
        self.world.step()

        # ########################## 终止条件判断 ##########################
        terminal_mul = self.terminal_judge()

        # ########################## 记录文件 #############################
        if pid == 0 and self.time_count % 10 == 0:
            record_tacview(self, pid)
        if self.log_csv:
            pass
            # record_csv(self)

        # 时间步长计数器+1
        self.time_count += 1

        return terminal_mul

    def terminal_judge(self):
        """
              判断是否满足终止条件：仿真时长最大/有一方飞机全灭且没有飞行中的导弹
              :return: terminal_mul 终止代码：-1正常运行;0仿真时长最大;1蓝方全灭;2红方全灭;3双方均全灭
              """
        # 默认为-1，表示仿真继续，一旦置为正数，则仿真终止
        terminal_mul = -1

        # 仿真时长达到最大
        if self.time_count >= (self.max_len - 1):
            terminal_mul = 0

        # 收集所有飞机的存活信息
        done_n = []
        for i, fighter in enumerate(self.world.fighters):
            done_n.append(fighter.combat_data.survive_info)

        # 任意一方所有飞机退出比赛，且该方没有飞行中的空空导弹

        # 判断蓝方是否全灭
        if all(item is False for item in done_n[self.num_RedFighter:]):
            # 判断蓝方有没有飞行中的导弹
            missile_fly = False
            for i, fighter in enumerate(self.world.fighters[self.num_RedFighter:]):
                if not missile_fly:
                    for j, missile in enumerate(fighter.missiles):
                        if missile.state == 1:
                            missile_fly = True
                            break
            if missile_fly == False:
                terminal_mul = 1

        # 判断红方是否全灭
        if all(item is False for item in done_n[0:self.num_RedFighter]):
            # 判断红方有没有飞行中的导弹
            missile_fly = False
            for i, fighter in enumerate(self.world.fighters[0:self.num_RedFighter]):
                if not missile_fly:
                    for j, missile in enumerate(fighter.missiles):
                        if missile.state == 1:
                            missile_fly = True
                            break
            if missile_fly == False:
                terminal_mul = 2
        # 当所有飞机都死亡，则直接结束该轮仿真
        if all(item is False for item in done_n[:]):
            terminal_mul = 3

        # 红机坠地
        red_v_elevation = get_elevation(self.world.fighters[0].fc_data.fLatitude,
                                        self.world.fighters[0].fc_data.fLongitude)
        if self.world.fighters[0].fc_data.fAltitude - red_v_elevation <= 5:
            terminal_mul = 4

        # 蓝机坠地
        blue_v_elevation = get_elevation(self.world.fighters[1].fc_data.fLatitude,
                                         self.world.fighters[1].fc_data.fLongitude)
        if self.world.fighters[1].fc_data.fAltitude - blue_v_elevation <= 5:
            terminal_mul = 5

        return terminal_mul

    def tcp_update(self, time_count):
        if self.tcp_use:
            self.visual_tcp.send_data_render(self.world.dt, time_count, self)
