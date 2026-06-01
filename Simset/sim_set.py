import random
from WVRENV_PHD.SimArg import InitialData
from WVRENV_PHD.SimInput import FighterDataIn
from WVRENV_PHD.basic.ObstacleModel import get_elevation
from WVRENV_PHD.basic.SensorModel import Terrain_sensor
from WVRENV_PHD.simulation_env import CombatEnv
from WVRENV_PHD.utils.GNCData import vector_angle, euler2vector, ned_to_body, wgs84ToNED, ned_to_wgs84
from scipy.spatial.transform import Rotation as R
import numpy as np
from point_cloud import Voxel


def make_sim_env(pargs):
    """
    :return: env 仿真环境实例 sim_in_list 仿真环境中所有飞机的控制输入实例
    """
    # 实例化环境
    env = CombatEnv()
    # 环境初始数据
    initial_data = InitialData()
    initial_data.log_tacview = True  # 是否开启内置的仿真文件记录功能

    initial_data.log_csv = False  # 是否输出记录文件
    initial_data.dll_str = '.\\MultiFighter.dll'  # 调用的模型路径，若加载失败，可改为绝对路径

    # 仿真设定
    initial_data.dt = 0.01  # 最小数据更新率（底层步长为0.01s）
    initial_data.len_max = pargs.sim_max_steps  # 单轮仿真长度

    # 红蓝双方战机数量
    initial_data.num_blue = 1  # 蓝机数量
    initial_data.num_red = 1  # 红机数量
    initial_data.originLongitude = 96.3  # 仿真的经纬度原点位置
    initial_data.originLatitude = 32.8

    # 初始载弹量
    initial_data.missiles_max = 0

    # 机载雷达范围设定
    initial_data.radar_range = 40000
    initial_data.radar_vertical_scan = 30  # 雷达垂直扫描范围
    initial_data.radar_horizontal_scan = 30  # 雷达水平扫描范围
    initial_data.eodas_range = 10000  # 光电分布式探测孔径系统（EODAS）探测范围
    initial_data.alert_missile_range = 5000  # 来袭导弹告警范围
    initial_data.missile_without_radar = True  # 导弹发射是否不依赖雷达锁定的开关

    sim_in_list = [FighterDataIn() for _ in range(initial_data.num_blue + initial_data.num_red)]

    # 飞机设定控制模式
    for i in range(initial_data.num_blue + initial_data.num_red):
        # ppo红机控制模式：0
        if i == 0 :
            sim_in_list[i].control_mode = 0
        if i == 1:
            sim_in_list[i].control_mode = 0

    # 完成初始化
    env.initial(sim_in_list, initial_data)

    #地形传感器
    env.terrain_sensor = Terrain_sensor(pargs)
    env.voxel = Voxel()

    return env, sim_in_list


def Reset(env, op_id,sim_set_dict = None):
    if sim_set_dict is None:
        # 载机高度、马赫数、导弹发射俯仰角 # 目标距离、安稳系下视线方位角、高低角 # 目标航向、马赫数
        # TODO：依据场景修改这里每个变量的随机范围
        sim_set_dict = {"alt": np.random.uniform(5000, 8000), "red_ma": np.random.uniform(0.7, 1.0),
                        "fire_pitch": 0, "dist": np.random.uniform(8000, 10000),
                        "body_q_t": np.random.uniform(-60, 60), "body_q_d": np.random.uniform(-20, 20),
                        "blue_v_yaw": np.random.uniform(-180, 180),
                        "blue_ma": np.random.uniform(0.7, 1.0)}

    #重置载机航向
    red_orientation = np.random.uniform(-180, 180)
    blue_orientation = sim_set_dict['blue_v_yaw']
    env.initial_data.orientation = []
    env.initial_data.orientation.append(red_orientation)
    env.initial_data.orientation.append(blue_orientation)

    #重置载机坐标
    env.initial_data.NED = []
    red_ned = [
        np.random.uniform(-5000, -3000),
        np.random.uniform(2000, 5000),
        -sim_set_dict["alt"]
    ]


    #计算蓝机位置
    los_stable_body = euler2vector(0, sim_set_dict['body_q_d'], sim_set_dict['body_q_t'])
    r_ned2body = R.from_euler('ZYX', [red_orientation, 0, 0], degrees=True).inv()
    R_StabbleBody2NED = r_ned2body.inv().as_matrix()
    los_NED = np.matmul(R_StabbleBody2NED, los_stable_body)
    blue_ned = [red_ned[0] + los_NED[0] * sim_set_dict['dist'], red_ned[1] + los_NED[1] * sim_set_dict['dist'],
               max(min(-sim_set_dict['alt'] + los_NED[2] * sim_set_dict['dist'], -5000), -8000)]


    #防止初始时刻载机高度在地形下
    red_lon, red_lat, red_alt = ned_to_wgs84(red_ned)
    red_rel_ele = red_alt - get_elevation(red_lat, red_lon)
    if red_rel_ele < 100:
        red_ned[2] = - (get_elevation(red_lat, red_lon) + 300)
    blue_lon, blue_lat, blue_alt = ned_to_wgs84(blue_ned)
    blue_rel_ele = blue_alt - get_elevation(blue_lat, blue_lon)
    if blue_rel_ele < 100:
        blue_ned[2] = - (get_elevation(blue_lat, blue_lon) + 300)
    env.initial_data.NED.append(red_ned)
    env.initial_data.NED.append(blue_ned)

    #重置马赫数
    env.initial_data.ma = [sim_set_dict['red_ma'], sim_set_dict['blue_ma']]

    #重置油量
    env.initial_data.FuelKg = [0.8 * 3000, 0.8 * 3000]

    env.reset(op_id)
    return

