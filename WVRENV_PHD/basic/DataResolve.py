from ctypes import *
from enum import Enum
import numpy as np
from WVRENV_PHD.SimArg import fighters_max_num, missiles_max


class CombatMsg(object):
    """与战斗相关的属性与信息的类"""
    def __init__(self):
        # 安全性变量
        self.bloods = 3
        self.survive_info = True
        self.err_time = 0
        # 战斗机红外信号特征
        self.infrared_flag = True
        # 坠毁时间（从坠毁后开始计时）
        self.death_time = 0
        # 武器变量
        self.left_bullet = 20
        # 单个RL步长内的击毁数量
        self.kill_num_one_step = 0
        # 单个RL步长内的机炮伤害量
        self.gun_harm_one_step = 0
        # 单个RL步长内受到的伤害量
        self.hurt_one_step = 0
        # 是否有效击杀
        self.be_effective_killed = 1        # 23WRZF特有标志量，默认为1，当因超速/失速/坠地/碰撞死亡时，该变量置0，表示不给对方加分


class ControlAction(object):
    def __init__(self):
        # 控制战斗机的连续动作
        self.u = np.zeros(4)
        # 机炮开火动作：0关1开
        self.fire_gun = 0

        # 导弹开火动作：0关1开
        self.fire_missile = 0
        self.fire_missile_mid = 0       # 中距弹发射指令

        # 选择目标
        self.discrete_c = fighters_max_num + 1


class ControlMode(Enum):
    """设定操纵方式的枚举Enum类"""
    # Typical control mode: control function api
    # Joystick control 4|操纵杆控制，4个控制输入量
    JOY_XYZT = 0
    # desired trajectory angle, vel 3|航迹角与速度控制，3个控制输入量
    VEL_TRAJECTORY_ANGLE = 1
    # normal load and lateral load in trajectory coordination, thrust 4|法向过载与横向过载控制+油门，4个控制输入量
    NORMAL_LATERAL_LOAD = 2
    # G force load and roll rate, thrust 3|轴向过载+体轴滚转角+油门，3个控制输入量
    G_FORCE_ROLL = 3
    # Autopilot 3|自动驾驶仪
    AUTOPILOT_MIX = 4


class AircraftState(object):
    """设定飞机基础属性的类"""
    def __init__(self):
        # Pos state
        self.ned_Pos = None
        # Vel state
        self.ned_Vel = None
        # # vel and vel angle
        # self.traj_Vel = None
        # # Attitude state
        # self.body_Angle = None
        # # Mach number
        # self.ma = None
        # # Mass state
        # self.mass = None


class InitMsg(Structure):
    """输入给F16 DLL模型的类"""
    _fields_ = [("iMediumRangeAAM", c_int),
                ("iShortRangeAAM", c_int),
                ("fFuelContentKg", c_double),
                ("fAltitude", c_double),
                ("fMach", c_double),
                ("fStep", c_double),
                ("fLongitude", c_double),
                ("fLatitude", c_double),
                ("Orientation", c_double)]


class OutMsg(Structure):
    """F16 DLL模型输出信息的类"""
    _fields_ = [("fNormalAcceleration", c_double),
                ("fLateralAcceleration", c_double),
                ("fLongitudinalAcceleration", c_double),

                ("fNorthAcceleration", c_double),
                ("fEastAcceleration", c_double),
                ("fVerticalAcceleration", c_double),

                ("fRollRate", c_double),
                ("fPitchRate", c_double),
                ("fYawRate", c_double),

                ("fRollAcceleration", c_double),
                ("fPitchAcceleration", c_double),
                ("fYawAcceleration", c_double),

                ("fNormalLoad", c_double),
                ("fLateralLoad", c_double),
                ("fLongitudeinalLoad", c_double),

                ("fNorthVelocity", c_double),
                ("fEastVelocity", c_double),
                ("fVerticalVelocity", c_double),

                ("fNormalVelocity", c_double),
                ("fLateralVelocity", c_double),
                ("fLongitudianlVelocity", c_double),

                ("fLongitude", c_double),
                ("fLatitude", c_double),
                ("fAltitude", c_double),

                ("fRollAngle", c_double),
                ("fPitchAngle", c_double),
                ("fYawAngle", c_double),

                ("fAttackAngle", c_double),
                ("fSideslipAngle", c_double),
                ("fTrueAirSpeed", c_double),
                ("fIndicatedAirSpeed", c_double),

                ("fGroundSpeed", c_double),

                ("fPathPitchAngle", c_double),
                ("fPathYawAngle", c_double),

                ("fNumberofFuel", c_double),
                ("fFuelConsumptionRate", c_double),

                ("fMachNumber", c_double),
                ("fAtmosphereTempreture", c_double),
                ("fTotalPressure", c_double),
                ("fStaticPressure", c_double),
                ("fImpactPressure", c_double),
                ("fAtmosphereDensity", c_double),

                ("fThrust", c_double),
                ("fThrottle", c_double),

                ("fMass", c_double),

                ("dLeftElevatorPositionDeg", c_double),
                ("dRightElevatorPositionDeg", c_double),
                ("dLeftAileronPositionDeg", c_double),
                ("dRightAileronPositionDeg", c_double),
                ("dRudderPositionDeg", c_double),
                ("dLEFlapPositionDeg", c_double),
                ("dSpeedBreakPosDeg", c_double),
                ]


class MultiInitMsg(Structure):
    _fields_ = [("initData_" + str(i), InitMsg) for i in range(fighters_max_num)]
    # 在 Python 中定义一个 c 类型的结构体，该结构体类必须继承自 ctypes.Structure
    # MultiInitMsg类为包含了数量为fighters_max_num个的类型为InitMsg的c类型的结构体


class MultiOutMsg(Structure):
    _fields_ = [("outData_" + str(i), OutMsg) for i in range(fighters_max_num)]


class MultiErrorMsg(Structure):
    _fields_ = [("outerr_" + str(i), c_bool) for i in range(fighters_max_num)]


class MissileInitial(Structure):
    _fields_ = [("step_length", c_double),
                ("k_pn", c_double),
                ("missile_mode", c_int),
                ("index", c_int)]


class MissileDataIn(Structure):
    _fields_ = [("fighter_LLA", c_double * 3),
                ("fighter_v", c_double * 3),
                ("fighter_angle", c_double * 3),
                ("target_longtitude", c_double),
                ("target_latitude", c_double),
                ("target_altitude", c_double),
                ("launch_num", c_int)]


class MissileDataOut(Structure):
    _fields_ = [("m_longitude", c_double),
                ("m_latitude", c_double),
                ("m_altitude", c_double),

                ("m_pitch", c_double),
                ("m_roll", c_double),
                ("m_yaw", c_double),
                ("m_pathpitch", c_double),
                ("m_pathyaw", c_double),
                ("m_alpha", c_double),
                ("m_beta", c_double),

                ("m_vx", c_double),         # 注意这里是北天东速度
                ("m_vy", c_double),
                ("m_vz", c_double),
                ("m_vbody", c_double),
                ("m_ma", c_double),

                ("m_ax", c_double),
                ("m_ay", c_double),
                ("m_az", c_double),

                ("m_Mass", c_double),
                ("time_fire", c_double),

                ("seeker_angle_pitch", c_double),
                ("seeker_angle_yaw", c_double),
                ("dotqz", c_double),
                ("dotqy", c_double),
                ("distance", c_double),
                ("seeker_state", c_double),

                ("missilemode", c_double),
                ("m_state", c_double),
                ("index", c_int)]


class Fighter_Data(object):
    def __init__(self):
        ##################### 本机信息 #######################（始终可以获得）
        self.selfdata = SelfData()

        ##################### 机载雷达信息 #######################（需要火控雷达范围内才可）
        self.radardata = RadarData()

        ##################### 态势预警信息 #######################（始终可以获得）
        self.statedata = StateData()

        ##################### 近距透明信息 #######################（距离在5KM内）
        self.closedata = CloseData()

        ##################### 告警系统信息 #######################
        self.alertdata = AlertData()

        ##################### 友机传递信息 #######################
        self.communication = [b'\x00', b'\x00', b'\x00', b'\x00', b'\x00']


class SelfData(object):
    def __init__(self):
        ##################### 本机信息 #######################（始终可以获得）
        # 控制模式状态
        self.control_mode = 0
        # 剩余航炮弹药量
        self.left_bullet = 0
        # 剩余空空导弹数量
        self.left_missile = 0
        # 剩余生命值
        self.left_bloods = 0

        # 北东地加速度
        self.NorthAcceleration = 0
        self.EastAcceleration = 0
        self.VerticalAcceleration = 0
        # 体轴滚转角,俯仰角，偏航角速度
        self.RollRate = 0
        self.PitchRate = 0
        self.YawRate = 0
        # 体轴法向，侧向，纵向过载
        self.NormalLoad = 0
        self.LateralLoad = 0
        self.LongitudeinalLoad = 0
        # 北东地速度
        self.NorthVelocity = 0
        self.EastVelocity = 0
        self.VerticalVelocity = 0
        # 体轴法向，侧向，纵向速度
        self.NormalVelocity = 0
        self.LateralVelocity = 0
        self.LongitudianlVelocity = 0
        # 经纬高位置
        self.Longitude = 0
        self.Latitude = 0
        self.Altitude = 0
        # 姿态角
        self.RollAngle = 0
        self.PitchAngle = 0
        self.YawAngle = 0
        # 攻角侧滑角
        self.AttackAngle = 0
        self.SideslipAngle = 0

        # 真空速
        self.TrueAirSpeed = 0
        # 指示空速
        self.IndicatedAirSpeed = 0
        # 地速
        self.GroundSpeed = 0
        # 剩余油量
        self.NumberofFuel = 0
        # 推力
        self.Thrust = 0

        # 导弹状态
        self.Missile1State = 0
        self.Missile2State = 0


class RadarData(object):
    def __init__(self):
        ##################### 机载雷达信息 #######################（需要火控雷达范围内才可）
        # 友机高低角
        self.friend_EleAngle = 0
        # 友机方位角
        self.friend_AziAngle = 0
        # 友机距离
        self.friend_Distance = 0
        # 北东地速度
        self.friend_NorthVelocity = 0
        self.friend_EastVelocity = 0
        self.friend_VerticalVelocity = 0

        # 敌机1编号
        self.target1_Index = 0
        # 敌机高低角
        self.target1_EleAngle = 0
        # 敌机方位角
        self.target1_AziAngle = 0
        # 敌机距离
        self.target1_Distance = 0
        # 北东地速度
        self.target1_NorthVelocity = 0
        self.target1_EastVelocity = 0
        self.target1_VerticalVelocity = 0

        # 敌机2编号
        self.target2_Index = 0
        # 敌机高低角
        self.target2_EleAngle = 0
        # 敌机方位角
        self.target2_AziAngle = 0
        # 敌机距离
        self.target2_Distance = 0
        # 北东地速度
        self.target2_NorthVelocity = 0
        self.target2_EastVelocity = 0
        self.target2_VerticalVelocity = 0


class StateData(object):
    def __init__(self):
        ##################### 态势预警信息 #######################
        # 友机信息
        # 经纬高位置
        self.friend_Longitude = 0
        self.friend_Latitude = 0
        self.friend_Altitude = 0
        self.friend_Survive = False

        # 敌机1编号
        self.target1_Index = 0
        # 经纬高位置
        self.target1_Longitude = 0
        self.target1_Latitude = 0
        self.target1_Altitude = 0
        self.target1_Survive = False

        # 敌机2编号
        self.target2_Index = 0
        # 经纬高位置
        self.target2_Longitude = 0
        self.target2_Latitude = 0
        self.target2_Altitude = 0
        self.target2_Survive = False

class CloseData(object):
    def __init__(self):
        ##################### 近距透明信息 #######################（距离在5KM内）
        # 友机信息
        # 友机高低角
        self.friend_EleAngle = 0
        # 友机方位角
        self.friend_AziAngle = 0
        # 友机距离
        self.friend_Distance = 0

        # 敌机1信息
        # 敌机1编号
        self.target1_Index = 0
        # 敌机高低角
        self.target1_EleAngle = 0
        # 敌机方位角
        self.target1_AziAngle = 0
        # 敌机距离
        self.target1_Distance = 0

        # 敌机2信息
        # 敌机2编号
        self.target2_Index = 0
        # 敌机高低角
        self.target2_EleAngle = 0
        # 敌机方位角
        self.target2_AziAngle = 0
        # 敌机距离
        self.target2_Distance = 0


class AlertData(object):
    def __init__(self):
        ##################### 告警系统信息 #######################
        self.emergency_num = 0
        self.emergency_EleAngle = []
        self.emergency_AziAngle = []

        self.emergency_missile_num = 0
        self.emergency_missile_EleAngle = []
        self.emergency_missile_AziAngle = []






