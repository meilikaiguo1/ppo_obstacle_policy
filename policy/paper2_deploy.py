import torch
import torch.nn as nn
import numpy as np
import time
import os
from WVRENV_PHD.utils.GNCData import ned_to_wgs84, wgs84ToNED, euler2vector
from scipy.spatial.transform import Rotation as R

K_W = 4  # 滚转速率P参数
K_W_D = 2  # 滚转速率D参数
K_N = 2.8  # 法向过载P参数
K_R = 1.5  # 俯仰抑制系数导数
K_Y = 0.07  # 方向舵参数
K_T = 2.5  # 油门参数


def euler2quat(roll=0, pitch=0, yaw=0):
    """
	:param roll: 滚转角 rad
	:param picth: 俯仰角 rad
	:param yaw: 偏航角 rad
	:return: 四元数
	"""
    phi = roll
    theta = pitch
    psi = yaw
    q = np.ones(4)
    q[3] = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    q[0] = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    q[1] = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    q[2] = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    return q


def vector_angle(vec1, vec2):
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    crossX = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    crossY = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    crossZ = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    norm = np.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)
    return (np.arctan2(norm, dot) / np.pi) * 180


def logistic(x, growth_rate, mid_point):
    if (-growth_rate * (x - mid_point)) > 700:
        S = 0
    else:
        S = 1 / (1 + np.exp(-growth_rate * (x - mid_point)))
    return S


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Dropout(p=0.05), act()]
        else:
            if act is nn.Softmax:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act(dim=0)]
            else:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Tanh)

    def forward(self, obs):
        mu = self.mu_net(obs)
        return mu


class VecDotCtrl(object):
    def __init__(self):
        # 当前机体向量
        self.ati_vec_cur = np.ones(3) / np.sqrt(3)
        # 期望机体向量
        self.ati_vec_des = np.ones(3) / np.sqrt(3)
        # 当前姿态四元数及逆
        self.quat_vec_cur_inv = np.ones(4) / 2
        self.quat_vec_cur = np.ones(4) / 2
        # 期望姿态四元数及逆
        self.quat_vec_des_inv = np.ones(4) / 2
        self.quat_vec_des = np.ones(4) / 2
        # 法向过载角误差 度
        self.theta_err = 0
        # 上一时刻的滚转误差
        self.last_roll_err = 0

    def vec_quat_cur(self, roll, pitch, yaw):
        rotation_1 = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
        self.quat_vec_cur_inv = rotation_1.as_quat()  # (x y z w) : (q1 q2 q3 q0)
        self.quat_vec_cur = R.from_quat(self.quat_vec_cur_inv).inv().as_quat()
        self.ati_vec_cur = euler2vector(roll, pitch, yaw)
        self.ati_vec_cur = self.ati_vec_cur / np.linalg.norm(self.ati_vec_cur)

    # print("quat cur: ", self.quat_vec_cur_inv)

    def vec_quat_des(self, vec_des):
        """
		获得期望向量+滚转的四元数
		:param vec_des: NED系下的期望向量
		"""
        if np.linalg.norm(vec_des) < 0.01:
            self.ati_vec_des[:] = 0
            return
        else:
            self.ati_vec_des[:] = vec_des / np.linalg.norm(vec_des)
        # 第一次旋转：俯仰，偏航
        pitch_d = np.arctan2(-vec_des[2], np.linalg.norm(vec_des[0:2]))
        yaw_d = np.arctan2(vec_des[1], vec_des[0])
        quat_des_1_inv = euler2quat(0, pitch_d, yaw_d)  # 第一次旋转的逆
        # print("quat_des_1_inv: ", quat_des_1_inv)

        # 第二次旋转：加上滚转
        r_axis_1 = np.cross(self.ati_vec_cur, self.ati_vec_des)  # 当前向量转到期望向量的转轴
        if np.linalg.norm(r_axis_1) > 0.001:
            ey_d = r_axis_1 / np.linalg.norm(r_axis_1)  # 期望机体y轴在NED的单位向量
        else:
            ey_d = R.from_quat(quat_des_1_inv).as_matrix()[:, 1]

        self.theta_err = vector_angle(self.ati_vec_cur, self.ati_vec_des)  # 当前向量与期望向量的空间角误差

        ey_1 = R.from_quat(quat_des_1_inv).as_matrix()[:, 1]  # 第一次旋转后的机体y轴单位向量

        r_axis_2 = np.cross(ey_1, ey_d)
        if np.linalg.norm(r_axis_2) > 0.01:
            r_axis_2 = r_axis_2 / np.linalg.norm(r_axis_2)  # 第二次旋转的转轴, 但实际转轴为期望机头向量
        angel_r2 = vector_angle(ey_1, ey_d) * np.sign(np.dot(r_axis_2, self.ati_vec_des))

        # print("roll err 1:", angel_r2)

        # print("r2: ", r_axis_2, "angel_r2: ", np.deg2rad(angel_r2))
        # print("quat_des_2: ", quat_des_2)

        # 复合两次旋转 ： (R2R1)^T = R1^T * R2^T
        rotate_x_2 = R.from_euler('X', [-angel_r2], degrees=True)
        rotate_x_2_inv = rotate_x_2.inv()
        rotate_1_inv = R.from_quat(quat_des_1_inv)
        self.quat_vec_des_inv = R.from_matrix(np.matmul(rotate_1_inv.as_matrix(), rotate_x_2_inv.as_matrix())).as_quat()
        self.quat_vec_des_inv = np.squeeze(self.quat_vec_des_inv)

    def get_cmd(self):
        # 当前与期望 机体y轴的单位向量（NED）
        ey_cur = R.from_quat(self.quat_vec_cur_inv).as_matrix()[:, 1]
        ey_des = R.from_quat(self.quat_vec_des_inv).as_matrix()[:, 1]

        # 分离滚转四元数
        rx = np.cross(ey_cur, ey_des)  # 滚转转轴，但实际转轴是当前机头向量
        if np.linalg.norm(rx) > 0.01:
            rx = rx / np.linalg.norm(rx)
        roll_angle = vector_angle(ey_cur, ey_des) * np.sign(np.around(100000000 * np.dot(rx, self.ati_vec_cur)))  # 滚转转角

        # print("roll err2: ", roll_angle)
        # omega = K_W * (roll_angle) + K_W_D * (0 - wx)
        # 在滚转误差较大时，抑制俯仰（法向过载）
        # 抑制比例
        # k_scale = (180 - K_R * abs(roll_angle)) / 180
        # print("test: ", roll_angle)
        # 当俯仰误差较小，滚转误差接近180，利用负过载改变机头指向
        pitch_angle = self.theta_err

        # load = k_scale * K_N * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8 # 可考虑加入导数项
        # load = k_scale * K_N * vel * pitch_angle / 9.8
        # 考虑加重补？
        return pitch_angle, roll_angle

    def get_err(self, roll, pitch, yaw, vec_des):
        """
		:param vel: 战斗机速度
		:param wx: 战斗机滚转角速度
		:param roll: 战斗机滚转 deg
		:param pitch: 战斗机俯仰 deg
		:param yaw: 战斗机偏航 deg
		:param vec_des: 期望机头指向
		:return: 法向过载、滚转速率
		"""
        self.vec_quat_cur(roll, pitch, yaw)
        self.vec_quat_des(vec_des)
        pitch_angle, roll_angle = self.get_cmd()
        return pitch_angle, roll_angle

    def ctrl_cmd(self, Ma, vel, roll, pitch, yaw, vec_des, dt):
        """
        :param Ma: 马赫数
        :param vel: 真空速
        :param roll: 战斗机滚转 deg
        :param pitch: 战斗机俯仰 deg
        :param yaw: 战斗机偏航 deg
        :param vec_des: 期望机头指向
        :param dt: 仿真步长
        :return: 法向过载、滚转速率
        """
        self.vec_quat_cur(roll, pitch, yaw)
        self.vec_quat_des(vec_des)
        pitch_angle, roll_angle = self.get_cmd()
        # print(f"roll dot: {roll_vec_dot}, pitch dot: {pitch_vec_dot}")
        omega = K_W * (roll_angle) + K_W_D * ((roll_angle - self.last_roll_err) / dt)
        # 滚转死区
        if (pitch_angle < 2) and (roll_angle > 30):
            omega = 0
        # print(f"roll angle: {roll_angle}, roll_dot: {np.rad2deg(roll_vec_dot)}, "
        # 	  f"pitch angel: {pitch_angle}, pitch_dot: {np.rad2deg(pitch_vec_dot)}")
        # 抑制比例
        k_scale = (180 - K_R * abs(roll_angle)) / 180
        load = K_N * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        if (Ma > 0.95):
            load = K_N * 0.6 * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        elif (Ma > 1.4):
            load = K_N * 0.2 * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        load *= k_scale

        # 偏航
        coordinate_trans = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).inv()
        trd_body_los = np.matmul(coordinate_trans.as_matrix(), vec_des)
        # print(f"body los: {trd_body_los}")
        body_yaw_des = np.rad2deg(np.arctan2(trd_body_los[1], trd_body_los[0]))
        body_pitch_des = np.rad2deg(-np.arctan2(trd_body_los[2], np.linalg.norm(trd_body_los[0:2])))
        if (pitch_angle > 120) and (body_yaw_des > 90) and (roll_angle > 40):
            load = 1
        # print(f"body_yaw_des: {body_yaw_des}")
        rudder = K_Y * np.deg2rad(body_yaw_des)
        if abs(rudder) > 1:
            rudder = np.sign(rudder)
        # 油门
        thrust = 0.5 + K_T * (0.9 - Ma)
        if thrust < 0.05:
            thrust = 0.05
        elif thrust > 1:
            thrust = 1
        self.last_roll_err = roll_angle
        return thrust, load, omega, rudder


class Observation(object):
    def __init__(self):
        self.last_body_yaw_r = 0
        self.last_body_pitch_r = 0
        self.last_body_yaw_b = 0
        self.last_body_pitch_b = 0
        self.ati_btt_guide = VecDotCtrl()

    def get_obs(self, fighter, env, delta_tiime):
        """

        Args:
            fighter: 环境中的战斗机对象
            env: 环境对象
            delta_tiime: 两次决策（调用函数）之间的仿真时间间隔

        Returns:

        """
        obs = {"DNN": []}

        for f in env.world.fighters:
            if f == env.world.fighters:
                continue
            else:
                target = f

        if fighter.side == 0 :
            target = env.world.fighters[1]
        else:
            target = env.world.fighters[0]

        l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                                   fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
        fighter.ati_vec_des = np.array([l_n, l_e, l_d])
        ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
        target_ati_vec = euler2vector(target.fc_data.fRollAngle, target.fc_data.fPitchAngle,
                                      target.fc_data.fYawAngle)
        ATA = vector_angle(ati_vec, fighter.ati_vec_des)
        AOT = vector_angle(target_ati_vec, fighter.ati_vec_des)
        dist = np.linalg.norm(fighter.ati_vec_des)

        # 高度
        alt_self = [0.000105 * fighter.fc_data.fAltitude if fighter.fc_data.fAltitude < 2000
                    else logistic(fighter.fc_data.fAltitude, 0.00053, 4500)]
        # 马赫数速度差
        delta_ma = [0.9 - fighter.fc_data.fMachNumber]
        # 指示空速
        ias_self = [fighter.fc_data.fIndicatedAirSpeed / 450]
        # 剩余油量
        fuel_self = [fighter.fc_data.fNumberofFuel / 3200]
        # 过载
        load_self = [fighter.fc_data.fNormalLoad / 9 if fighter.fc_data.fNormalLoad >= 0
                     else fighter.fc_data.fNormalLoad / 3]
        # 滚转速率
        omega_self = [fighter.fc_data.fRollRate / 300]

        # 距离
        dist_in = [2 * logistic(dist, 0.00076, 0) - 1]
        # 接近率
        vel = np.array(
            [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
        vel_t = np.array(
            [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
        if dist > 1:
            dist_dot = np.linalg.norm(vel_t) * \
                       (vel_t.dot(fighter.ati_vec_des) / (np.linalg.norm(vel_t) * dist)) - \
                       np.linalg.norm(vel) * \
                       (vel.dot(fighter.ati_vec_des) / (np.linalg.norm(vel) * dist))
        else:
            dist_dot = 0
        d_dot_in = [2 * logistic(dist_dot, 0.009, 0) - 1]
        # ATA
        ata_in = [ATA / 180]
        # AOT
        aot_in = [AOT / 180]
        # 期望矢量的在NED下的俯仰偏航
        los_pitch = np.rad2deg(np.arctan2(-fighter.ati_vec_des[2], np.linalg.norm(fighter.ati_vec_des[0:2])))
        los_yaw = np.rad2deg(np.arctan2(fighter.ati_vec_des[1], fighter.ati_vec_des[0]))
        ned_los = [los_pitch / 90, los_yaw / 180]
        # 期望矢量在机体系下的俯仰和偏航
        coordinate_t = R.from_euler('ZYX', [fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                                            fighter.fc_data.fRollAngle], degrees=True).inv()
        body_los = np.matmul(coordinate_t.as_matrix(), fighter.ati_vec_des)
        # print(f"body los: {trd_body_los}")
        body_yaw = np.rad2deg(np.arctan2(body_los[1], body_los[0]))
        body_pitch = np.rad2deg(-np.arctan2(body_los[2], np.linalg.norm(body_los[0:2])))
        hmd_los = [body_pitch / 90, body_yaw / 180]
        # 滚转误差
        pitch_err, roll_err = self.ati_btt_guide.get_err(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                                         fighter.fc_data.fYawAngle, fighter.ati_vec_des)
        roll_in = [roll_err / 180]
        # 机体系下的视线角速度
        e_body_vel = np.array([fighter.fc_data.fLongitudianlVelocity, fighter.fc_data.fLateralVelocity,
                               - fighter.fc_data.fNormalVelocity]) / (np.linalg.norm(vel) + 0.001)  # 注意测试一下法向的正负
        e_body_vel_t = np.matmul(coordinate_t.as_matrix(), vel_t) / (np.linalg.norm(vel_t) + 0.001)
        e_body_los = body_los / dist if dist > 0.1 else body_los / 0.1
        los_omega_I_in_body = (1 / (dist + 0.001)) * (np.linalg.norm(vel_t) * np.cross(e_body_los, e_body_vel_t) -
                                                      np.linalg.norm(vel) * np.cross(e_body_los, e_body_vel))
        omega_body = np.array([np.deg2rad(fighter.fc_data.fRollRate), np.deg2rad(fighter.fc_data.fPitchRate),
                               np.deg2rad(fighter.fc_data.fYawRate)])
        los_omega_body = los_omega_I_in_body - omega_body
        # body_yaw_dot = los_omega_body[2]
        body_yaw_dot = (body_yaw - self.last_body_yaw_b) / delta_tiime if fighter.side \
            else (body_yaw - self.last_body_yaw_r) / delta_tiime
        body_pitch_dot = (-np.sin(np.deg2rad(body_yaw))) * los_omega_body[0] + np.cos(np.deg2rad(body_yaw)) * \
                         los_omega_body[1]
        body_pitch_dot = np.rad2deg(body_pitch_dot)
        body_los_dot_in = [2 * logistic(body_yaw_dot, 0.036, 0) - 1, 2 * logistic(body_pitch_dot, 0.036, 0) - 1]

        if fighter.side == 0:
            self.last_body_pitch_r = body_pitch
            self.last_body_yaw_r = body_yaw
        else:
            self.last_body_pitch_b = body_pitch
            self.last_body_yaw_b = body_yaw

        obs_in = alt_self + delta_ma + ias_self + fuel_self + load_self + omega_self + dist_in + d_dot_in + ata_in + \
                 aot_in + ned_los + hmd_los + roll_in + body_los_dot_in

        obs["DNN"] += obs_in
        obs["DNN"] = np.array(obs["DNN"])
        # print(obs)
        return obs


if __name__ == '__main__':
    # -----初始化调用-------#
    # 构建观测对象
    obs_agent = Observation()
    # 构建bot对象
    bot = VecDotCtrl()
    # 构建模型结构
    policy = Policy(17, 5, hidden_sizes=(128, 128, 128, 128, 128, 128), activation=nn.LeakyReLU)
    # 加载模型参数
    # policy.mu_net = torch.load("E:\\ReinforcementLearning\\ppo_4.1.0\\CC_DRL_Torch\\experiment\\test\\results\\" +
    #                            "paper_2.1_test\\paper_2.1_test_s8\\pyt_save\\pi_net_0.pt")
    # torch.save(policy.state_dict(), "pi_state_dict.pt")
    policy.load_state_dict(torch.load("pi_state_dict.pt"))
    policy.eval()

    # -----决策步调用-------#
    # 注意fighter的control mode 应为0
    # 决策步长0.1s
    obs = obs_agent.get_obs(fighter, env, 0.1)['DNN']
    obs = torch.as_tensor(obs, dtype=torch.float32)
    # obs = torch.zeros(17, dtype=torch.float32)
    action = policy(obs)
    load = 9 * action[0] if action[0] > 0 else (3 * action[0])
    load = min(9, max(-3, load))  # 法向过载 -3 ~ 9
    omega = min(300, max(-300, 300 * action[1]))  # 滚转速率 -300 ~ 300
    rudder = min(1., max(-1., action[2]))  # 方向舵 -1 ~ 1
    thrust = min(1., max(0.1, action[3]))  # 油门 0 ~ 1
    print(action)

    # 高度保护
    load_alt, omega_alt, rudder_alt = 0, 0, 0
    if (fighter.fc_data.fAltitude < 500 + 5 * fighter.fc_data.fVerticalVelocity) and \
            (fighter.fc_data.fVerticalVelocity > 0):
        now_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
        safe_vec = [now_vec[0], now_vec[1], -0.5]
        thrust, load_alt, omega_alt, rudder_alt = bot.ctrl_cmd(fighter.fc_data.fMachNumber,
                                                               fighter.fc_data.fTrueAirSpeed,
                                                               fighter.fc_data.fRollAngle,
                                                               fighter.fc_data.fPitchAngle,
                                                               fighter.fc_data.fYawAngle, safe_vec,
                                                               0.01)
    load += 0.6 * load_alt
    omega += 0.6 * omega_alt
    rudder += 0.6 * rudder_alt

    # 输入给战斗机指令的形式
    # fighter.action.u[0] = thrust
    # fighter.action.u[1] = (load / 9) if load > 0 else (load / 3)
    # fighter.action.u[2] = omega / 300
    # fighter.action.u[3] = rudder
