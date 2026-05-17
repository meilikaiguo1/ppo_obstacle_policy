from WVRENV_PHD.basic.ObstacleModel import get_elevation
from WVRENV_PHD.utils.GNCData import euler2vector, ned_to_body, wgs84ToNED, vector_angle
from fighter_bot import VecDotCtrl
from observation.observation import logistic
import numpy as np


def compute_target_distance(env, des_ned):
    """
    计算当前飞机到目标点的欧氏距离
    """
    fighter_ned = np.asarray(env.world.fighters[0].state.ned_Pos, dtype=np.float32)
    des_ned = np.asarray(des_ned, dtype=np.float32)
    return float(np.linalg.norm(des_ned - fighter_ned))


ati_btt_guide = VecDotCtrl()


def reward_func(env, prev_target_bloods ,terminal):
    """
     奖励函数

     参数
     ----------
     env : 仿真环境
     des_ned : 目标点的 NED 坐标
     prev_dist : 上一步到目标点的距离
     init_dist : 当前回合初始距离，用于距离归一化
     min_dis : 当前最近障碍物距离
     terminal : 终止类型
         0 -> 超时
         1 -> 坠地
         2 -> 碰撞
         3 -> 到达目标
         4 -> 失速
         其他 -> 未终止

     返回
     ----------
     reward : 标量总奖励
     reward_info : 各项奖励组成
     cur_dist : 当前到目标点距离
     """

    fighter = env.world.fighters[0]
    target = env.world.fighters[1]

    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
    los = np.array([l_n, l_e, l_d])
    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
    target_ati_vec = euler2vector(target.fc_data.fRollAngle, target.fc_data.fPitchAngle, target.fc_data.fYawAngle)
    ATA = vector_angle(ati_vec, los)
    AA = vector_angle(target_ati_vec, los)
    dist = np.linalg.norm(los)

    vel = np.array(
        [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
    vel_t = np.array(
        [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
    if dist > 1:
        dist_dot = np.linalg.norm(vel_t) * \
                   (vel_t.dot(los) / (np.linalg.norm(vel_t) * dist)) - \
                   np.linalg.norm(vel) * \
                   (vel.dot(los) / (np.linalg.norm(vel) * dist))
    else:
        dist_dot = 0


    #相对位置奖励
    r_rel_pos = 0.2 * ((ATA / 180 - 2) * 0.5 * (-logistic(AA / 180, 18, 1 / 6) + logistic(AA / 180, 18, 5 / 6))
                       - ATA / 180 - 1)

    #瞄准奖励
    pitch_err, roll_err = ati_btt_guide.get_err(
        fighter.fc_data.fRollAngle,
        fighter.fc_data.fPitchAngle,
        fighter.fc_data.fYawAngle,
        los,
    )
    los_vec_body = ned_to_body(los, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                               fighter.fc_data.fRollAngle)
    los_pitch_body = np.rad2deg(-np.arctan2(los_vec_body[2], np.linalg.norm(los_vec_body[0:2])))
    los_yaw_body = np.rad2deg(np.arctan2(los_vec_body[1], los_vec_body[0]))
    r_los_body = - 0.9 * (0.2 * ((los_yaw_body / 180) ** 2) + 1.2 * ((pitch_err / 180) ** 2) + 0.2 * ((roll_err / 180) ** 2))

    #接近率奖励
    r_closure = logistic(dist_dot, 0.016, -50) * (- logistic(dist, 0.0029, 7000))

    # 低速 + 大攻角惩罚（防止失速）
    r_ma_alpha = 1. * ((2 * logistic(fighter.fc_data.fMachNumber, 12., 0.3) - 2) *
                       logistic(fighter.fc_data.fAttackAngle, 0.86, 28))

    #姿态/角速率稳定性惩罚
    pitch_rate_risk = logistic(abs(fighter.fc_data.fPitchRate), 0.15, 25.0)
    roll_rate_risk = logistic(abs(fighter.fc_data.fRollRate), 0.032, 75.0) - 0.08
    normal_load_risk = logistic(abs(fighter.fc_data.fNormalLoad), 1.0, 7.0)
    r_stability = - 0.6 * (
        0.5 * pitch_rate_risk +
        0.6 * roll_rate_risk +
        0.4 * normal_load_risk
    )

    #命中敌机奖励
    if prev_target_bloods != target.combat_data.bloods:
        r_fire = 5 +  2.5 * (prev_target_bloods - target.combat_data.bloods)
    else:
        r_fire = 0

    #降低血量惩罚
    r_bloods = - 2 * ((3 - fighter.combat_data.bloods) / 3)

    #低空俯冲惩罚
    alt_rel = fighter.fc_data.fAltitude - get_elevation(
        fighter.fc_data.fLatitude,
        fighter.fc_data.fLongitude,
    )
    r_alt = - 2 * (1 - logistic(alt_rel, 0.009, 0)) * logistic(
        fighter.fc_data.fVerticalVelocity,
        0.18,
        40,
    )

    #终止事件奖励/惩罚
    red_win_reward = 0
    blue_win_reward = 0
    red_fall_reward = 0
    blue_fall_reward = 0
    draw_reward = 0

    if terminal == 0:
        if env.world.fighters[0].combat_data.bloods > env.world.fighters[1].combat_data.bloods:
            red_win_reward += 50
        elif env.world.fighters[0].combat_data.bloods < env.world.fighters[1].combat_data.bloods:
            blue_win_reward -= 50
        else:
            draw_reward += 20

    if terminal == 1:
        red_win_reward += 100
    if terminal == 2:
        blue_win_reward -= 100
    if terminal == 3:
        draw_reward += 20
    if terminal == 4:
        blue_win_reward -= 50
        red_fall_reward -= 50
    if terminal == 5:
        red_win_reward += 50
        blue_fall_reward += 50

    #奖励明细，便于记录分析
    reward_info = {
        "r_los_body": r_los_body,
        "r_rel_pos": r_rel_pos,
        "r_closure": r_closure,
        "r_ma_alpha": r_ma_alpha,
        "r_stability": r_stability,
        "r_fire": r_fire,
        "r_bloods": r_bloods,
        "r_alt": r_alt,
        "red_win_reward": red_win_reward,
        "blue_win_reward": blue_win_reward,
        "red_fall_reward": red_fall_reward,
        "blue_fall_reward": blue_fall_reward,
        "draw_reward":draw_reward,
    }
    reward = (
        r_los_body + r_rel_pos + r_closure + red_win_reward + blue_win_reward + r_alt +
        red_fall_reward + blue_fall_reward + draw_reward + r_ma_alpha + r_stability + r_fire + r_bloods
    )
    return reward, reward_info
