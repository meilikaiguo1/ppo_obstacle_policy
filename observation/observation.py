import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from WVRENV_PHD.basic.ObstacleModel import get_elevation
from WVRENV_PHD.utils.GNCData import wgs84ToNED, euler2vector, vector_angle, ned_to_body



def discrete_alert(x, num_bin):
    if x > 0.999:
        xd = num_bin
    elif x < -0.999:
        xd = 1
    else:
        xd = int(np.floor((x - (-1)) / (2 / num_bin))) + 1
    return xd


def logistic(x, growth_rate, mid_point):
    if (-growth_rate * (x - mid_point)) > 700:
        S = 0
    else:
        S = 1 / (1 + np.exp(-growth_rate * (x - mid_point)))
    return S

def compute_min_obstacle_distance(env, terrain_hits=None, default_far=5000.0):
    """
    返回飞机到最近障碍点的欧氏距离
    terrain_hits: terrain_scan 返回的点云，shape [N,3]，NED坐标
    """
    fighter_ned = np.asarray(env.world.fighters[0].state.ned_Pos, dtype=np.float32)

    if terrain_hits is None:
        return default_far

    terrain_hits = np.asarray(terrain_hits, dtype=np.float32)
    if terrain_hits.size == 0:
        return default_far

    terrain_hits = terrain_hits.reshape(-1, 3)
    dists = np.linalg.norm(terrain_hits - fighter_ned.reshape(1, 3), axis=1)
    return float(np.min(dists))


def get_avoidance_obs(env, fighter, des_ned):

    # 计算距离终点的距离，角度
    fighter_ned = fighter.state.ned_Pos
    los_ned = des_ned - fighter_ned
    los_body = ned_to_body(los_ned, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle, \
                           fighter.fc_data.fRollAngle)
    distance = np.linalg.norm(los_ned)
    los_pitch_body = -np.arctan2(los_body[2], np.linalg.norm(los_body[0:2]))
    los_yaw_body = np.arctan2(los_body[1], los_body[0])

    # 目标接近率
    vel = np.array([fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, \
                    fighter.fc_data.fVerticalVelocity])
    if distance > 1:
        dist_dot = np.dot(vel, los_ned / distance)
    else:
        dist_dot = 0
    # ATA
    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, \
                           fighter.fc_data.fYawAngle)
    ATA = vector_angle(ati_vec, los_ned)

    #本机相对高度
    rel_alt = fighter.fc_data.fAltitude - get_elevation(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude)

    obs_dict = {
        "distance": distance, #距离目标点距离
        "sin_los_pitch_body": np.sin(los_pitch_body), #距离目标点视线俯仰角
        "cos_los_pitch_body":np.cos(los_pitch_body),
        "sin_los_yaw_body": np.sin(los_yaw_body), # #距离目标点视线视线方位角
        "cos_los_yaw_body": np.cos(los_yaw_body),
        "dist_dot": dist_dot, # 距离目标点接近率
        "ATA": ATA, #距离目标点ATA角
        "rel_alt": rel_alt, #相对高度
        "ma":fighter.fc_data.fMachNumber, # 马赫数
        "pitch": fighter.fc_data.fPitchAngle, #俯仰角
        "roll": fighter.fc_data.fRollAngle, # 滚转角
        'v_pitch': fighter.fc_data.fPathPitchAngle, #爬升角
        'alpha': fighter.fc_data.fAttackAngle, #攻角
        'beta': fighter.fc_data.fSideslipAngle, #侧滑角
        'pitch_rate': fighter.fc_data.fPitchRate, #俯仰速度
        'roll_rate': fighter.fc_data.fRollRate, #滚转速率
        'yaw_rate': fighter.fc_data.fYawRate, #偏航速率
        "normal_load": fighter.fc_data.fNormalLoad, #法向过载
        "lateral_load": fighter.fc_data.fLateralLoad, #侧向过载
        "longitude_load": fighter.fc_data.fLongitudeinalLoad, # 机体纵向过载
    }
    obs_norm = normalized_avoidance_obs(obs_dict)
    obs_self_vec = avoidance_obs_dict_to_array(obs_norm)

    # terrain_hits = env.terrain_sensor.terrain_scan(fighter)
    # terrain_voxels = env.voxel.terrain_to_voxel(fighter, terrain_hits)
    # _, normal_voxels = env.voxel.compute_voxel_local_coordinates(terrain_voxels["voxel_dict"])

    return obs_self_vec

def get_dogfight_obs(env, fighter, target):

    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,\
                       fighter.fc_data.fYawAngle)


    ###--------------------------------- 敌机相关状态量
    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
    t_los_ned = np.array([l_n, l_e, l_d])

    #机体系下目标距离， 高低角， 方位角
    t_los_body = ned_to_body(t_los_ned, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,\
                           fighter.fc_data.fRollAngle)

    t_los_yaw_body = np.rad2deg(np.arctan2(t_los_body[1], t_los_body[0]))
    t_los_pitch_body = np.rad2deg(-np.arctan2(t_los_body[2], np.linalg.norm(t_los_body[0:2])))
    t_distance = np.linalg.norm(t_los_body)

    # 目标接近率
    vel = np.array(
        [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
    vel_t = np.array(
        [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
    if t_distance > 1:
        t_dist_dot = np.linalg.norm(vel_t) * \
                   (vel_t.dot(t_los_ned) / (np.linalg.norm(vel_t) * t_distance)) - \
                   np.linalg.norm(vel) * \
                   (vel.dot(t_los_ned) / (np.linalg.norm(vel) * t_distance))
    else:
        t_dist_dot = 0

    # 目标三维视线偏置角
    t_ATA = vector_angle(ati_vec, t_los_ned)

    #目标爬升角
    t_v_pitch = target.fc_data.fPathPitchAngle

    #目标马赫数
    t_ma = target.fc_data.fMachNumber

    # 目标水平面内的进入角 (-180~180, 目标速度相对视线向右偏为正)
    AA_hori = vector_angle([l_n, l_e, 0], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, 0])
    AA_hori *= np.sign(np.cross([l_n, l_e], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity]))

    #本机相对高度
    rel_alt = fighter.fc_data.fAltitude - get_elevation(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude)

    obs_dict = {
        "rel_alt": rel_alt, #相对高度
        "ma":fighter.fc_data.fMachNumber, # 马赫数
        "pitch": fighter.fc_data.fPitchAngle, #俯仰角
        "roll": fighter.fc_data.fRollAngle, # 滚转角
        'v_pitch': fighter.fc_data.fPathPitchAngle, #爬升角
        'alpha': fighter.fc_data.fAttackAngle, #攻角
        'beta': fighter.fc_data.fSideslipAngle, #侧滑角
        'pitch_rate': fighter.fc_data.fPitchRate, #俯仰速度
        'roll_rate': fighter.fc_data.fRollRate, #滚转速率
        'yaw_rate': fighter.fc_data.fYawRate, #偏航速率
        "normal_load": fighter.fc_data.fNormalLoad, #法向过载
        "lateral_load": fighter.fc_data.fLateralLoad, #侧向过载
        "longitude_load": fighter.fc_data.fLongitudeinalLoad, # 机体纵向过载
        "self_bloods": fighter.combat_data.bloods,
    }
    t_obs_dict = {
        "t_distance": t_distance,
        "los_yaw_body": t_los_yaw_body,
        "los_pitch_body": t_los_pitch_body,
        "t_dist_dot": t_dist_dot,
        "t_ATA": t_ATA,
        "t_v_pitch": t_v_pitch,
        "t_ma": t_ma,
        "AA_hori": AA_hori,
        "t_bloods": target.combat_data.bloods,
    }
    obs_norm, t_obs_norm = normalized_dogfight_obs(obs_dict, t_obs_dict)
    obs_self_vec, obs_t_vec = dogfight_obs_dict_to_array(obs_norm, t_obs_norm)

    terrain_hits = env.terrain_sensor.terrain_scan(fighter)
    terrain_voxels = env.voxel.terrain_to_voxel(fighter, terrain_hits)
    _, normal_voxels = env.voxel.compute_voxel_local_coordinates(terrain_voxels["voxel_dict"])
    return obs_self_vec, obs_t_vec, normal_voxels


def dogfight_obs_dict_to_array(obs_norm, t_obs_norm):
    """
    将归一化后的观测字典按固定顺序转换为 numpy 向量
    返回: np.ndarray, shape = [state_dim], dtype = np.float32
    """
    obs_keys = [
        "rel_alt",
        "ma",
        "pitch",
        "roll",
        "v_pitch",
        "alpha",
        "beta",
        "pitch_rate",
        "roll_rate",
        "yaw_rate",
        "normal_load",
        "lateral_load",
        "longitude_load",
        "self_bloods",
    ]
    t_obs_keys = [
        "t_distance",
        "los_yaw_body",
        "los_pitch_body",
        "t_dist_dot",
        "t_ATA",
        "t_v_pitch",
        "t_ma",
        "AA_hori",
        "t_bloods",
    ]

    obs_vec = np.array([obs_norm[k] for k in obs_keys], dtype=np.float32)
    t_obs_vec = np.array([t_obs_norm[k] for k in t_obs_keys], dtype=np.float32)
    return obs_vec, t_obs_vec

def avoidance_obs_dict_to_array(obs_norm):
    """
    将归一化后的观测字典按固定顺序转换为 numpy 向量
    返回: np.ndarray, shape = [state_dim], dtype = np.float32
    """
    obs_keys = [
        "distance",
        "sin_los_pitch_body",
        "cos_los_pitch_body",
        "sin_los_yaw_body",
        "cos_los_yaw_body",
        "dist_dot",
        "ATA",
        "rel_alt",
        "ma",
        "pitch",
        "roll",
        "v_pitch",
        "alpha",
        "beta",
        "pitch_rate",
        "roll_rate",
        "yaw_rate",
        "normal_load",
        "lateral_load",
        "longitude_load",
    ]
    obs_vec = np.array([obs_norm[k] for k in obs_keys], dtype=np.float32)
    return obs_vec

def normalized_avoidance_obs(obs):
    for key , value in obs.items():
        if key == "distance":
            obs[key] = np.clip(value / 2000.0, 0.0, 1.0)
        # elif key == "los_pitch_body":
        #     obs[key] = value / 90
        # elif key == "los_yaw_body":
        #     obs[key] = value / 180
        elif key == "dist_dot":
            obs[key] = 2 * logistic(value, 0.009, 0) - 1
        elif key == "ATA":
            obs[key] = value / 180
        elif key == "rel_alt":
            obs[key] = np.clip(value / 5000.0, -1.0, 1.0)
        elif key == "ma":
            obs[key] = logistic(float(value), 7, 0.75)
        elif key == "pitch":
            obs[key] = value / 90
        elif key == "roll":
            obs[key] = value / 180
        elif key == 'v_pitch':
            obs[key] = value / 90
        elif key == 'alpha':
            obs[key] = 2 * logistic(value, 0.073, 0) - 1
        elif key == 'beta':
            obs[key] = 2 * logistic(value, 0.056, 0) - 1
        elif key == 'pitch_rate':
            obs[key] = value / 25
        elif key == 'roll_rate':
            obs[key] = value / 180
        elif key == 'yaw_rate':
            obs[key] = 2 * logistic(value, 0.058, 0) - 1
        elif key == "normal_load":
            obs[key] = value / 9 if value >= 0 else value / 3
        elif key == "lateral_load":
            obs[key] = 2 * logistic(value, 1.1, 0) - 1
        elif key == "longitude_load":
            obs[key] = 2 * logistic(value, 0.56, 0) - 1
    return obs

def normalized_dogfight_obs(obs, t_obs):
    for key, value in obs.items():
        if key == "rel_alt":
            obs[key] = np.clip(value / 5000.0, -1.0, 1.0)
        elif key == "ma":
            obs[key] = logistic(float(value), 7, 0.75)
        elif key == "pitch":
            obs[key] = value / 90
        elif key == "roll":
            obs[key] = value / 180
        elif key == 'v_pitch':
            obs[key] = value / 90
        elif key == 'alpha':
            obs[key] = 2 * logistic(value, 0.073, 0) - 1
        elif key == 'beta':
            obs[key] = 2 * logistic(value, 0.056, 0) - 1
        elif key == 'pitch_rate':
            obs[key] = value / 25
        elif key == 'roll_rate':
            obs[key] = value / 180
        elif key == 'yaw_rate':
            obs[key] = 2 * logistic(value, 0.058, 0) - 1
        elif key == "normal_load":
            obs[key] = value / 9 if value >= 0 else value / 3
        elif key == "lateral_load":
            obs[key] = 2 * logistic(value, 1.1, 0) - 1
        elif key == "longitude_load":
            obs[key] = 2 * logistic(value, 0.56, 0) - 1
        elif key == "self_bloods":
            obs[key] = value / 3.0

    for key, value in t_obs.items():
        if key == "t_distance":
            t_obs[key] = 2 * logistic(value, 0.00076, 0) - 1
        elif key == "los_pitch_body":
            t_obs[key] = value / 90
        elif key == "los_yaw_body":
            t_obs[key] = value / 180
        elif key == "t_dist_dot":
            t_obs[key] = 2 * logistic(value, 0.009, 0) - 1
        elif key == "t_ATA":
            t_obs[key] = value / 180
        elif key == "t_v_pitch":
            t_obs[key] = value / 90
        elif key == "t_ma":
            t_obs[key] = logistic(float(value), 7, 0.75)
        elif key == "AA_hori":
            t_obs[key] = value / 180
        elif key == "t_bloods":
            t_obs[key] = value / 3.0
    return  obs ,t_obs




    
