import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from WVRENV_PHD.basic.ObstacleModel import unit, get_elevation, ned_to_wgs84_batch, get_elevation_batch
from WVRENV_PHD.utils.GNCData import wgs84ToNED, ned_to_body, ned_to_wgs84, body_to_ned


class Sensors(object):
    """机载传感器模块"""
    def __init__(self, data_initial):
        # 雷达参数
        self.radar_range = data_initial.radar_range                       # 雷达探测范围
        self.radar_vertical_scan = data_initial.radar_vertical_scan             # 雷达垂直扫描范围
        self.radar_horizontal_scan = data_initial.radar_horizontal_scan           # 雷达水平扫描范围

        self.eodas_range = data_initial.eodas_range                    # 目视范围

        self.alert_missile_range = data_initial.alert_missile_range                 # 来袭导弹告警范围

        # 火控雷达
        self.radar_list = []           # 存储火控雷达探测状态的列表
        self.eodas_list = []      # 存储5KM态势探测状态的列表

        # 导弹告警
        self.alert_missile = 0              # 告警数量
        self.alert_pitch = []       # 弹目俯仰角存储列表
        self.alert_yaw = []         # 弹目方位角存储列表

        # 被敌方飞机锁定告警
        self.alert_radar = 0
        self.alert_radar_pitch = []
        self.alert_radar_yaw = []

    def sensor_reset(self):
        self.radar_list = []           # 存储火控雷达探测状态的列表
        self.eodas_list = []      # 存储5KM态势探测状态的列表

        # 导弹告警
        self.alert_missile = 0              # 告警数量
        self.alert_pitch = []       # 弹目俯仰角存储列表
        self.alert_yaw = []         # 弹目方位角存储列表

        # 被敌方飞机锁定告警
        self.alert_radar = 0
        self.alert_radar_pitch = []
        self.alert_radar_yaw = []

    def fire_control_radar(self, fighter, fighters):
        """
        模拟机载火控雷达，判断是否有敌机满足距离和角度约束
        :param fighter: 本机
        :param fighters: 空域所有飞机
        :return:
        """
        # 列表清空
        fighter.sensors.radar_list.clear()

        # 以遍历的形式依次对所有敌机进行判断
        # 首先判断自己是否存活
        if fighter.combat_data.survive_info:
            # 遍历所有其他飞机
            for j, target in enumerate(fighters):
                # 判断是否为敌机，判断敌机是否存活
                if target.combat_data.survive_info and (fighter.index != target.index):

                    # # 计算目标机的北东地坐标
                    # target_North, target_East, target_Down = wgs84ToNED(target.fc_data.fLatitude,
                    #                                                     target.fc_data.fLongitude,
                    #                                                     target.fc_data.fAltitude)
                    # target_NEDpos = np.array([target_North, target_East, target_Down])
                    # # 计算自己的北东地坐标
                    # fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                    #                                                        fighter.fc_data.fLongitude,
                    #                                                        fighter.fc_data.fAltitude)
                    # fighter_NEDpos = np.array([fighter_North, fighter_East, fighter_Down])

                    # 敌我位置差
                    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                               target.fc_data.fAltitude,
                                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                               fighter.fc_data.fAltitude)
                    # delta_pos = target_NEDpos - fighter_NEDpos  # 一条由自己指向目标的矢量
                    delta_pos = np.array([l_n, l_e, l_d])  # 一条由自己指向目标的矢量

                    # 双方距离模值
                    distance = delta_pos.dot(delta_pos) ** 0.5  # 向量点乘，求得内积，再开方

                    # 体坐标系下的视线矢量
                    roll = fighter.fc_data.fRollAngle
                    yaw = fighter.fc_data.fYawAngle
                    pitch = fighter.fc_data.fPitchAngle
                    vec_body = ned_to_body(delta_pos, yaw, pitch, roll)

                    # 视线矢量在体坐标系下的俯仰角和偏航角
                    enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
                    enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

                    # 最后判断目标是否在记载雷达扫描范围内
                    if distance <= self.radar_range and abs(enemy_pitch) <= self.radar_vertical_scan and abs(enemy_yaw) <= self.radar_horizontal_scan:  # 目标位于雷达探测范围内
                        fighter.sensors.radar_list.append(target.index)
                    else:
                        pass
                else:
                    pass
        else:
            pass

    def Eodas(self, fighter, fighters):
        # 列表清空
        fighter.sensors.eodas_list.clear()

        # 以遍历的形式依次对所有敌机进行判断
        # 首先判断自己是否存活
        if fighter.combat_data.survive_info:
            # 遍历所有其他飞机
            for j, target in enumerate(fighters):
                # 判断是否为敌机，判断敌机是否存活
                if target.combat_data.survive_info and (fighter.index != target.index):

                    # # 计算目标机的北东地坐标
                    # target_North, target_East, target_Down = wgs84ToNED(target.fc_data.fLatitude,
                    #                                                     target.fc_data.fLongitude,
                    #                                                     target.fc_data.fAltitude)
                    # target_NEDpos = np.array([target_North, target_East, target_Down])
                    # # 计算自己的北东地坐标
                    # fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                    #                                                        fighter.fc_data.fLongitude,
                    #                                                        fighter.fc_data.fAltitude)
                    # fighter_NEDpos = np.array([fighter_North, fighter_East, fighter_Down])
                    #
                    # # # 敌我位置差
                    # # delta_pos = target_NEDpos - fighter_NEDpos

                    # 敌我位置差
                    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                               target.fc_data.fAltitude,
                                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                               fighter.fc_data.fAltitude)
                    delta_pos = np.array([l_n, l_e, l_d])  # 一条由自己指向目标的矢量

                    # 双方距离模值
                    distance = delta_pos.dot(delta_pos) ** 0.5  # 向量点乘，求得内积，再开方

                    # 最后判断目标是否在视线范围内
                    if distance <= self.eodas_range:  # 距离5KM内 态势透明
                        fighter.sensors.eodas_list.append(target.index)
                    else:
                        pass
                else:
                    pass
        else:
            pass

    def missile_alert(self, fighter, fighters):
        '''导弹来袭告警'''

        # 清空上一个步长的储存记录
        fighter.sensors.alert_missile = 0              # 告警数量
        fighter.sensors.alert_pitch.clear()       # 弹目俯仰角存储列表
        fighter.sensors.alert_yaw.clear()         # 弹目方位角存储列表

        # 首先判断自己是否存活
        if fighter.combat_data.survive_info:
            for i, other_fighter in enumerate(fighters):         # 循环，在所有战机中判断
                if fighter.index != other_fighter.index:                # 判断，排除自己发射的导弹
                    for j, missile in enumerate(other_fighter.missiles):    # 循环，每枚导弹 单独进行判断
                        if missile.state == 1:                          # 判断，这枚导弹是否处于正常工作中

                            # # 导弹北东地坐标
                            # North, East, Down = wgs84ToNED(lat=missile.dataout.m_latitude,
                            #                                lon=missile.dataout.m_longitude,
                            #                                h=missile.dataout.m_altitude)
                            # missile_nedpos = np.array([North, East, Down])  # 导弹北东地坐标
                            #
                            # # 自己北东地坐标 （相对于导弹是目标）
                            # target_North, target_East, target_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                            #                                                     fighter.fc_data.fLongitude,
                            #                                                     fighter.fc_data.fAltitude)
                            # fighter_nedpos = np.array([target_North, target_East, target_Down])
                            #
                            # # # 敌我位置差
                            # # delta_pos = missile_nedpos - fighter_nedpos  # 一条由自己指向来袭导弹的矢量

                            # 敌我位置差
                            l_n, l_e, l_d = wgs84ToNED(missile.dataout.m_latitude, missile.dataout.m_longitude,
                                                       missile.dataout.m_altitude,
                                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                                       fighter.fc_data.fAltitude)
                            delta_pos = np.array([l_n, l_e, l_d])  # 一条由自己指向目标的矢量

                            # 双方距离模值
                            distance = delta_pos.dot(delta_pos) ** 0.5  # 向量点乘，求得内积，再开方

                            roll = fighter.fc_data.fRollAngle
                            yaw = fighter.fc_data.fYawAngle
                            pitch = fighter.fc_data.fPitchAngle

                            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
                            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
                            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

                            # if distance <= self.alert_missile_range and missile.dataout.time_fire <= 5.23 and missile.dataout.time_fire > 0:
                            if distance <= 2000 and missile.dataout.time_fire > 0:
                                # 距离小于5KM同时发动机工作中
                                fighter.sensors.alert_missile += 1       # 目标机收到告警+1
                                fighter.sensors.alert_pitch.append(enemy_pitch)        # 目标机列表存储来袭导弹高低角
                                fighter.sensors.alert_yaw.append(enemy_yaw)            # 目标机列表存储来袭导弹方位角
                            elif distance <= self.alert_missile_range and missile.dataout.time_fire <= 5.23 and missile.dataout.time_fire > 0:
                                fighter.sensors.alert_missile += 1       # 目标机收到告警+1
                                fighter.sensors.alert_pitch.append(enemy_pitch)        # 目标机列表存储来袭导弹高低角
                                fighter.sensors.alert_yaw.append(enemy_yaw)            # 目标机列表存储来袭导弹方位角

    def radar_alert(self, fighter, fighters):
        '''飞机雷达告警'''

        # 清空上一个步长的储存记录
        fighter.sensors.alert_radar = 0             # 告警数量置零
        fighter.sensors.alert_radar_pitch.clear()   # 列表清零
        fighter.sensors.alert_radar_yaw.clear()     # 列表清零

        # 首先判断自己是否存活
        if fighter.combat_data.survive_info:
            for i, enemy in enumerate(fighters):         # 循环，在所有战机中判断
                # 跳过友机, 且锁定自己的敌机得存活，自身处于敌机火控雷达范围内，且被选择锁定
                if (fighter.side != enemy.side) and enemy.combat_data.survive_info and (fighter.index in enemy.sensors.radar_list) and (fighter.index == enemy.target_index):

                    # # 敌机北东地坐标
                    # target_North, target_East, target_Down = wgs84ToNED(enemy.fc_data.fLatitude,
                    #                                                     enemy.fc_data.fLongitude,
                    #                                                     enemy.fc_data.fAltitude)
                    # target_nedpos = np.array([target_North, target_East, target_Down])
                    #
                    # # 自己北东地坐标
                    # fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                    #                                                        fighter.fc_data.fLongitude,
                    #                                                        fighter.fc_data.fAltitude)
                    # fighter_nedpos = np.array([fighter_North, fighter_East, fighter_Down])
                    #
                    # # # 敌我位置差
                    # # delta_pos = target_nedpos - fighter_nedpos  # 一条由自己指向来袭导弹的矢量

                    # 敌我位置差
                    l_n, l_e, l_d = wgs84ToNED(enemy.fc_data.fLatitude, enemy.fc_data.fLongitude,
                                               enemy.fc_data.fAltitude,
                                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                               fighter.fc_data.fAltitude)
                    # delta_pos = target_NEDpos - fighter_NEDpos  # 一条由自己指向目标的矢量
                    delta_pos = np.array([l_n, l_e, l_d])  # 一条由自己指向目标的矢量

                    # 自身姿态角
                    roll = fighter.fc_data.fRollAngle
                    yaw = fighter.fc_data.fYawAngle
                    pitch = fighter.fc_data.fPitchAngle

                    # 敌机姿态矢量
                    vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
                    enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
                    enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

                    fighter.sensors.alert_radar += 1  # 目标机收到告警+1
                    fighter.sensors.alert_radar_pitch.append(enemy_pitch)  # 目标机列表存储来袭导弹高低角
                    fighter.sensors.alert_radar_yaw.append(enemy_yaw)  # 目标机列表存储来袭导弹方位角


class Terrain_sensor:
    def __init__(self, params):
        """
        params.azimuth_range   = (-30, 30, 1)    # 方位角范围 [min, max, step]，deg，机体坐标系
        params.elevation_range = (-30, 30, 1)    # 俯仰角范围 [min, max, step]，deg，机体坐标系
        params.step_m          = 200.0           # 射线沿弧长的步长（粗扫）
        params.max_range       = 2000.0         # 最大扫描距离
        params.accuracy        = 10.0            # 二分法弧长精度
        params.max_iter        = 20              # 二分法最大迭代次数
        """
        self.azimuth_range = params.azimuth_range
        self.elevation_range = params.elevation_range
        self.step_m = params.step_m
        self.accuracy = params.accuracy
        self.max_range = params.max_range
        self.max_iter = params.max_iter

        #方位/俯仰角的扫描网格
        az_min, az_max, az_step = self.azimuth_range
        el_min, el_max, el_step = self.elevation_range

        az_list = np.arange(az_min, az_max + az_step, az_step)
        el_list = np.arange(el_min, el_max + el_step, el_step)

        self.scan_dirs_body = []
        for el_deg in el_list:
            el = np.deg2rad(el_deg)
            cos_el = np.cos(el)
            sin_el = np.sin(el)

            for az_deg in az_list:
                az = np.deg2rad(az_deg)
                dir_body = np.array([
                    cos_el * np.cos(az),
                    cos_el * np.sin(az),
                    sin_el
                ], dtype=np.float64)
                self.scan_dirs_body.append(dir_body)

    def ray_terrain_intersection_dir(self, fighter, dir_ned):
        """
        :param fighter: 飞行器智能体
        :param dir_ned: NED坐标系下的射线方向
        :return:
            hit: bool
            p_hit_ned: [3] 或 None
            lon_hit, lat_hit, elev_hit
        """
        dir_ned = unit(dir_ned)
        fighter_ned = np.asarray(fighter.state.ned_Pos, dtype=np.float64)

        step_m = self.step_m
        max_range = self.max_range
        accuracy = self.accuracy
        max_iter = self.max_iter

        # -----------------------------
        # Step 1: 粗扫（批量）
        # -----------------------------
        s_vals = np.arange(0.0, max_range + step_m, step_m, dtype=np.float64)

        p_ned_all = np.empty((len(s_vals), 3), dtype=np.float64)
        p_ned_all[:, 0] = fighter_ned[0] + s_vals * dir_ned[0]
        p_ned_all[:, 1] = fighter_ned[1] + s_vals * dir_ned[1]
        p_ned_all[:, 2] = fighter_ned[2] - s_vals * dir_ned[2]

        lon_all, lat_all, alt_all = ned_to_wgs84_batch(p_ned_all)
        elev_all = get_elevation_batch(lat_all, lon_all)
        h_all = alt_all - elev_all

        hit_idx = np.where(h_all <= 0)[0]
        if len(hit_idx) == 0:
            return False, None, None, None, None

        i = hit_idx[0]
        if i == 0:
            s_start, s_end = 0.0, step_m
        else:
            s_start, s_end = s_vals[i - 1], s_vals[i]

        # -----------------------------
        # Step 2: 二分细化（保持原逻辑）
        # -----------------------------
        def sample(s):
            p_ned = np.array([
                fighter_ned[0] + s * dir_ned[0],
                fighter_ned[1] + s * dir_ned[1],
                fighter_ned[2] - s * dir_ned[2]
            ], dtype=np.float64)

            p_lon, p_lat, p_alt = ned_to_wgs84(p_ned)
            elev = get_elevation(p_lat, p_lon)
            h = p_alt - elev
            return h, p_ned, p_lon, p_lat, p_alt, elev

        h_s, p_s, lon_s, lat_s, alt_s, elev_s = sample(s_start)

        p_mid = None
        lon_mid = lat_mid = elev_mid = None

        for _ in range(int(max_iter)):
            mid = 0.5 * (s_start + s_end)
            h_mid, p_mid, lon_mid, lat_mid, alt_mid, elev_mid = sample(mid)

            # 收敛条件：区间足够小，或者高度差接近0
            if (s_end - s_start) < accuracy or abs(h_mid) < 1.0:
                break

            if h_mid * h_s < 0:
                s_end = mid
            else:
                s_start = mid
                h_s = h_mid

        return True, p_mid, lon_mid, lat_mid, elev_mid

    def terrain_scan(self, fighter):
        """
        根据 azimuth_range / elevation_range 扫描机体前方空间
        返回范围内所有与地形相交的障碍点（NED坐标）
        """
        hits = []
        hits_append = hits.append

        yaw_deg = fighter.fc_data.fYawAngle
        pitch_deg = fighter.fc_data.fPitchAngle
        roll_deg = fighter.fc_data.fRollAngle

        for dir_body in self.scan_dirs_body:
            dir_ned = body_to_ned(dir_body, yaw_deg, pitch_deg, roll_deg)

            hit, p_hit_ned, lon_hit, lat_hit, elev_hit = self.ray_terrain_intersection_dir(fighter, dir_ned)
            if hit:
                hits_append(p_hit_ned)

        return np.asarray(hits, dtype=np.float64)

    def plot_terrain_hits_ned(self, hits, fighter):
        """
           hits: [[lon, lat, elev], ...]
           fighter: 当前飞机对象 env.world.fighters[0]

           飞机和障碍点都画在同一个全局 NED 坐标系下，
           不把飞机放到原点，也不画连线。
           """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 飞机真实 NED 坐标
        fighter_ned = np.array(fighter.state.ned_Pos, dtype=np.float64)
        fighter_n = fighter_ned[0]
        fighter_e = fighter_ned[1]
        fighter_u = -fighter_ned[2]  # 显示时用 Up 更直观

        # 先画飞机
        ax.scatter(
            [fighter_n],
            [fighter_e],
            [fighter_u],
            s=120,
            marker='^',
            label='fighter'
        )

        if hits is not None and len(hits) > 0:
            hits = np.array(hits, dtype=np.float64)

            # 经纬高 -> NED
            ned_points = []
            for lat, lon, alt in hits:
                north, east, down = wgs84ToNED(lat, lon, alt)
                ned_points.append([north, east, down])

            ned_points = np.array(ned_points, dtype=np.float64)

            north = ned_points[:, 0]
            east = ned_points[:, 1]
            up = -ned_points[:, 2]

            # 只画障碍点
            ax.scatter(
                north,
                east,
                up,
                s=12,
                marker='o',
                label='terrain hits'
            )

            # 把飞机和障碍点一起纳入坐标范围
            all_n = np.append(north, fighter_n)
            all_e = np.append(east, fighter_e)
            all_u = np.append(up, fighter_u)

            max_range = np.array([
                all_n.max() - all_n.min(),
                all_e.max() - all_e.min(),
                all_u.max() - all_u.min()
            ]).max() / 2.0

            if max_range < 1.0:
                max_range = 1.0

            mid_n = (all_n.max() + all_n.min()) * 0.5
            mid_e = (all_e.max() + all_e.min()) * 0.5
            mid_u = (all_u.max() + all_u.min()) * 0.5

            ax.set_xlim(mid_n - max_range, mid_n + max_range)
            ax.set_ylim(mid_e - max_range, mid_e + max_range)
            ax.set_zlim(mid_u - max_range, mid_u + max_range)

        ax.set_xlabel("North (m)")
        ax.set_ylabel("East (m)")
        ax.set_zlabel("Up (m)")
        ax.set_title("Discrete Terrain Obstacle Points and Fighter")
        ax.legend()

        plt.show()









