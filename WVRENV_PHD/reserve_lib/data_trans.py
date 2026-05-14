import numpy as np
from WVRENV_PHD.utils.GNCData import angle_2vec, ned_to_body, wgs84ToNED
import math
import copy


def get_data_self(outdata, world):
    # 本机信息应该每个步长都进行更新
    for i, fighter in enumerate(world.fighters):
        # #################### 本机信息 #######################（始终可以获得）
        # 控制模式状态
        outdata[i].selfdata.control_mode = fighter.control_mode.value
        # 剩余航炮弹药量
        outdata[i].selfdata.left_bullet = fighter.combat_data.left_bullet
        # 剩余空空导弹数量
        outdata[i].selfdata.left_missile = fighter.missiles_left
        # 剩余生命值
        outdata[i].selfdata.left_bloods = fighter.combat_data.bloods
        if outdata[i].selfdata.left_bloods < 0:
            outdata[i].selfdata.left_bloods = 0

        # 北东地加速度
        outdata[i].selfdata.NorthAcceleration = fighter.fc_data.fNorthAcceleration
        outdata[i].selfdata.EastAcceleration = fighter.fc_data.fEastAcceleration
        outdata[i].selfdata.VerticalAcceleration = fighter.fc_data.fVerticalAcceleration
        # 体轴滚转角,俯仰角，偏航角速度
        outdata[i].selfdata.RollRate = fighter.fc_data.fRollRate
        outdata[i].selfdata.PitchRate = fighter.fc_data.fPitchRate
        outdata[i].selfdata.YawRate = fighter.fc_data.fYawRate
        # 体轴法向，侧向，纵向过载
        outdata[i].selfdata.NormalLoad = fighter.fc_data.fNormalLoad
        outdata[i].selfdata.LateralLoad = fighter.fc_data.fLateralLoad
        outdata[i].selfdata.LongitudeinalLoad = fighter.fc_data.fLongitudeinalLoad
        # 北东地速度
        outdata[i].selfdata.NorthVelocity = fighter.fc_data.fNorthVelocity
        outdata[i].selfdata.EastVelocity = fighter.fc_data.fEastVelocity
        outdata[i].selfdata.VerticalVelocity = fighter.fc_data.fVerticalVelocity
        # 体轴法向，侧向，纵向速度
        outdata[i].selfdata.NormalVelocity = fighter.fc_data.fNormalVelocity
        outdata[i].selfdata.LateralVelocity = fighter.fc_data.fLateralVelocity
        outdata[i].selfdata.LongitudianlVelocity = fighter.fc_data.fLongitudianlVelocity
        # 经纬高位置
        outdata[i].selfdata.Longitude = fighter.fc_data.fLongitude
        outdata[i].selfdata.Latitude = fighter.fc_data.fLatitude
        outdata[i].selfdata.Altitude = fighter.fc_data.fAltitude
        # 姿态角
        outdata[i].selfdata.RollAngle = fighter.fc_data.fRollAngle
        outdata[i].selfdata.PitchAngle = fighter.fc_data.fPitchAngle
        outdata[i].selfdata.YawAngle = fighter.fc_data.fYawAngle
        # 攻角侧滑角
        outdata[i].selfdata.AttackAngle = fighter.fc_data.fAttackAngle
        outdata[i].selfdata.SideslipAngle = fighter.fc_data.fSideslipAngle

        # 真空速
        outdata[i].selfdata.TrueAirSpeed = fighter.fc_data.fTrueAirSpeed
        # 指示空速
        outdata[i].selfdata.IndicatedAirSpeed = fighter.fc_data.fIndicatedAirSpeed
        # 地速
        outdata[i].selfdata.GroundSpeed = fighter.fc_data.fGroundSpeed
        # 剩余油量
        outdata[i].selfdata.NumberofFuel = fighter.fc_data.fNumberofFuel
        # 推力
        outdata[i].selfdata.Thrust = fighter.fc_data.fThrust

        # 导弹状态
        if fighter.missiles[0].state == 4:
            outdata[i].selfdata.Missile1State = int(fighter.missiles[0].state - 1)
        else:
            outdata[i].selfdata.Missile1State = int(fighter.missiles[0].state)

        if fighter.missiles[1].state == 4:
            outdata[i].selfdata.Missile2State = int(fighter.missiles[1].state - 1)
        else:
            outdata[i].selfdata.Missile2State = int(fighter.missiles[1].state)



    return outdata


def get_data(outdata, world):
    for i, fighter in enumerate(world.fighters):
        ##################### 机载雷达信息 #######################（需要火控雷达范围内才可）
        if fighter.side == 0:
            friend_index = abs(1 - fighter.index)
            Target1_index = fighter.side + 2
            Target2_index = fighter.side + 3
        elif fighter.side == 1:
            friend_index = abs(5 - fighter.index)
            Target1_index = fighter.side - 1
            Target2_index = fighter.side

        if friend_index in fighter.radar.radar_state_list:
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[friend_index].fc_data.fLatitude, world.fighters[friend_index].fc_data.fLongitude,
                                                                world.fighters[friend_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[friend_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # 友机高低角
            outdata[i].radardata.friend_EleAngle = enemy_pitch
            # 友机方位角
            outdata[i].radardata.friend_AziAngle = enemy_yaw
            # 友机距离
            outdata[i].radardata.friend_Distance = distance
            # 北东地速度
            outdata[i].radardata.friend_NorthVelocity = world.fighters[friend_index].fc_data.fNorthVelocity
            outdata[i].radardata.friend_EastVelocity = world.fighters[friend_index].fc_data.fEastVelocity
            outdata[i].radardata.friend_VerticalVelocity = world.fighters[friend_index].fc_data.fVerticalVelocity
        else:
            outdata[i].radardata.friend_EleAngle = 0
            # 友机方位角
            outdata[i].radardata.friend_AziAngle = 0
            # 友机距离
            outdata[i].radardata.friend_Distance = 0
            # 北东地速度
            outdata[i].radardata.friend_NorthVelocity = 0
            outdata[i].radardata.friend_EastVelocity = 0
            outdata[i].radardata.friend_VerticalVelocity = 0

        # 敌机1编号
        if fighter.side == 0:
            outdata[i].radardata.target1_Index = Target1_index - 2
        else:
            outdata[i].radardata.target1_Index = Target1_index
        if Target1_index in fighter.radar.radar_state_list:
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[Target1_index].fc_data.fLatitude, world.fighters[Target1_index].fc_data.fLongitude,
                                                                world.fighters[Target1_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[Target1_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # 敌机高低角
            outdata[i].radardata.target1_EleAngle = enemy_pitch
            # 敌机方位角
            outdata[i].radardata.target1_AziAngle = enemy_yaw
            # 敌机距离
            outdata[i].radardata.target1_Distance = distance
            # 北东地速度
            outdata[i].radardata.target1_NorthVelocity = world.fighters[Target1_index].fc_data.fNorthVelocity
            outdata[i].radardata.target1_EastVelocity = world.fighters[Target1_index].fc_data.fEastVelocity
            outdata[i].radardata.target1_VerticalVelocity = world.fighters[Target1_index].fc_data.fVerticalVelocity
        else:
            # outdata[i].radardata.target1_Index =0
            # 敌机高低角
            outdata[i].radardata.target1_EleAngle = 0
            # 敌机方位角
            outdata[i].radardata.target1_AziAngle = 0
            # 敌机距离
            outdata[i].radardata.target1_Distance = 0
            # 北东地速度
            outdata[i].radardata.target1_NorthVelocity = 0
            outdata[i].radardata.target1_EastVelocity = 0
            outdata[i].radardata.target1_VerticalVelocity = 0

        if fighter.side == 0:
            outdata[i].radardata.target2_Index = Target2_index - 2
        else:
            outdata[i].radardata.target2_Index = Target2_index

        if Target2_index in fighter.radar.radar_state_list:
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[Target2_index].fc_data.fLatitude, world.fighters[Target2_index].fc_data.fLongitude,
                                                                world.fighters[Target2_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[Target2_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # 敌机1编号
            # outdata[i].radardata.target2_Index = Target2_index
            # 敌机高低角
            outdata[i].radardata.target2_EleAngle = enemy_pitch
            # 敌机方位角
            outdata[i].radardata.target2_AziAngle = enemy_yaw
            # 敌机距离
            outdata[i].radardata.target2_Distance = distance
            # 北东地速度
            outdata[i].radardata.target2_NorthVelocity = world.fighters[Target2_index].fc_data.fNorthVelocity
            outdata[i].radardata.target2_EastVelocity = world.fighters[Target2_index].fc_data.fEastVelocity
            outdata[i].radardata.target2_VerticalVelocity = world.fighters[Target2_index].fc_data.fVerticalVelocity
        else:
            # outdata[i].radardata.target2_Index = 0
            # 敌机高低角
            outdata[i].radardata.target2_EleAngle = 0
            # 敌机方位角
            outdata[i].radardata.target2_AziAngle = 0
            # 敌机距离
            outdata[i].radardata.target2_Distance = 0
            # 北东地速度
            outdata[i].radardata.target2_NorthVelocity = 0
            outdata[i].radardata.target2_EastVelocity = 0
            outdata[i].radardata.target2_VerticalVelocity = 0

        ##################### 近距透明信息 #######################（距离在5KM内）
        # 友机信息
        if friend_index in fighter.radar.radar_situation_list:
            # 解算高低角，方位角，友机距离
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[friend_index].fc_data.fLatitude, world.fighters[friend_index].fc_data.fLongitude,
                                                                world.fighters[friend_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[friend_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # 友机高低角
            outdata[i].closedata.friend_EleAngle = enemy_pitch
            # 友机方位角
            outdata[i].closedata.friend_AziAngle = enemy_yaw
            # 友机距离
            outdata[i].closedata.friend_Distance = distance
        else:
            # 友机高低角
            outdata[i].closedata.friend_EleAngle = 0
            # 友机方位角
            outdata[i].closedata.friend_AziAngle = 0
            # 友机距离
            outdata[i].closedata.friend_Distance = 0

        if fighter.side == 0:
            outdata[i].closedata.target1_Index = Target1_index - 2
        else:
            outdata[i].closedata.target1_Index = Target1_index

        if Target1_index in fighter.radar.radar_situation_list:
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[Target1_index].fc_data.fLatitude, world.fighters[Target1_index].fc_data.fLongitude,
                                                                world.fighters[Target1_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[Target1_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # outdata[i].closedata.target1_Index = Target1_index
            # 敌机高低角
            outdata[i].closedata.target1_EleAngle = enemy_pitch
            # 敌机方位角
            outdata[i].closedata.target1_AziAngle = enemy_yaw
            # 敌机距离
            outdata[i].closedata.target1_Distance = distance
        else:
            # outdata[i].closedata.target1_Index = 0
            # 敌机高低角
            outdata[i].closedata.target1_EleAngle = 0
            # 敌机方位角
            outdata[i].closedata.target1_AziAngle = 0
            # 敌机距离
            outdata[i].closedata.target1_Distance = 0

        if fighter.side == 0:
            outdata[i].closedata.target2_Index = Target2_index - 2
        else:
            outdata[i].closedata.target2_Index = Target2_index

        if Target2_index in fighter.radar.radar_situation_list:
            target_North, target_East, target_Down = wgs84ToNED(world.fighters[Target2_index].fc_data.fLatitude, world.fighters[Target2_index].fc_data.fLongitude,
                                                                world.fighters[Target2_index].fc_data.fAltitude)
            delta_pos_target = np.array([target_North, target_East, target_Down])
            fighter_North, fighter_East, fighter_Down = wgs84ToNED(fighter.fc_data.fLatitude,
                                                                   fighter.fc_data.fLongitude,
                                                                   fighter.fc_data.fAltitude)
            delta_pos_fighter = np.array([fighter_North, fighter_East, fighter_Down])
            delta_pos = delta_pos_target - delta_pos_fighter  # 敌我位置差
            # delta_pos = world.fighters[Target2_index].state.ned_Pos - fighter.state.ned_Pos  # 敌我位置差

            distance = np.sqrt(pow(delta_pos[0], 2) + pow(delta_pos[1], 2) + pow(delta_pos[2], 2))  # 双方距离模值

            roll = fighter.fc_data.fRollAngle
            yaw = fighter.fc_data.fYawAngle
            pitch = fighter.fc_data.fPitchAngle

            vec_body = ned_to_body(delta_pos, yaw, pitch, roll)
            enemy_pitch = np.rad2deg(-math.atan2(vec_body[2], np.linalg.norm(vec_body[0:2])))
            enemy_yaw = np.rad2deg(math.atan2(vec_body[1], vec_body[0]))

            # outdata[i].closedata.target2_Index = Target2_index
            # 敌机高低角
            outdata[i].closedata.target2_EleAngle = enemy_pitch
            # 敌机方位角
            outdata[i].closedata.target2_AziAngle = enemy_yaw
            # 敌机距离
            outdata[i].closedata.target2_Distance = distance
        else:
            # outdata[i].closedata.target2_Index = 0
            # 敌机高低角
            outdata[i].closedata.target2_EleAngle = 0
            # 敌机方位角
            outdata[i].closedata.target2_AziAngle = 0
            # 敌机距离
            outdata[i].closedata.target2_Distance = 0

        ##################### 告警系统信息 #######################
        outdata[i].alertdata.emergency_num = fighter.radar.radar_alert
        outdata[i].alertdata.emergency_EleAngle = fighter.radar.radar_alert_pitch
        outdata[i].alertdata.emergency_AziAngle = fighter.radar.radar_alert_yaw

        outdata[i].alertdata.emergency_missile_num = fighter.radar.alert
        outdata[i].alertdata.emergency_missile_EleAngle = fighter.radar.alert_pitch
        outdata[i].alertdata.emergency_missile_AziAngle = fighter.radar.alert_yaw

        ######################### 通信链路 #########################
        outdata[i].communication = world.fighters[friend_index].communication

        if len(world.fighters[friend_index].communication_mul) >= 10:
            outdata[i].communication = copy.deepcopy(world.fighters[friend_index].communication_mul[0])
            del world.fighters[friend_index].communication_mul[0]
            # print('delete 0 ')
        else:
            outdata[i].communication = [b'\x00', b'\x00', b'\x00', b'\x00', b'\x00']
        # if i == 0:
        #     print('debugggggggggggggggg', i, len(world.fighters[friend_index].communication_mul), outdata[i].communication)

    return outdata


def get_data_state(outdata, world):
    for i, fighter in enumerate(world.fighters):
        if fighter.side == 0:
            friend_index = abs(1 - fighter.index)
            Target1_index = fighter.side + 2
            Target2_index = fighter.side + 3
        elif fighter.side == 1:
            friend_index = abs(5 - fighter.index)
            Target1_index = fighter.side - 1
            Target2_index = fighter.side

        ##################### 态势预警信息 #######################
        outdata[i].statedata.friend_Longitude = world.fighters[friend_index].fc_data.fLongitude
        outdata[i].statedata.friend_Latitude = world.fighters[friend_index].fc_data.fLatitude
        outdata[i].statedata.friend_Altitude = world.fighters[friend_index].fc_data.fAltitude
        outdata[i].statedata.friend_Survive = world.fighters[friend_index].combat_data.survive_info

        # 敌机1编号
        if fighter.side == 0:
            outdata[i].statedata.target1_Index = Target1_index - 2
        else:
            outdata[i].statedata.target1_Index = Target1_index

        # 经纬高位置
        outdata[i].statedata.target1_Longitude = world.fighters[Target1_index].fc_data.fLongitude
        outdata[i].statedata.target1_Latitude = world.fighters[Target1_index].fc_data.fLatitude
        outdata[i].statedata.target1_Altitude = world.fighters[Target1_index].fc_data.fAltitude
        outdata[i].statedata.target1_Survive = world.fighters[Target1_index].combat_data.survive_info

        # 敌机2编号
        # outdata[i].statedata.target2_Index = Target2_index
        if fighter.side == 0:
            outdata[i].statedata.target2_Index = Target2_index - 2
        else:
            outdata[i].statedata.target2_Index = Target2_index

        # 经纬高位置
        outdata[i].statedata.target2_Longitude = world.fighters[Target2_index].fc_data.fLongitude
        outdata[i].statedata.target2_Latitude = world.fighters[Target2_index].fc_data.fLatitude
        outdata[i].statedata.target2_Altitude = world.fighters[Target2_index].fc_data.fAltitude
        outdata[i].statedata.target2_Survive = world.fighters[Target2_index].combat_data.survive_info


    return outdata





