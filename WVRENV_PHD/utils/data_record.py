import numpy as np
import csv
import os

from numpy.distutils.misc_util import blue_text

from Spinup.mpi_torch_utils import proc_id
from WVRENV_PHD.SimArg import num_red
from WVRENV_PHD.basic.ObstacleModel import ray_terrain_intersection, get_elevation
from WVRENV_PHD.utils.GNCData import wgs84ToNED, euler2vector



def record_tacview_outside(video_dir, env, epoch, terminal, sim_in_list, proc_id=0):
    """
    以tacview的格式记录数据文件
    :param env: 整个环境输入env
    :return:
    """
    if env.time_count == 0:
        env.acmi_obj = open(video_dir + "\\trained_epoch_" + str(epoch) + "_" + str(proc_id) + ".txt", "w")
        env.acmi_obj.write("FileType=text/acmi/tacview\n")
        env.acmi_obj.write("FileVersion=2.1\n")
        env.acmi_obj.write("0,ReferenceTime=2022-10-01T00:00:00Z\n")
        env.acmi_obj.write("0,Title = test simple aircraft\n")
    else:
        env.acmi_obj.write("#" + str(env.world.dt * env.time_count) + "\n")

        for i, fighter in enumerate(env.world.fighters):
            data = str(fighter.obj_index) + "," + "T=" + str(fighter.fc_data.fLongitude) + "|" + \
                   str(fighter.fc_data.fLatitude) + "|" + \
                   str(fighter.fc_data.fAltitude) + "|" + str(format(fighter.fc_data.fRollAngle, '.3f')) + "|" + \
                   str(format(fighter.fc_data.fPitchAngle, '.3f')) + "|" + str(format(fighter.fc_data.fYawAngle, '.3f'))
            data += ", Type=Air+FixedWing,Coalition=Allies,Color="
            if fighter.side == 0:
                if (fighter.index) % 2 == 0:
                    data += "Red"
                elif (fighter.index) % 2 == 1:
                    data += "Orange"
                else:
                    data += "Red"
            else:
                if (fighter.index - env.num_RedFighter) % 2 == 0:
                    data += "Blue"
                elif (fighter.index - env.num_RedFighter) % 2 == 1:
                    data += "Cyan"
                else:
                    data += "Blue"

            data += ",Name=" + fighter.type + ",Mach=" + str(format(fighter.fc_data.fMachNumber, '.1f'))
            # data += ",RadarMode=1" + ",RadarRange=" + str(fighter.sensors.radar_range) + ",RadarHorizontalBeamwidth=" + str(
            #     2 * fighter.sensors.radar_horizontal_scan) + ",RadarVerticalBeamwidth=" + str(2 * fighter.sensors.radar_vertical_scan)
            data += ",ShortName=" + fighter.type + "  " + str(fighter.index) + "  " + str(
                format(fighter.combat_data.bloods, '.1f'))
            data += "  T=" + str(format(sim_in_list[i].control_input[0], '.2f'))
            data += "  az=" + str(format(sim_in_list[i].control_input[1], '.2f'))
            data += "  w=" + str(format(sim_in_list[i].control_input[2], '.2f'))
            data += "  ay=" + str(format(sim_in_list[i].control_input[3], '.2f'))
            data += "  fire=" + str(sim_in_list[i].missile_fire)
            data += "  tg=" + str(sim_in_list[i].target_index)
            data += "  mispre1=" + str(fighter.missiles[0].launch_prepare_time)
            data += "  mispre2=" + str(fighter.missiles[1].launch_prepare_time)
            data += "  alert=" + str(fighter.sensors.alert_missile)
            data += "\n"
            env.acmi_obj.write(data)

            # 导弹部分的数据写入
            for j in range(int(env.initial_data.missiles_max)):
                if fighter.missiles[j].state == 1:
                    data_missile = str(
                        fighter.obj_index * 100 + fighter.missiles[j].index) + "," + "T=" + \
                                   str(fighter.missiles[j].dataout.m_longitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_latitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_altitude) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_roll)) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_pitch)) + "|" + \
                                   str(np.rad2deg(-fighter.missiles[j].dataout.m_yaw))
                    data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color="
                    if fighter.side == 0:
                        if (fighter.index) % 2 == 0:
                            data_missile += "Red"
                        elif (fighter.index) % 2 == 1:
                            data_missile += "Orange"
                        else:
                            data_missile += "Red"
                    else:
                        if (fighter.index - env.num_RedFighter) % 2 == 0:
                            data_missile += "Blue"
                        elif (fighter.index - env.num_RedFighter) % 2 == 1:
                            data_missile += "Cyan"
                        else:
                            data_missile += "Blue"
                    data_missile += ",Name=" + fighter.missiles[j].type
                    data_missile += ",ShortName=" + fighter.missiles[j].type + "    Tar=" + str(fighter.missiles[j].target_index)

                    # if fighter.side == 1:
                    #     if fighter.index == 2:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color=Red,Name="
                    #     if fighter.index == 3:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color=Orange,Name="
                    # else:
                    #     if fighter.index == 0:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Enemies,Color=Blue,Name="
                    #     if fighter.index == 1:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Enemies,Color=Cyan,Name="
                    data_missile += "  Mach=" + str(format(fighter.missiles[j].dataout.m_ma, '.1f'))
                    data_missile += "  state=" + str(fighter.missiles[j].state)
                    data_missile += "\n"
                    env.acmi_obj.write(data_missile)
        if terminal:
            env.acmi_obj.close()


def record_nn_act_and_des_vec(csv_dir, env, epoch,
                              delta_los_elevation, delta_los_azimuth,
                              los_des_elevation, los_des_azimuth, los_des_elevation_body, los_des_azimuth_body,
                              fighter_normal_load, fighter_roll_rate,
                              proc_id=0):
    """
    记录神经网络映射到视线角变化量后的动作输出，以及期望的NED视线高低角和方位角
    :param env: 整个环境输入env
    :return:
    """
    if env.time_count == 0:
        with open(csv_dir + "\\trained_epoch_" + str(epoch) + "_" + str(proc_id) + "_act.txt", "w") as csvfile:
            writer = csv.writer(csvfile)
            fieldname = ['time', 'delta_elevation', 'delta_azimuth', 'los_elevation', 'los_azimuth',
                         'los_elevation_body', 'los_azimuth_body', 'normal_load', 'roll_rate']
            writer.writerow(fieldname)
    else:
        with open(csv_dir + '\\trained_epoch_' + str(epoch) + "_" + str(proc_id) + "_act.txt", 'a', newline='') as csvfile:
            writer_n = csv.writer(csvfile)
            los_data = [env.time_count,
                        delta_los_elevation, delta_los_azimuth,
                        los_des_elevation, los_des_azimuth,
                        los_des_elevation_body, los_des_azimuth_body,
                        fighter_normal_load, fighter_roll_rate]
            writer_n.writerow(los_data)


def record_tacview(self, pid):
    """
    以tacview的格式记录数据文件
    :param self: 整个环境输入env
    :return:
    """
    if self.time_count == 0:
        self.acmi_obj = open(self.file_dir + "\\epoch_"+ str(pid) + "_" + str(self.epoch) + ".txt", "w")
        self.acmi_obj.write("FileType=text/acmi/tacview\n")
        self.acmi_obj.write("FileVersion=2.1\n")
        self.acmi_obj.write("0,ReferenceTime=2022-10-01T00:00:00Z\n")
        self.acmi_obj.write("0,Title = test simple aircraft\n")
    else:
        self.acmi_obj.write("#" + str(self.world.dt * self.time_count) + "\n")

        for fighter in self.world.fighters:
            data = str(fighter.obj_index) + "," + "T=" + str(fighter.fc_data.fLongitude) + "|" + \
                   str(fighter.fc_data.fLatitude) + "|" + \
                   str(fighter.fc_data.fAltitude) + "|" + str(format(fighter.fc_data.fRollAngle, '.3f')) + "|" + \
                   str(format(fighter.fc_data.fPitchAngle, '.3f')) + "|" + str(format(fighter.fc_data.fYawAngle, '.3f'))
            data += ", Type=Air+FixedWing,Coalition=Allies,Color="
            if fighter.side == 0:
                data += "Red"
                # if (fighter.index - self.num_RedFighter) % 2 == 0:
                #     data += "Red"
                # elif (fighter.index - self.num_RedFighter) % 2 == 1:
                #     data += "Orange"
                # else:
                #     data += "Red"
            else:
                data += "Blue"
                # if fighter.index % 2 == 0:
                #     data += "Blue"
                # elif fighter.index % 2 == 1:
                #     data += "Cyan"
                # else:
                #     data += "Blue"

            data += ",Name=" + fighter.type + ",Mach=" + str(format(fighter.fc_data.fMachNumber, '.3f'))
            data += ",ShortName=" + fighter.type + "  " + str(fighter.index) + "  " + str(
                format(fighter.combat_data.bloods, '.2f'))
            data += ",RadarMode=1" + ",RadarRange=" + str(fighter.sensors.radar_range) + ",RadarHorizontalBeamwidth=" + str(
                2 * fighter.sensors.radar_horizontal_scan) + ",RadarVerticalBeamwidth=" + str(2 * fighter.sensors.radar_vertical_scan)
            data += "\n"
            self.acmi_obj.write(data)

            # 导弹部分的数据写入
            for j in range(int(self.initial_data.missiles_max)):
                if fighter.missiles[j].state > 0:
                    data_missile = str(
                        fighter.obj_index * 10 + fighter.missiles[j].index) + "," + "T=" + \
                                   str(fighter.missiles[j].dataout.m_longitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_latitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_altitude) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_roll)) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_pitch)) + "|" + \
                                   str(np.rad2deg(-fighter.missiles[j].dataout.m_yaw))
                    data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color="
                    if fighter.side == 0:
                        if (fighter.index - self.num_RedFighter) % 2 == 0:
                            data_missile += "Red"
                        elif (fighter.index - self.num_RedFighter) % 2 == 1:
                            data_missile += "Orange"
                        else:
                            data_missile += "Red"
                    else:
                        if fighter.index % 2 == 0:
                            data_missile += "Blue"
                        elif fighter.index % 2 == 1:
                            data_missile += "Cyan"
                        else:
                            data_missile += "Blue"
                    data_missile += ",Name=" + fighter.missiles[j].type
                    data_missile += ",ShortName=" + fighter.missiles[j].type + "    Tar=" + str(fighter.missiles[j].target_index) + "    State=" + str(fighter.missiles[j].state)
                    # if fighter.side == 1:
                    #     if fighter.index == 2:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color=Red,Name="
                    #     if fighter.index == 3:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color=Orange,Name="
                    # else:
                    #     if fighter.index == 0:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Enemies,Color=Blue,Name="
                    #     if fighter.index == 1:
                    #         data_missile += ", Type=Medium+Weapon+Missile,Coalition=Enemies,Color=Cyan,Name="
                    data_missile += ",Mach=" + str(fighter.missiles[j].dataout.m_ma)
                    data_missile += "\n"
                    self.acmi_obj.write(data_missile)


def record_csv(self, datain, dataout):
    if self.time_count == 0:
        # csv文件里的态势记录
        self.csv_obj_state = open(self.file_dir + "\\csv_PC_" + str(self.epoch) + ".csv", "w+", newline='')
        self.csv_state_writer = csv.writer(self.csv_obj_state)
        csv_data_head = []
        csv_data_head.append("time_step")
        for i, data in enumerate(datain):
            csv_data_head.append("obj_index_" + str(i))
            csv_data_head.append("control_mode")
            csv_data_head.append("control_input[0]")
            csv_data_head.append("control_input[1]")
            csv_data_head.append("control_input[2]")
            csv_data_head.append("control_input[3]")
            csv_data_head.append("target_index")
            csv_data_head.append("fire")
            csv_data_head.append("missile_fire")
            csv_data_head.append("communication[0]")
            csv_data_head.append("communication[1]")
            csv_data_head.append("communication[2]")
            csv_data_head.append("communication[3]")
            csv_data_head.append("communication[4]")
        for i, data in enumerate(dataout):
            # ————————————————————————————— self data ——————————————————————————————
            csv_data_head.append("obj_index_" + str(i))
            csv_data_head.append("control_mode")
            csv_data_head.append("left_bullet")
            csv_data_head.append("left_missile")
            csv_data_head.append("left_bloods")

            csv_data_head.append("NorthAcceleration")
            csv_data_head.append("EastAcceleration")
            csv_data_head.append("VerticalAcceleration[0]")
            csv_data_head.append("PitchRate")
            csv_data_head.append("RollRate")
            csv_data_head.append("YawRate")
            csv_data_head.append("NormalLoad")
            csv_data_head.append("LateralLoad")
            csv_data_head.append("LongitudeinalLoad")

            csv_data_head.append("NorthVelocity")
            csv_data_head.append("EastVelocity")
            csv_data_head.append("VerticalVelocity")
            csv_data_head.append("NormalVelocity")
            csv_data_head.append("LateralVelocity")
            csv_data_head.append("LongitudianlVelocity")
            csv_data_head.append("Longitude")
            csv_data_head.append("Latitude")
            csv_data_head.append("Altitude")

            csv_data_head.append("PitchAngle")
            csv_data_head.append("RollAngle")
            csv_data_head.append("YawAngle")
            csv_data_head.append("AttackAngle")
            csv_data_head.append("SideslipAngle")

            csv_data_head.append("TrueAirSpeed")
            csv_data_head.append("IndicatedAirSpeed")
            csv_data_head.append("GroundSpeed")

            csv_data_head.append("NumberofFuel")
            csv_data_head.append("Thrust")
            csv_data_head.append("Missile1State")
            csv_data_head.append("Missile2State")

            # ————————————————————————————— radar data ——————————————————————————————
            csv_data_head.append("friend_EleAngle")
            csv_data_head.append("friend_AziAngle")
            csv_data_head.append("friend_Distance")
            csv_data_head.append("friend_NorthVelocity")
            csv_data_head.append("friend_EastVelocity")
            csv_data_head.append("friend_VerticalVelocity")

            csv_data_head.append("target1_Index")
            csv_data_head.append("target1_EleAngle")
            csv_data_head.append("target1_AziAngle")
            csv_data_head.append("target1_Distance")
            csv_data_head.append("target1_NorthVelocity")
            csv_data_head.append("target1_EastVelocity")
            csv_data_head.append("target1_VerticalVelocity")

            # ————————————————————————————— state data ——————————————————————————————
            csv_data_head.append("friend_Survive")
            csv_data_head.append("friend_Longitude")
            csv_data_head.append("friend_Latitude")
            csv_data_head.append("friend_Altitude")

            csv_data_head.append("target1_Index")
            csv_data_head.append("target1_Survive")
            csv_data_head.append("target1_Longitude")
            csv_data_head.append("target1_Latitude")
            csv_data_head.append("target1_Altitude")

            csv_data_head.append("target2_Index")
            csv_data_head.append("target2_Survive")
            csv_data_head.append("target2_Longitude")
            csv_data_head.append("target2_Latitude")
            csv_data_head.append("target2_Altitude")

        # ————————————————————————————— close data ——————————————————————————————
            csv_data_head.append("friend_EleAngle")
            csv_data_head.append("friend_AziAngle")
            csv_data_head.append("friend_Distance")

            csv_data_head.append("target1_Index")
            csv_data_head.append("target1_EleAngle")
            csv_data_head.append("target1_AziAngle")
            csv_data_head.append("target1_Distance")

            csv_data_head.append("target2_Index")
            csv_data_head.append("target2_EleAngle")
            csv_data_head.append("target2_AziAngle")
            csv_data_head.append("target2_Distance")
        # ————————————————————————————— alert data ——————————————————————————————
            csv_data_head.append("emergency_num")
            csv_data_head.append("emergency_EleAngle")
            csv_data_head.append("emergency_AziAngle")

            csv_data_head.append("emergency_missile_num")
            csv_data_head.append("emergency_missile_EleAngle")
            csv_data_head.append("emergency_missile_AziAngle")
        self.csv_state_writer.writerow(csv_data_head)
    else:
        # CSV里写入数据
        csv_data = []
        csv_data.append(self.world.dt * self.time_count)
        for i, data in enumerate(datain):
            csv_data.append(str(i))
            csv_data.append(data.control_mode)
            csv_data.append(data.control_input[0])
            csv_data.append(data.control_input[1])
            csv_data.append(data.control_input[2])
            csv_data.append(data.control_input[3])
            csv_data.append(data.target_index)
            csv_data.append(data.fire)
            csv_data.append(data.missile_fire)
            csv_data.append(data.communication[0])
            csv_data.append(data.communication[1])
            csv_data.append(data.communication[2])
            csv_data.append(data.communication[3])
            csv_data.append(data.communication[4])
        for i, data in enumerate(dataout):
            csv_data.append(str(i))
            csv_data.append(data.selfdata.control_mode)
            csv_data.append(data.selfdata.left_bullet)
            csv_data.append(data.selfdata.left_missile)
            csv_data.append(data.selfdata.left_bloods)

            csv_data.append(data.selfdata.NorthAcceleration)
            csv_data.append(data.selfdata.EastAcceleration)
            csv_data.append(data.selfdata.VerticalAcceleration)
            csv_data.append(data.selfdata.PitchRate)
            csv_data.append(data.selfdata.RollRate)
            csv_data.append(data.selfdata.YawRate)
            csv_data.append(data.selfdata.NormalLoad)
            csv_data.append(data.selfdata.LateralLoad)
            csv_data.append(data.selfdata.LongitudeinalLoad)

            csv_data.append(data.selfdata.NorthVelocity)
            csv_data.append(data.selfdata.EastVelocity)
            csv_data.append(data.selfdata.VerticalVelocity)
            csv_data.append(data.selfdata.NormalVelocity)
            csv_data.append(data.selfdata.LateralVelocity)
            csv_data.append(data.selfdata.LongitudianlVelocity)
            csv_data.append(data.selfdata.Longitude)
            csv_data.append(data.selfdata.Latitude)
            csv_data.append(data.selfdata.Altitude)

            csv_data.append(data.selfdata.PitchAngle)
            csv_data.append(data.selfdata.RollAngle)
            csv_data.append(data.selfdata.YawAngle)
            csv_data.append(data.selfdata.AttackAngle)
            csv_data.append(data.selfdata.SideslipAngle)
            csv_data.append(data.selfdata.TrueAirSpeed)
            csv_data.append(data.selfdata.IndicatedAirSpeed)
            csv_data.append(data.selfdata.GroundSpeed)

            csv_data.append(data.selfdata.NumberofFuel)
            csv_data.append(data.selfdata.Thrust)
            csv_data.append(data.selfdata.Missile1State)
            csv_data.append(data.selfdata.Missile2State)
            # ————————————————————————————— radar data ——————————————————————————————
            csv_data.append(data.radardata.friend_EleAngle)
            csv_data.append(data.radardata.friend_AziAngle)
            csv_data.append(data.radardata.friend_Distance)
            csv_data.append(data.radardata.friend_NorthVelocity)
            csv_data.append(data.radardata.friend_EastVelocity)
            csv_data.append(data.radardata.friend_VerticalVelocity)

            csv_data.append(data.radardata.target1_Index)
            csv_data.append(data.radardata.target1_EleAngle)
            csv_data.append(data.radardata.target1_AziAngle)
            csv_data.append(data.radardata.target1_Distance)
            csv_data.append(data.radardata.target1_NorthVelocity)
            csv_data.append(data.radardata.target1_EastVelocity)
            csv_data.append(data.radardata.target1_VerticalVelocity)
            # ————————————————————————————— state data ——————————————————————————————
            csv_data.append(data.statedata.friend_Survive)
            csv_data.append(data.statedata.friend_Longitude)
            csv_data.append(data.statedata.friend_Latitude)
            csv_data.append(data.statedata.friend_Altitude)
            csv_data.append(data.statedata.target1_Index)
            csv_data.append(data.statedata.target1_Survive)
            csv_data.append(data.statedata.target1_Longitude)
            csv_data.append(data.statedata.target1_Latitude)
            csv_data.append(data.statedata.target1_Altitude)
            csv_data.append(data.statedata.target2_Index)
            csv_data.append(data.statedata.target2_Survive)
            csv_data.append(data.statedata.target2_Longitude)
            csv_data.append(data.statedata.target2_Latitude)
            csv_data.append(data.statedata.target2_Altitude)
        # ————————————————————————————— close data ——————————————————————————————
            csv_data.append(data.closedata.friend_EleAngle)
            csv_data.append(data.closedata.friend_AziAngle)
            csv_data.append(data.closedata.friend_Distance)

            csv_data.append(data.closedata.target1_Index)
            csv_data.append(data.closedata.target1_EleAngle)
            csv_data.append(data.closedata.target1_AziAngle)
            csv_data.append(data.closedata.target1_Distance)

            csv_data.append(data.closedata.target2_Index)
            csv_data.append(data.closedata.target2_EleAngle)
            csv_data.append(data.closedata.target2_AziAngle)
            csv_data.append(data.closedata.target2_Distance)
            # ————————————————————————————— alert data ——————————————————————————————
            csv_data.append(data.alertdata.emergency_num)
            csv_data.append(data.alertdata.emergency_EleAngle)
            csv_data.append(data.alertdata.emergency_AziAngle)

            csv_data.append(data.alertdata.emergency_missile_num)
            csv_data.append(data.alertdata.emergency_missile_EleAngle)
            csv_data.append(data.alertdata.emergency_missile_AziAngle)

        self.csv_state_writer.writerow(csv_data)

def record_trajectory(env, t, epoch, log_path):
    trajectory_dir = os.path.join(log_path, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    trajectory_data_path = os.path.join(trajectory_dir, f"trajectory_data_{proc_id()}_{epoch}.csv")

    if t == 0 :
        with open(trajectory_data_path, 'w',encoding='utf-8', newline='') as first:
            csv_write = csv.writer(first)
            name = ['time',
                    'red_lon', 'red_lat', 'red_alt', "red_ele",
                    'red_north','red_east','red_down',
                    'red_roll', 'red_pitch', 'red_yaw',
                    "red_V_g",
                    "red_norm_load",
                    "red_Later_load",
                    "red_bloods",

                    'blue_lon', 'blue_lat', 'blue_alt',"blue_ele",
                    'blue_north', 'blue_east', 'blue_down',
                    'blue_roll', 'blue_pitch', 'blue_yaw',
                    "blue_V_g",
                    "blue_norm_load",
                    "blue_Later_load",
                    "blue_bloods"
                     ]
            csv_write.writerow(name)
            traj_out_with_attitude = [
             0,

            env.world.fighters[0].fc_data.fLongitude,
            env.world.fighters[0].fc_data.fLatitude,
            env.world.fighters[0].fc_data.fAltitude,

            get_elevation(env.world.fighters[0].fc_data.fLatitude, env.world.fighters[0].fc_data.fLongitude),


            env.world.fighters[0].state.ned_Pos[0],
            env.world.fighters[0].state.ned_Pos[1],
            env.world.fighters[0].state.ned_Pos[2],


            env.world.fighters[0].fc_data.fRollAngle,
            env.world.fighters[0].fc_data.fPitchAngle,
            env.world.fighters[0].fc_data.fYawAngle,

            env.world.fighters[0].fc_data.fGroundSpeed,
            env.world.fighters[0].fc_data.fNormalLoad,
            env.world.fighters[0].fc_data.fLateralLoad,

            env.world.fighters[0].combat_data.bloods,


            env.world.fighters[1].fc_data.fLongitude,
            env.world.fighters[1].fc_data.fLatitude,
            env.world.fighters[1].fc_data.fAltitude,
            get_elevation(env.world.fighters[1].fc_data.fLatitude, env.world.fighters[1].fc_data.fLongitude),

            env.world.fighters[1].state.ned_Pos[0],
            env.world.fighters[1].state.ned_Pos[1],
            env.world.fighters[1].state.ned_Pos[2],

            env.world.fighters[1].fc_data.fRollAngle,
            env.world.fighters[1].fc_data.fPitchAngle,
            env.world.fighters[1].fc_data.fYawAngle,

            env.world.fighters[1].fc_data.fGroundSpeed,
            env.world.fighters[1].fc_data.fNormalLoad,
            env.world.fighters[1].fc_data.fLateralLoad,

            env.world.fighters[1].combat_data.bloods,

            ]
            csv_write.writerow(traj_out_with_attitude)
            first.close()
    else:
        with open(trajectory_data_path, 'a',encoding='utf-8', newline='') as traj_out:

            csv_write = csv.writer(traj_out)

            red_terrain_height = get_elevation(env.world.fighters[0].fc_data.fLatitude, env.world.fighters[0].fc_data.fLongitude)

            traj_out_with_attitude = [t / 10,
                                      env.world.fighters[0].fc_data.fLongitude,
                                      env.world.fighters[0].fc_data.fLatitude,
                                      env.world.fighters[0].fc_data.fAltitude,

                                      get_elevation(env.world.fighters[0].fc_data.fLatitude,
                                                    env.world.fighters[0].fc_data.fLongitude),

                                      env.world.fighters[0].state.ned_Pos[0],
                                      env.world.fighters[0].state.ned_Pos[1],
                                      env.world.fighters[0].state.ned_Pos[2],

                                      env.world.fighters[0].fc_data.fRollAngle,
                                      env.world.fighters[0].fc_data.fPitchAngle,
                                      env.world.fighters[0].fc_data.fYawAngle,

                                      env.world.fighters[0].fc_data.fGroundSpeed,
                                      env.world.fighters[0].fc_data.fNormalLoad,
                                      env.world.fighters[0].fc_data.fLateralLoad,

                                      env.world.fighters[0].combat_data.bloods,

                                      env.world.fighters[1].fc_data.fLongitude,
                                      env.world.fighters[1].fc_data.fLatitude,
                                      env.world.fighters[1].fc_data.fAltitude,
                                      get_elevation(env.world.fighters[1].fc_data.fLatitude,
                                                    env.world.fighters[1].fc_data.fLongitude),

                                      env.world.fighters[1].state.ned_Pos[0],
                                      env.world.fighters[1].state.ned_Pos[1],
                                      env.world.fighters[1].state.ned_Pos[2],

                                      env.world.fighters[1].fc_data.fRollAngle,
                                      env.world.fighters[1].fc_data.fPitchAngle,
                                      env.world.fighters[1].fc_data.fYawAngle,

                                      env.world.fighters[1].fc_data.fGroundSpeed,
                                      env.world.fighters[1].fc_data.fNormalLoad,
                                      env.world.fighters[1].fc_data.fLateralLoad,
                                      env.world.fighters[1].combat_data.bloods,
                                      ]
            csv_write.writerow(traj_out_with_attitude)
            traj_out.close()


def win_statistics(env, terminal):
    red_win_num = 0
    red_fall = 0
    blue_win_num = 0
    blue_fall = 0


    # 计算红蓝机本轮伤害
    red_harm = 3 - env.world.fighters[1].combat_data.bloods
    blue_harm = 3 - env.world.fighters[0].combat_data.bloods

    # 仿真达到最大步长
    if terminal == 0:
        if env.world.fighters[0].combat_data.bloods > env.world.fighters[1].combat_data.bloods:
            red_win_num = 1

        if env.world.fighters[1].combat_data.bloods > env.world.fighters[0].combat_data.bloods:
            blue_win_num = 1
    # 蓝机被击落
    if terminal == 1:
        red_win_num = 1
    # 红机被击落
    if terminal == 2:
        blue_win_num = 1
    # 红机坠地
    if terminal == 4:
        red_fall = 1
        blue_win_num = 1
    # 蓝机坠地
    if terminal == 5:
        blue_fall = 1
        red_win_num = 1
    return red_win_num, red_fall, red_harm, blue_win_num, blue_fall, blue_harm


