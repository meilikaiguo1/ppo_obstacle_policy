import socket
import numpy as np
from WVRENV_PHD.basic.DataResolve import missiles_max


class TcpRender(object):
    def __init__(self, port):
        # 1. 创建套接字 socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 2. 绑定本地信息 bind
        self.server_socket.bind(("", port))

        # 3. 让默认的套接字由主动变为被动 listen
        self.server_socket.listen(128)

        # 4. 等待客户端的链接 accept
        print("-----1----")
        self.new_client_socket, client_addr = self.server_socket.accept()
        print("-----2----")

        print(client_addr)

        self.head = "XtraLib.Stream.0\n" + "Tacview.RealTimeTelemetry.0\n" + "Host username\n" + \
                    "\0" + \
                    "FileType=text/acmi/tacview\n" + "FileVersion" \
                                                     "=2.2\n" + \
                    "0, ReferenceTime = 2021-03-01T05:27:00Z\n" + \
                    "0, RecordingTime = 2021-03-09T16:17:49Z\n" + \
                    "0, Title = test simple aircraft\n" + \
                    "0, DataRecorder = DCS2ACMI 1.6.0\n" + \
                    "0, DataSource = DCS 1.5.6.1938\n" + \
                    "0, Author = AuthorName\n" + \
                    "0, ReferenceLongitude = 125 \n" + \
                    "0, ReferenceLatitude = 20 \n"

    def send_head(self):
        self.new_client_socket.send(self.head.encode("utf-8"))

    def send_data_render(self, dt, time_str, env):

        # 这一帧的时间
        data = "#" + str(dt * time_str) + "\n"

        # 每架飞机的数据
        for i, fighter in enumerate(env.world.fighters):
            data += str(fighter.obj_index) + "," + "T=" + str(fighter.fc_data.fLongitude) + "|" + \
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
            data += "  mispre1=" + str(fighter.missiles[0].launch_prepare_time)
            data += "  mispre2=" + str(fighter.missiles[1].launch_prepare_time)
            data += "  alert=" + str(fighter.sensors.alert_missile)
            data += "\n"


            # 导弹部分的数据写入
            for j in range(int(env.initial_data.missiles_max)):
                if fighter.missiles[j].state == 1:
                    data += str(
                        fighter.obj_index * 100 + fighter.missiles[j].index) + "," + "T=" + \
                                   str(fighter.missiles[j].dataout.m_longitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_latitude) + "|" + \
                                   str(fighter.missiles[j].dataout.m_altitude) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_roll)) + "|" + \
                                   str(np.rad2deg(fighter.missiles[j].dataout.m_pitch)) + "|" + \
                                   str(np.rad2deg(-fighter.missiles[j].dataout.m_yaw))
                    data += ", Type=Medium+Weapon+Missile,Coalition=Allies,Color="
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
                    data += ",Name=" + fighter.missiles[j].type
                    data += ",ShortName=" + fighter.missiles[j].type + "    Tar=" + str(fighter.missiles[j].target_index)

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
                    data += "  Mach=" + str(format(fighter.missiles[j].dataout.m_ma, '.1f'))
                    data += "  state=" + str(fighter.missiles[j].state)
                    data += "\n"

        # 把数据发送出去
        self.new_client_socket.send(data.encode("utf-8"))

    def send_end_data(self, dict):
        data = "End,"
        data += "Winner="
        if dict["blue_win"] == 1:
            data += str(0)
        elif dict["red_win"] == 1:
            data += str(1)
        else:
            data += str(2)
        # 蓝色机的信息
        data += ",Blue_Score=" + str(dict["blue_score"])
        data += ",Blue_Fighters=" + str(dict["blue_fighters"])
        data += ",Blue_Time=" + str(dict["blue_time"])
        data += ",Blue_Bloods=" + str(dict["blue_left_bloods"])
        data += ",Blue_Missiles=" + str(dict["blue_left_missile"])

        # 红色机的信息
        data += ",Red_Score=" + str(dict["red_score"])
        data += ",Red_Fighters=" + str(dict["red_fighters"])
        data += ",Red_Time=" + str(dict["red_time"])
        data += ",Red_Bloods=" + str(dict["red_left_bloods"])
        data += ",Red_Missiles=" + str(dict["red_left_missile"])

        # 把数据发送出去
        self.new_client_socket.send(data.encode("utf-8"))

    def close(self):
        self.new_client_socket.close()
        self.server_socket.close()





