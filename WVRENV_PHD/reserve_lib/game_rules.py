import numpy as np
from WVRENV_PHD.utils.GNCData import wgs84ToNED


class Game_Rule(object):
    def __init__(self):
        self.time_count = 0
        self.red_time_count = 0
        self.blue_time_count = 0
        self.blue_time_count_level1 = 0
        self.red_time_count_level1 = 0

    def reset(self):
        self.time_count = 0
        self.red_time_count = 0
        self.blue_time_count = 0

        self.blue_time_count_level1 = 0
        self.red_time_count_level1 = 0

    def rule_update(self, world):
        self.time_count += 1

        # 解算是否有飞机在要地空域内
        blue_in_tag, red_in_tag = self.judge_distance(world)

        # 解算哪一方累计一级制空时间（1s内)
        if (blue_in_tag == True) and (red_in_tag == False):
            self.blue_time_count_level1 += 1
        elif (blue_in_tag == False) and (red_in_tag == True):
            self.red_time_count_level1 += 1
        else:
            pass

        # 解算是否超过1s，真正的增加制空时间
        if self.blue_time_count_level1 >= 100:
            self.blue_time_count += 1
            self.blue_time_count_level1 = 0
        if self.red_time_count_level1 >= 100:
            self.red_time_count += 1
            self.red_time_count_level1 = 0

        # 解算一级制空时间是否置零
        if (blue_in_tag == True) and (red_in_tag == True):
            self.blue_time_count_level1 = 0
            self.red_time_count_level1 = 0

        # 解算二级制空时间是否置零
        if (blue_in_tag == False):
            self.blue_time_count = 0
            self.blue_time_count_level1 = 0
        if (red_in_tag == False):
            self.red_time_count = 0
            self.red_time_count_level1 = 0

        # print('蓝方与红方的制空时间', self.blue_time_count_level1,  self.blue_time_count,
        #       self.red_time_count_level1, self.red_time_count)

        return self.blue_time_count, self.red_time_count

    def judge_distance(self, world):
        distance_mul = []
        fighter_in_tag = []
        for i, fighter in enumerate(world.fighters[0:4]):
            # distance_sing = fighter.state.ned_Pos[0:2] - [0, 0]
            # distance_mul.append((distance_sing[0]**2 + distance_sing[1]**2) ** 0.5)

            distance_sing = wgs84ToNED(fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, 0)
            distance_mul.append((distance_sing[0]**2 + distance_sing[1]**2) ** 0.5)

            # 判断是否在要地内的条件：距离小于30000，高度小于12000，且活着
            if ((distance_mul[i] <= 30000) and (fighter.fc_data.fAltitude <= 12000) and (fighter.combat_data.survive_info == True)):
                fighter_in_tag.append(True)
            else:
                fighter_in_tag.append(False)

        if (fighter_in_tag[0] == True) or (fighter_in_tag[1] == True ):
            blue_in_tag = True
        else:
            blue_in_tag = False
        if (fighter_in_tag[2] == True) or (fighter_in_tag[3] == True ):
            red_in_tag = True
        else:
            red_in_tag = False

        # print('规则系统里离制空区域的距离', distance_mul, blue_in_tag, red_in_tag)

        return blue_in_tag, red_in_tag




