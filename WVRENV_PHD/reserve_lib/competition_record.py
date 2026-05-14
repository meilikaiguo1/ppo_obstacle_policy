import numpy as np


class CompetitionRecord(object):
    def __init__(self):
        self.POS_X_BIAS = 0
        self.POS_Y_BIAS = 0
        self.MAP_LIMIT = 120 * 1000  # /m

        self.dict = {
            # 蓝方
            "blue_shoot_count": 0,  # 击杀数
            "blue_be_shot_count": 0,  # 被击杀数
            "blue_net_shot_count": 0,  # 净击杀数
            # "blue_damage_value": 0,  # 毁伤值
            "blue_fighters": 0,         # 剩余飞机
            "blue_time": 0,         # 制空时间
            "blue_left_bloods": 0,  # 剩余生命值
            "blue_left_missile": 0,              # 剩余导弹数
            "blue_win": 0,              # 蓝方获胜
            "blue_score": 0,            # 总积分
            # "blue_break_through": False,    # 是否穿透

            # 红方
            "red_shoot_count": 0,     # 击杀数
            "red_be_shot_count": 0,   # 被击杀数
            "red_net_shot_count": 0,  # 净击杀数
            # "red_damage_value": 0,    # 毁伤值
            "red_fighters": 0,  # 剩余飞机
            "red_time": 0,
            "red_left_bloods": 0,  # 剩余生命值
            "red_left_missile": 0,  # 剩余导弹数
            "red_win": 0,  # 蓝方获胜
            "red_score": 0,  # 总积分

            # 平局
            "dead_heat": 0,       # 平局

        }

    def generate_record(self, env):
        # 击杀数计算
        blue_kill = 0
        if env.world.fighters[2].combat_data.survive_info == False and env.world.fighters[2].combat_data.be_effective_killed == 1:
            blue_kill += 1
        if env.world.fighters[3].combat_data.survive_info == False and env.world.fighters[3].combat_data.be_effective_killed == 1:
            blue_kill += 1
        self.dict["blue_shoot_count"] = blue_kill
        red_kill = 0
        if env.world.fighters[1].combat_data.survive_info == False and env.world.fighters[1].combat_data.be_effective_killed == 1:
            red_kill += 1
        if env.world.fighters[0].combat_data.survive_info == False and env.world.fighters[0].combat_data.be_effective_killed == 1:
            red_kill += 1
        self.dict["red_shoot_count"] = red_kill

        # 被击杀数计算
        self.dict["blue_be_shot_count"] = self.dict["red_shoot_count"]
        self.dict["red_be_shot_count"] = self.dict["blue_shoot_count"]

        # 净击杀数计算
        self.dict["blue_net_shot_count"] = self.dict["blue_shoot_count"] - self.dict["blue_be_shot_count"]
        self.dict["red_net_shot_count"] = self.dict["red_shoot_count"] - self.dict["red_be_shot_count"]

        # 制空时间计算
        self.dict["blue_time"] = env.game_rule.blue_time_count
        self.dict["red_time"] = env.game_rule.red_time_count

        # 剩余飞机计算
        if env.world.fighters[0].combat_data.survive_info == True or env.world.fighters[1].combat_data.survive_info == True:
            self.dict["blue_fighters"] = 1
        if env.world.fighters[0].combat_data.survive_info == True and env.world.fighters[1].combat_data.survive_info == True:
            self.dict["blue_fighters"] = 2
        if env.world.fighters[0].combat_data.survive_info == False and env.world.fighters[1].combat_data.survive_info == False:
            self.dict["blue_fighters"] = 0
        if env.world.fighters[2].combat_data.survive_info == True or env.world.fighters[3].combat_data.survive_info == True:
            self.dict["red_fighters"] = 1
        if env.world.fighters[2].combat_data.survive_info == True and env.world.fighters[3].combat_data.survive_info == True:
            self.dict["red_fighters"] = 2
        if env.world.fighters[2].combat_data.survive_info == False and env.world.fighters[3].combat_data.survive_info == False:
            self.dict["red_fighters"] = 0

        # 剩余生命值计算
        self.dict["blue_left_bloods"] = env.world.fighters[0].combat_data.bloods + env.world.fighters[1].combat_data.bloods
        self.dict["red_left_bloods"] = env.world.fighters[2].combat_data.bloods + env.world.fighters[3].combat_data.bloods

        # 剩余导弹数计算
        self.dict["blue_left_missile"] = env.world.fighters[0].missiles_left + env.world.fighters[1].missiles_left
        self.dict["red_left_missile"] = env.world.fighters[2].missiles_left + env.world.fighters[3].missiles_left

    def scole_record(self, terminal):
        # 0： 蓝方占领30s  4： 红方都死亡，蓝方获胜      1：红方占领30s   3：蓝方都死亡，红方获胜
        # 是否有一方全灭
        if terminal == 4:
            self.dict["blue_win"] = 1
        elif terminal == 3:
            self.dict["red_win"] = 1
        else:
            # 是否有一方占领30s
            if terminal == 0:
                self.dict["blue_win"] = 1
            elif terminal == 1:
                self.dict["red_win"] = 1
            else:
                # 都未占领30s，哪一方的剩余数量多
                if self.dict["blue_fighters"] > self.dict["red_fighters"]:
                    self.dict["blue_win"] = 1
                elif self.dict["blue_fighters"] < self.dict["red_fighters"]:
                    self.dict["red_win"] = 1
                else:
                    # 剩余数量也一样，哪一方的占领时间长
                    if self.dict["blue_time"] > self.dict["red_time"]:
                        self.dict["blue_win"] = 1
                    elif self.dict["blue_time"] < self.dict["red_time"]:
                        self.dict["red_win"] = 1
                    else:
                        # 占领时间也一样，哪一方的剩余血量总和多
                        if self.dict["blue_left_bloods"] > self.dict["red_left_bloods"]:
                            self.dict["blue_win"] = 1
                        elif self.dict["blue_left_bloods"] < self.dict["red_left_bloods"]:
                            self.dict["red_win"] = 1
                        else:
                            # 剩余血量也一样，哪一方的剩余导弹总和多
                            if self.dict["blue_left_missile"] > self.dict["red_left_missile"]:
                                self.dict["blue_win"] = 1
                            elif self.dict["blue_left_missile"] < self.dict["red_left_missile"]:
                                self.dict["red_win"] = 1
                            else:
                                # 还没决出胜负，就平局吧
                                self.dict["blue_win"] = 0
                                self.dict["red_win"] = 0
                                self.dict["dead_heat"] = 1

        # 分数计算
        self.dict["blue_score"] = 2 * self.dict["blue_shoot_count"] + 5 * self.dict["blue_win"] + 1 * self.dict["dead_heat"]

        self.dict["red_score"] = 2 * self.dict["red_shoot_count"] + 5 * self.dict["red_win"] + 1 * self.dict["dead_heat"]
