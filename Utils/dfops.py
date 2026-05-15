import random

import numpy as np
import torch.nn as nn
from Utils.height_protection import rel_height_protection, Comb_glorithm, abs_height_protection
from WVRENV_PHD.utils.GNCData import wgs84ToNED, body_to_ned
from model.net_dogfight import Dogfight_ActorCritic, Avoidance_pi
from observation.observation import get_dogfight_obs, get_avoidance_obs
from policy.paper2_deploy import Policy, Observation
import torch
from Spinup.mpi_torch_utils import proc_id
from policy.warcraft import Warcraft








class DFOPS(object):
    def __init__(self, args,p_id ,device = torch.device("cuda:0")):
        self.init_kar = args.init_kar
        self.proc_id = proc_id()
        self.net_device =  device
        self.pid = p_id

        #boltzman 分布中的温度系数
        self.sample_scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.opponent_id = 0
        self.temp = 4
        self.last_index = 0

        #目标视线
        self.target_los = [1, 0, 0]

        #加载cc_ppo网络
        self.cc_policy = Policy(17, 5, hidden_sizes=(128, 128, 128, 128, 128, 128), activation=nn.LeakyReLU)
        self.cc_policy.load_state_dict(torch.load("policy/net/pi_state_dict.pt"))
        self.cc_policy.eval()
        self.obs_agent = Observation()

        #论文避障策略
        self.ao = Comb_glorithm()

        #dc状态机
        self.warcraft = Warcraft()

        #历史策略保存路径
        self.self_play_dir = (args.model_dir + "\\" + str(args.seed) + "\\ActorCritic\\history_model\\" )

        #加载自博弈参数
        self.device = torch.device("cuda:0")
        self.act_dim = args.dogfight_act_dim
        self.obs_self_dim = args.dogf_obs_self_dim
        self.obs_target_dim = args.obs_target_dim
        self.obs_terrain_dim = args.obs_terrain_dim
        self.mlp_feat_dim = args.mlp_feat_dim

        self.avoidance_obs_self_dim = args.avo_obs_self_dim
        self.avoidance_act_dim = args.avoidance_act_dim

        self.avoidance_pi = Avoidance_pi(self.avoidance_obs_self_dim, self.obs_terrain_dim, self.avoidance_act_dim,
                                         self.mlp_feat_dim).to(self.device)
        self.avoidance_pi.eval()

        # 对手策略集
        self.opponent_list = [
            {"type": "ppg", "ManeuverLib": True, "Pram,net": self.init_kar, "index": 0},
            {"type": "cc_ppo", "ManeuverLib": True, "Pram,net": self.init_kar, "index": 1},
            {"type": "cc_abs", "ManeuverLib": True, "Pram,net": self.init_kar, "index": 2},
            {"type": "warcraft", "ManeuverLib": True, "Pram,net": self.init_kar, "index": 3},
        ]
        self.self_play_index = 1

        #记录50个epoch中，每个策略的统计量
        self.epi_count = 0
        self.score_red_win = np.zeros(4, dtype=np.float32)
        self.score_blue_win = np.zeros(4, dtype=np.float32)
        self.score_draw = np.zeros(4, dtype=np.float32)

        # 保留一点随机采样，防止某些策略永远选不到
        self.min_sample_weight = 0.1



        if args.continue_sf:
            for i in range(args.history_sf_num):
                history_id = args.history_start_index + i
                new_opponent_dict = {"type":"net",
                                    "ManeuverLib": False,
                                     "Pram,net": Dogfight_ActorCritic(self.obs_self_dim, self.obs_target_dim,
                                                                      self.obs_terrain_dim, self.act_dim,
                                                                      self.mlp_feat_dim).to(self.device),
                                     "index": 0}
                new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "history_policy_" + str(history_id) + ".pt")
                new_opponent_dict['index'] = history_id
                self.append_new_opponent(new_opponent_dict, init_score=0.5)
            self.self_play_index = args.history_start_index + args.history_sf_num

    def record_episode_result(self, opponent_id, result_flag):
        #记录每个episode的对战结果
        if opponent_id < 0 or opponent_id >= len(self.sample_scores):
            return
        #平局
        if result_flag == 0:
            self.score_draw[opponent_id] += 1
        #红方胜
        elif result_flag == 1:
            self.score_red_win[opponent_id] += 1
        #蓝方胜
        elif result_flag == 2:
            self.score_blue_win[opponent_id] += 1

    def update_score(self):
        for i in range(4):
            games = self.score_red_win[i] + self.score_blue_win[i] + self.score_draw[i]
            if games > 0 :
                win_rate = ( self.score_red_win[i] + self.score_draw[i] * 0.5 ) / games
                self.sample_scores[i] = win_rate
            else:
                self.sample_scores[i] = self.sample_scores[i]

        self.score_red_win = np.zeros(4, dtype=np.float32)
        self.score_blue_win = np.zeros(4, dtype=np.float32)
        self.score_draw = np.zeros(4, dtype=np.float32)


    def meta_solver(self):
        '''
        根据 sample_scores 选择对手。
        sample_scores 表示红方对各策略的胜率估计。
        胜率越接近 0.5，采样概率越高；
        胜率接近 0 或 1，采样概率越低。
        '''
        self.epi_count += 1

        if self.epi_count % 50 == 0:
            self.update_score()

        scores = np.asarray(self.sample_scores, dtype=np.float32)
        #距离0.5越近，weight越大
        weights = 1.0 - 2.0 * np.abs(scores - 0.5)

        # 防止某个策略概率完全为 0
        weights = np.clip(weights, self.min_sample_weight, None)

        #归一化成概率
        probs = weights / np.sum(weights)
        self.opponent_id = random.choices(np.arange(len(probs)), weights=probs, k=1)[0]
        if self.opponent_list[self.opponent_id]["type"] == 'warcraft':
            self.warcraft.policy_reset
        if self.pid == 0:
            print(f'当前对手策略为 = {self.opponent_list[self.opponent_id]["type"]}')


    def sampled_policy(self, env, maneuver_lib):
        fighter = env.world.fighters[1]
        target = env.world.fighters[0]


        if self.opponent_list[self.opponent_id]["type"] == "ppg":
            action = self.ppg_policy(fighter, target, self.opponent_list[self.opponent_id]["Pram,net"],maneuver_lib)

        elif self.opponent_list[self.opponent_id]["type"] == "cc_ppo":
            action = self.cc_ppo_policy(env, fighter)

        elif self.opponent_list[self.opponent_id]["type"] == "cc_abs":
            action = self.cc_ppo_abs_policy(env, fighter)

        elif self.opponent_list[self.opponent_id]["type"] == "warcraft":
            action = self.warcraft_policy(fighter, target)

        elif self.opponent_list[self.opponent_id]["type"] == "net":
            ac_model  = self.opponent_list[self.opponent_id]['Pram,net']
            ac_model.eval()
            dogf_obs_self, dogf_obs_t, normal_voxels = get_dogfight_obs(env, fighter, target)
            dogf_obs_self = torch.as_tensor(dogf_obs_self, dtype=torch.float32, device=self.device)
            dogf_obs_t = torch.as_tensor(dogf_obs_t, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                terrain_grid = ac_model.encode_terrain(normal_voxels)
                a, _, _, _ = ac_model.get_action_and_value(dogf_obs_self, dogf_obs_t, terrain_grid)
            a = a.squeeze(0)
            ele_ang = a[0].item() * 45
            azi_ang = a[1].item() * 90
            dis = 500 + (1 + a[2].item()) * 500

            action = dogf2avoidance(ele_ang, azi_ang, dis, terrain_grid, env, fighter, self.avoidance_pi, device = self.device)
        return action


    def update_opponent_list(self, epoch, avg_proc_win_rate, avg_proc_op_win, history_index):
        if (epoch % 5 == 0) and (self.opponent_list[-1]["ManeuverLib"]):
            if self.opponent_list[-1]['Pram,net'] > 0.1:
                for op_d in self.opponent_list:
                    if op_d['ManeuverLib'] == True :
                        op_d['Pram,net'] *= 0.989

                else:
                    pass
        if ((avg_proc_win_rate / (avg_proc_op_win + 0.001)) > 1.05) and (epoch % 100 == 0) and (epoch != 0 ):
            # 随机初始化的网络
            new_opponent_dict = {"type":"net",
                                "ManeuverLib": False,
                                 "Pram,net": Dogfight_ActorCritic(self.obs_self_dim, self.obs_target_dim,self.obs_terrain_dim, self.act_dim, self.mlp_feat_dim ).to(self.device),
                                 "index": 0}

            if self.opponent_list[-1]['ManeuverLib']:
                # 判断制导的随机参数是否 < 1
                if self.opponent_list[-1]['Pram,net'] > 0.1:
                    new_opponent_dict['ManeuverLib'] = True
                    new_opponent_dict['Pram,net'] = self.opponent_list[-1]['Pram,net']
                    new_opponent_dict["index"] = self.opponent_list[-1]['index']

                else:
                    if (history_index > 0) and (self.self_play_index < history_index):
                        new_opponent_dict['ManeuverLib'] = False
                        new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "history_policy_" + str(self.self_play_index) + ".pt")
                        new_opponent_dict["index"] = self.self_play_index
                        self.self_play_index += 1

                    elif history_index > 0:
                        sfpid = np.random.randint(1, history_index, 1)[0]
                        new_opponent_dict['ManeuverLib'] = False
                        new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "history_policy_" + str(sfpid) + ".pt")

            else:
                if (history_index > 0) and (self.self_play_index < history_index):
                    new_opponent_dict['ManeuverLib'] = False
                    new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "history_policy_" + str(self.self_play_index) + ".pt")
                    new_opponent_dict["index"] = self.self_play_index
                    self.self_play_index += 1
                elif history_index > 0:
                    sfpid = np.random.randint(1, history_index, 1)[0]
                    new_opponent_dict['ManeuverLib'] = False
                    new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "history_policy_" + str(sfpid) + ".pt")
                    new_opponent_dict["index"] = sfpid
            self.append_new_opponent(new_opponent_dict, init_score=0.5)
            if all([op['ManeuverLib'] ==  False for op in self.opponent_list]):
                self.opponent_list[0]['ManeuverLib'] = True
                self.opponent_list[0]['Pram,net'] = 0.08
                self.opponent_list[0]['index'] = 0
                self.opponent_list[0]['type'] = np.random.choice(["ppg", "cc_ppo", "cc_abs", "warcraft"])

                # 新插入的策略重新按 0.5 初始化
                self.sample_scores[0] = 0.5
                self.score_red_win[0] = 0
                self.score_blue_win[0] = 0
                self.score_draw[0] = 0

    def append_new_opponent(self, new_opponent_dict, init_score=0.5):
        """
        将新的对手策略加入策略池末尾，同时同步更新得分和统计量。
        逻辑：
            1. 删除最前面的旧策略
            2. 新策略加入最后
            3. sample_scores 同步左移
            4. 新策略得分初始化为 init_score
            5. 新策略统计量清零
        """

        # 策略池左移并加入新策略
        self.opponent_list = self.opponent_list[1:] + [new_opponent_dict]

        # 得分同步左移，新策略初始得分设为 0.5
        self.sample_scores = np.concatenate([
            self.sample_scores[1:],
            np.array([init_score], dtype=np.float32)
        ])

        # 统计量同步左移，新策略统计清零
        self.score_red_win = np.concatenate([
            self.score_red_win[1:],
            np.array([0.0], dtype=np.float32)
        ])

        self.score_blue_win = np.concatenate([
            self.score_blue_win[1:],
            np.array([0.0], dtype=np.float32)
        ])

        self.score_draw = np.concatenate([
            self.score_draw[1:],
            np.array([0.0], dtype=np.float32)
        ])


    def ppg_policy(self,fighter, target, kar, maneuver_lib):

        #计算视线
        l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                   target.fc_data.fAltitude,
                                   fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                   fighter.fc_data.fAltitude)
        self.target_los = [l_n, l_e, l_d]

        #ppg策略
        thrust, load, omega, rudder = maneuver_lib.kar_ppg(fighter, kar = kar, k = 0.9, target_los = [l_n, l_e, l_d])
        thrust = min(1., max(0.1, thrust))
        load = min(9., max(-3., load))
        omega = min(300., max(-300., omega))
        rudder = min(1., max(-1., rudder))
        action = [thrust, (load / 9) if load > 0 else (load / 3), omega / 300, rudder]

        #高度保护
        action = rel_height_protection(fighter, action)
        return action

    def cc_ppo_policy(self, env, fighter):
        obs = self.obs_agent.get_obs(fighter, env, 0.1)['DNN']
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.cc_policy(obs)
        load = 9 * action[0] if action[0] > 0 else (3 * action[0])
        load = min(9, max(-3, load))  # 法向过载 -3 ~ 9
        omega = min(300, max(-300, 300 * action[1]))  # 滚转速率 -300 ~ 300
        rudder = min(1., max(-1., action[2]))  # 方向舵 -1 ~ 1
        thrust = min(1., max(0.1, action[3]))

        action = [thrust, (load / 9) if load > 0 else (load / 3), omega / 300, rudder]
        action = self.ao.step(fighter, action)
        return action

    def cc_ppo_abs_policy(self, env, fighter):
        obs = self.obs_agent.get_obs(fighter, env, 0.1)['DNN']
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.cc_policy(obs)
        load = 9 * action[0] if action[0] > 0 else (3 * action[0])
        load = min(9, max(-3, load))  # 法向过载 -3 ~ 9
        omega = min(300, max(-300, 300 * action[1]))  # 滚转速率 -300 ~ 300
        rudder = min(1., max(-1., action[2]))  # 方向舵 -1 ~ 1
        thrust = min(1., max(0.1, action[3]))

        action = [thrust, (load / 9) if load > 0 else (load / 3), omega / 300, rudder]
        action = abs_height_protection(fighter, action)
        return action

    def warcraft_policy(self, fighter, target):
        action = self.warcraft.aircraft(fighter, target)
        load = action[0] / 9 if action[0] > 0 else (action[0] / 3)
        omega = action[1] / 300
        thrust = action[2] / 100
        rudder = 0
        action = [thrust, load, omega, rudder]
        action = abs_height_protection(fighter, action)
        return action



def dogf2avoidance(ele, azi, dis, terrain_grid, env, fighter, avo_actor, device =torch.device("cuda:0")):
    ele_rad = np.deg2rad(ele)
    azi_rad = np.deg2rad(azi)
    des_body = [dis * np.cos(ele_rad) * np.cos(azi_rad), dis * np.cos(ele_rad) * np.sin(azi_rad),
                - dis * np.sin(ele_rad)]
    des_ned = body_to_ned(des_body, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fRollAngle)
    des_ned = [fighter.state.ned_Pos[0] + des_ned[0], fighter.state.ned_Pos[1] + des_ned[1],
               fighter.state.ned_Pos[2] + des_ned[2]]

    avo_obs_self = get_avoidance_obs(env, fighter, des_ned)
    avo_obs_self = torch.as_tensor(avo_obs_self, dtype=torch.float32, device=device)

    with torch.no_grad():
        action = avo_actor(avo_obs_self, terrain_grid)
    action = action.squeeze(0)
    a0 = action[0].item()
    a1 = action[1].item()
    a2 = action[2].item()
    a3 = action[3].item()

    load = 9 * a0 if a0 > 0 else 3 * a0
    omega = 300 * a1
    rudder = a2
    thrust = 0.75 + 0.25 * a3

    load = min(9, max(-3, load))
    omega = min(300, max(-300, omega))
    rudder = min(1., max(-1., rudder))
    thrust = min(1., max(0.1, thrust))
    action = [thrust, (load / 9) if load > 0 else (load / 3), omega / 300, rudder]
    return  action









