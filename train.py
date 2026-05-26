import argparse
from Reward.reward import reward_func
from Spinup.mpi_torch_utils import proc_id, setup_pytorch_for_mpi, num_procs, mpi_statistics_scalar, mpi_sum, mpi_fork, \
    mpi_avg
import torch.distributed as dist
import numpy as np
import os
import csv

from Utils.dfops import dogf2avoidance, DFOPS
from Utils.util import RunningMeanStd, win_rate
import torch.distributed as t_dist
import torch.multiprocessing as mp
from Spinup.torch_distribute import statistics_scalar_torch, average_x_torch
from WVRENV_PHD.utils.data_record import record_trajectory
from fighter_bot import ManeuversLib
from model.ppo_avoidance import PPOAgent
import torch
import time
from Simset.sim_set import make_sim_env, Reset
from torch.utils.tensorboard import SummaryWriter

from model.ppo_dogfight import DogfightAgent
from observation.observation import get_avoidance_obs, get_dogfight_obs
import datetime


def train(args):
    #仿真环境
    env, sim_in_list = make_sim_env(args)

    #随机种子
    p_id = dist.get_rank() if args.torch_dist else proc_id()
    torch.manual_seed(args.seed + p_id *1000)
    np.random.seed(args.seed + p_id *1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + p_id *1000)

    #储存路径
    file_dir = os.path.join(".", "output", str(args.seed) , "logfiles")
    train_dir = os.path.join(file_dir, "training_data")
    reward_dir = os.path.join(file_dir, "reward_info")
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    file_name = os.path.join(train_dir, "training_data.csv")

    #训练数据记录表头
    if p_id ==0 and not args.continue_train:
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch",  "epi_turn", "step_mean_return" ,"pi_loss", "v_loss", "entropy", "kl", "cf", "red_win", "blue_win", "draw", "red_fall","blue_fall",\
                 "red_harm", "blue_harm","red_win_rate", "blue_win_rate", "draw_rate", "red_fall_rate", "blue_fall_rate","policy" ])


    #tensorboard
    tb_writer = SummaryWriter("runs") if p_id == 0 else None

    if not args.torch_dist:
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

    #reward_rms
    rewards_rms = RunningMeanStd(shape=1)

    #device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    #红蓝机控制指令agent
    ppo_agent = DogfightAgent(p_id, args, trainable = args.trainable, torch_dist= args.torch_dist, device = device)

    #蓝机策略池
    dfops = DFOPS(args, p_id)
    maneuver_lib = ManeuversLib(0.1)
    #历史策略编号
    history_index = args.sf_history_index

    #恢复训练时加载模型
    if args.continue_train:
        ppo_agent.actor_critic.load_state_dict(torch.load(f"./output/{args.seed}/ActorCritic/trained_model/dogfight_actor_critic_{args.continue_epoch}.pt", map_location=device))
        ppo_agent.ac_optimizer.load_state_dict(torch.load(f"./output/{args.seed}/ActorCritic/trained_model/dogfight_ac_optimizer_{args.continue_epoch}.pt", map_location=device))

    #更新网络使用的buffer
    obs_self_buf = torch.zeros((args.per_steps ,args.dogf_obs_self_dim),dtype=torch.float32, device=device)
    obs_target_buf = torch.zeros((args.per_steps, args.obs_target_dim), dtype=torch.float32, device=device)
    action_buf = torch.zeros((args.per_steps, args.dogfight_act_dim), dtype=torch.float32, device=device)
    logprob_buf = torch.zeros(args.per_steps, dtype=torch.float32, device=device)
    rewards_buf = torch.zeros(args.per_steps,dtype=torch.float32, device=device)
    done_buf = torch.zeros(args.per_steps, dtype=torch.float32, device=device)
    value_buf = torch.zeros(args.per_steps, dtype=torch.float32, device=device)
    obs_terrain_grid_buf = torch.zeros(
        (args.per_steps, args.mlp_feat_dim, 10, 20, 10),
        dtype=torch.float16,
        device="cpu"
    )


    # reset
    dfops.meta_solver()
    Reset(env, dfops.opponent_id)
    maneuver_lib.reset_lib()

    #获取初始观测
    next_obs_self, next_obs_target, next_terrain_voxel = get_dogfight_obs(env, env.world.fighters[0], env.world.fighters[1])
    next_obs_self = torch.as_tensor(next_obs_self, dtype=torch.float32, device=device)
    next_obs_target = torch.as_tensor(next_obs_target, dtype=torch.float32, device=device)
    next_done = torch.zeros(1, dtype=torch.float32, device=device)
    with torch.no_grad():
        next_terrain_grid = ppo_agent.actor_critic.encode_terrain(next_terrain_voxel)

    prev_blue_bloods = env.world.fighters[1].combat_data.bloods

    #记录文件使用的num
    if args.continue_train:
        record_num = args.continue_epoch
    else:
        record_num = 0

    #记录训练消耗的时间
    start_time = time.time()
    end_time = start_time

    #计算epoch梳理
    epochs = int(args.total_timesteps / (args.per_steps * args.procs))

    #记录奖励
    epi_return = 0
    epi_turns = []

    #开始训练循环
    for epoch in range(epochs):
        # 统计数据
        red_win_num = 0
        blue_win_num = 0
        draw_num = 0
        red_fall_num = 0
        blue_fall_num = 0
        red_harm_num = 0
        blue_harm_num = 0

        #记录每个步长各项详细奖励的表头
        if p_id == 0 and epoch %10 == 0:
            reward_data_name = os.path.join(reward_dir, f"reward_data_{epoch + 1 + args.continue_epoch}.csv")
            with open(reward_data_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["step", "step_reward", "r_los_body","r_rel_pos", "r_closure", "r_ma_alpha", "r_stability", "r_fire", "r_bloods", "r_alt","red_win_reward", "blue_win_reward",
                     "red_fall_reward","blue_fall_reward", "draw_reward"])

        rms_update_batch = []

        #学习率退火
        if args.anneal_lr:
            global_epoch = epoch + args.continue_epoch if args.continue_train else epoch
            frac = max(0.0, 1.0 - global_epoch / epochs)
            lr_now = args.lr * frac
            for param_group in ppo_agent.ac_optimizer.param_groups:
                param_group["lr"] = lr_now

        # rollout
        for step in range(args.per_steps):
            for i in range(2):
                #红机控制指令
                if i == 0:
                    obs_self_buf[step] = next_obs_self
                    obs_target_buf[step] = next_obs_target
                    obs_terrain_grid_buf[step].copy_(next_terrain_grid.detach().cpu().to(torch.float16))
                    done_buf[step] = next_done
                    #网络输出动作
                    with torch.no_grad():
                        action, logprob, _, value = ppo_agent.actor_critic.get_action_and_value(next_obs_self, next_obs_target, next_terrain_grid.to(device))
                        action = action.squeeze(0)
                        logprob = logprob.squeeze(0)
                        value = value.squeeze(0)
                        value_buf[step] = value

                    action_buf[step] =  action
                    logprob_buf[step] = logprob

                    a = action.squeeze(0)
                    ele_ang = a[0].item() * 90
                    azi_ang = a[1].item() * 180
                    dis = 400 + (1 + a[2].item()) * 800

                    #将dogf输出的目标点给到训练好的avoidance网络
                    action = dogf2avoidance(ele_ang, azi_ang, dis, next_terrain_grid, env, env.world.fighters[0], ppo_agent.avoidance_pi)
                    sim_in_list[0].control_input = action

                if i == 1:
                    action = dfops.sampled_policy(env, maneuver_lib)
                    sim_in_list[1].control_input = action

            #更新环境
            for i in range(10):
                terminal = env.update(p_id,sim_in_list)
                if terminal >=0:
                    break

            next_obs_self, next_obs_target, next_terrain_voxel = get_dogfight_obs(env, env.world.fighters[0], env.world.fighters[1])

            reward, info = reward_func(env, prev_blue_bloods, terminal)
            prev_blue_bloods = env.world.fighters[1].combat_data.bloods

            epi_return += reward
            # 记录轨迹
            if p_id == 0 and record_num % 10 == 0:
                record_trajectory(env, step, record_num, file_dir)

            #进程0每隔10个epoch记录每步奖励
            if p_id == 0 and epoch % 10 == 0:
                with open(reward_data_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [step, reward, info["r_los_body"],info["r_rel_pos"],info["r_closure"] , info["r_ma_alpha"], info["r_stability"], info["r_fire"], info["r_bloods"], info["r_alt"],\
                         info["red_win_reward"], info["blue_win_reward"],info["red_fall_reward"], info["blue_fall_reward"], info["draw_reward"]])

            if terminal >= 0:
                if p_id == 0:
                    print(f'terminal: {terminal}')
                epi_turns.append(epi_return)
                epi_return = 0
                record_num += 1
                next_done = True
                #统计胜率
                red_win_num, blue_win_num, draw_num, red_fall_num, blue_fall_num, red_harm_num, blue_harm_num = win_rate(env, dfops, terminal, red_win_num, blue_win_num, draw_num, red_fall_num, blue_fall_num, red_harm_num, blue_harm_num)

                #reset
                dfops.meta_solver()
                Reset(env, dfops.opponent_id)
                maneuver_lib.reset_lib()
                next_obs_self, next_obs_target, next_terrain_voxel = get_dogfight_obs(env, env.world.fighters[0], env.world.fighters[1])

            else:
                next_done = False

            #更新观测
            next_obs_self = torch.as_tensor(next_obs_self, dtype=torch.float32, device=device)
            next_obs_target = torch.as_tensor(next_obs_target, dtype=torch.float32, device=device)
            with torch.no_grad():
                next_terrain_grid = ppo_agent.actor_critic.encode_terrain(next_terrain_voxel)

            reward_scale = np.sqrt(rewards_rms.var + 1e-8)
            rewards_buf[step] = torch.as_tensor(reward / reward_scale, dtype=torch.float32, device=device)
            next_obs_self = torch.as_tensor(next_obs_self, dtype=torch.float32, device=device)
            next_obs_target = torch.as_tensor(next_obs_target, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

            rms_update_batch += [reward]

        # update rms
        if args.torch_dist:
            rms_mean, rms_std = statistics_scalar_torch(rms_update_batch)
            rms_mean = rms_mean.cpu().item()
            rms_std = rms_std.cpu().item()
            global_rms_batch_len, _ = statistics_scalar_torch([len(rms_update_batch)])
            global_rms_batch_len = args.procs * (global_rms_batch_len.cpu().item())
        else:
            rms_mean, rms_std = mpi_statistics_scalar(rms_update_batch)
            global_rms_batch_len = mpi_sum(len(rms_update_batch))
        rewards_rms.update_from_moments(rms_mean, rms_std ** 2, global_rms_batch_len)

        #GAE + return
        with torch.no_grad():
            next_value = ppo_agent.actor_critic.get_value(next_obs_self, next_obs_target,next_terrain_grid.to(device))
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros(1, dtype=torch.float32, device=device)

            for t in reversed(range(args.per_steps)):
                if t == args.per_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = value_buf[t + 1]

                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - value_buf[t]
                lastgaelam = delta + args.gamma * args.lam * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buf


        data = {
            "obs_self": obs_self_buf.detach().cpu(),
            "obs_target": obs_target_buf.detach().cpu(),
            "obs_terrain_grid": obs_terrain_grid_buf,
            "act": action_buf.detach().cpu(),
            "logp": logprob_buf.detach().cpu(),
            "adv": advantages.detach().cpu(),
            "ret": returns.detach().cpu(),
            "val": value_buf.detach().cpu(),
        }

        #update
        update_out = ppo_agent.update(data)
        if update_out is None:
            continue
        pi_l, v_l, ent, kl, cf = update_out



        #平均各进程事件
        if args.torch_dist:
            ava_red_win_num = average_x_torch(red_win_num)
            avg_blue_win_num = average_x_torch(blue_win_num)
            ava_draw_num = average_x_torch(draw_num)
            ava_red_fall_num = average_x_torch(red_fall_num)
            avg_blue_fall_num = average_x_torch(blue_fall_num)
            ava_red_harm_num = average_x_torch(red_harm_num)
            avg_blue_harm_num = average_x_torch(blue_harm_num)
            epi_mean_turns = average_x_torch(np.mean(epi_turns) if len(epi_turns) > 0 else 0.0,)


        else:
            ava_red_win_num = mpi_avg(red_win_num)
            avg_blue_win_num = mpi_avg(blue_win_num)
            ava_draw_num = mpi_avg(draw_num)
            ava_red_fall_num = mpi_avg(red_fall_num)
            avg_blue_fall_num = mpi_avg(blue_fall_num)
            ava_red_harm_num = mpi_avg(red_harm_num)
            avg_blue_harm_num = mpi_avg(blue_harm_num)
            epi_mean_turns = mpi_avg(np.mean(epi_turns) if len(epi_turns) > 0 else 0.0,)

        games = ava_red_win_num.item() + avg_blue_win_num.item() + ava_draw_num.item()
        if games > 0:
            ava_red_win_rate = ava_red_win_num.item() / games
            ava_blue_win_rate = avg_blue_win_num.item() / games
            ava_draw_rate = ava_draw_num.item() / games
            ava_red_fall_rate = ava_red_fall_num.item() / games
            ava_blue_fall_rate = avg_blue_fall_num.item() / games
        else:
            ava_red_win_rate = 0.0
            ava_blue_win_rate = 0.0
            ava_draw_rate = 0.0
            ava_red_fall_rate = 0.0
            ava_blue_fall_rate = 0.0

        #保存模型到历史策略池
        if epoch % 50 == 0:
            if p_id == 0:
                history_dir = os.path.join(args.model_dir, str(args.seed),"ActorCritic","history_model")
                os.makedirs(history_dir, exist_ok=True)

                self_play_name = os.path.join(
                    history_dir,
                    f"history_policy_{history_index}.pt"
                )
                torch.save(ppo_agent.actor_critic, self_play_name)
            history_index += 1
        dfops.update_opponent_list(epoch, ava_red_win_rate, ava_blue_win_rate, history_index)


        #记录训练曲线
        if p_id == 0:
            tb_writer.add_scalar("learning_rate", float(lr_now), epoch)
            tb_writer.add_scalar("epi_return", float(epi_mean_turns), epoch)
            tb_writer.add_scalar("pi_loss", float(pi_l), epoch)
            tb_writer.add_scalar("v_loss", float(v_l), epoch)
            tb_writer.add_scalar("entropy", float(ent), epoch)
            tb_writer.add_scalar("kl", float(kl), epoch)
            tb_writer.add_scalar("cf", float(cf), epoch)
            tb_writer.add_scalar("rms_mean", float(np.mean(rms_update_batch)), epoch)
            if args.continue_train:
                with open(file_name, "a", newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + args.continue_epoch + 1, np.mean(epi_turns) if len(epi_turns) > 0 else 0.0,np.mean(rms_update_batch), pi_l, v_l, ent, kl, cf, ava_red_win_num.item(),\
                                     avg_blue_win_num.item(), ava_draw_num.item(), ava_red_fall_num.item(), avg_blue_fall_num.item(), ava_red_harm_num.item(), avg_blue_harm_num.item(),\
                                     ava_red_win_rate, ava_blue_win_rate, ava_draw_rate, ava_red_fall_rate, ava_blue_fall_rate, dfops.opponent_list[dfops.opponent_id]['type']])
            else:
                with open(file_name, "a", newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, np.mean(epi_turns) if len(epi_turns) > 0 else 0.0,np.mean(rms_update_batch), pi_l, v_l, ent, kl, cf,\
                                     ava_red_win_num.item(), avg_blue_win_num.item(), ava_draw_num.item(), ava_red_fall_num.item(), avg_blue_fall_num.item(),ava_red_harm_num.item(), avg_blue_harm_num.item(),\
                                     ava_red_win_rate, ava_blue_win_rate, ava_draw_rate, ava_red_fall_rate, ava_blue_fall_rate,dfops.opponent_list[dfops.opponent_id]['type']])

            print("epoch:", epoch if not args.continue_train else epoch + args.continue_epoch + 1,"step_mean_return:", np.mean(rms_update_batch), "epi_return:", float(epi_mean_turns), "pi_loss:", pi_l,\
                  "v_loss:", v_l, "entropy:", ent, "kl:", kl, "cf:", cf)
            print("red_win_num:", ava_red_win_num.item(),
                  "blue_win_num:", avg_blue_win_num.item(), "draw_num:", ava_draw_num.item(), "red_fall:", ava_red_fall_num.item(),"blue_fall:", avg_blue_fall_num.item(), "red_harm:", ava_red_harm_num.item(),"blue_harm:", avg_blue_harm_num.item())
            print("red_win_rate:", ava_red_win_rate, "blue_win_rate:", ava_blue_win_rate, "draw_rate:", ava_draw_rate,
                  "red_fall_rate:", ava_red_fall_rate, "blue_fall_rate:", ava_blue_fall_rate)
            print(f"当前策略得分 = {dfops.sample_scores}, 当前策略分布为 = [{dfops.opponent_list[0]['type'], dfops.opponent_list[1]['type'], dfops.opponent_list[2]['type'], dfops.opponent_list[3]['type']}]")
            print('消耗总时间 = ', time.time() - start_time,f'epoch = {epoch}消耗时间 = ',time.time() - end_time)
            end_time = time.time()

            if epoch % 5 == 0:
                if args.continue_train:
                    net_save_epoch = epoch + args.continue_epoch
                else:
                    net_save_epoch = epoch
                ppo_agent.save(net_save_epoch)

    if p_id == 0 and tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def init_process(rank, size, pargs, fn, backend = 'gloo'):
    t_dist.init_process_group(backend, init_method='tcp://localhost:64543', rank=rank, world_size=size,
                              timeout=datetime.timedelta(0, 300))
    fn(pargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=525)
    parser.add_argument('--model_dir', type=str, default=".\\output")

    parser.add_argument('--epoch_train_iters', type=int, default=4)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--target_kl', type=float, default=0.01)

    parser.add_argument('--per_steps', type=int, default=4096)
    parser.add_argument('--procs', type=int, default=3)
    parser.add_argument('--num_minibatches', type=int, default=8)

    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--anneal_lr', type=bool, default=True)
    parser.add_argument('--total_timesteps', type=int, default=10000000)
    parser.add_argument('--sim_max_steps', type=int, default=18000)
    parser.add_argument('--torch_dist', type=bool, default=False)
    parser.add_argument('--trainable', type=bool, default=True)

    parser.add_argument('--ent_coef', type=float, default=0.005)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--clip_vloss', type=bool, default=True)

    parser.add_argument('--init_kar', type=float, default=0.20)

    parser.add_argument('--avo_obs_self_dim', type=int, default=20)
    parser.add_argument('--avoidance_act_dim', type=int, default=4)

    parser.add_argument('--dogf_obs_self_dim', type=int, default=14)
    parser.add_argument('--obs_target_dim', type=int, default=9)
    parser.add_argument('--obs_terrain_dim', type=int, default=256)
    parser.add_argument('--dogfight_act_dim', type=int, default=3)
    parser.add_argument('--mlp_feat_dim', type=int, default=64)


    parser.add_argument('--azimuth_range', type=float, default=(-30, 30, 2))
    parser.add_argument('--elevation_range', type=float, default=(-30, 30, 2))
    parser.add_argument('--step_m', type=float, default=400.0)
    parser.add_argument('--max_range', type=float, default = 2000.0 )
    parser.add_argument('--accuracy', type=float, default=20.0)
    parser.add_argument('--max_iter', type=float, default=8)
    parser.add_argument('--voxel_mlp_ckpt', type=str,
                        default='./output/pretrain_voxel_mlp/voxel_mlp_pretrained_best.pt')

    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--continue_epoch', type=int, default=0)
    parser.add_argument('--continue_sf', type=bool, default=False)
    parser.add_argument('--sf_history_index', type=int, default=0) #储存的历史网络策略数量
    parser.add_argument('--history_sf_num', type=int, default=0)  #策略池中历史网络策略数量
    parser.add_argument('--history_start_index', type=int, default=0) #加载的历史网络策略起始编号



    pargs = parser.parse_args()

    if pargs.torch_dist:
        size = pargs.procs
        processes = []
        mp.set_start_method('spawn')
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, pargs, train))
            p.start()
            processes.append(p)

    else:
        mpi_fork(pargs.procs)
        train(pargs)









