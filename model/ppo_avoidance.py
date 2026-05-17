import os
import torch
from torch.optim import Adam
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from Spinup.mpi_torch_utils import num_procs, proc_id, mpi_avg_grads, sync_params, mpi_avg
from Spinup.torch_distribute import average_gradients_torch, average_x_torch, statistics_scalar_torch, sync_params_torch
from Utils.tensor_util import count_vars
from model.net_avoidance import Avoidance_ActorCritic
import time


class PPOAgent():
    def __init__(self, index,args, trainable = True, torch_dist = False, device = torch.device("cuda:0")):
        self.torch_dist = torch_dist
        self.device = device

        #set save path
        self.save_dir = os.path.join(args.model_dir, "ActorCritic", str(args.seed), "trained_model")
        os.makedirs(self.save_dir, exist_ok=True)

        #set hyperparameters
        self.epoch_train_iters = args.epoch_train_iters
        self.clip_ratio = args.clip_ratio
        self.gamma = args.gamma
        self.lam = args.lam
        self.target_kl = args.target_kl
        self.ent_coef = args.ent_coef
        self.num_minibatches = args.num_minibatches
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.clip_vloss = args.clip_vloss

        self.trainable = trainable


        #set network
        self.act_dim = args.avoidance_act_dim
        self.obs_self_dim = args.avo_obs_self_dim
        self.obs_terrain_dim = args.obs_terrain_dim
        self.mlp_feat_dim = args.mlp_feat_dim
        self.actor_critic = Avoidance_ActorCritic(self.obs_self_dim, self.obs_terrain_dim, self.act_dim, self.mlp_feat_dim).to(self.device)

        if trainable:
            if torch_dist:
                sync_params_torch(self.actor_critic)
            else:
                # 把模型放到cpu上，同步tensor参数
                sync_params(self.actor_critic.cpu())
                # 同步参数后，移回原设备
                self.actor_critic.to(self.device)
            self.ac_optimizer = Adam(self.actor_critic.parameters(), lr=args.lr, betas=(0.9, 0.999))
            self.var_counts = tuple(count_vars(module) for module in [self.actor_critic.actor, self.actor_critic.critic])
            p_id = dist.get_rank() if torch_dist else proc_id()
            if p_id == 0 and index == 0:
                print("Total Parameters: total",sum([p.nelement() for p in self.actor_critic.parameters()]),
                      "actor:", sum([p.nelement() for p in self.actor_critic.actor.parameters()]),
                      "critic:", sum([p.nelement() for p in self.actor_critic.critic.parameters()]))
        #迭代次数
        self.update_count = 0

    def update(self, data):
        self.update_count += 1
        if not self.trainable:
            return None

        #data 在flatten之后传入
        b_obs_self = data["obs_self"]
        b_act = data["act"].float()
        b_logp = data["logp"]
        b_adv = data["adv"]
        b_ret = data["ret"]
        b_val = data["val"]
        b_voxel_grid = data["obs_terrain_grid"]

        batch_size = b_obs_self.shape[0]
        minibatch_size = batch_size // self.num_minibatches
        b_inds = np.arange(batch_size)

        pi_l_old = 0.0
        v_l_old = 0.0
        ent = 0.0
        approx_kl = 0.0
        clipfrac = 0.0

        if len(b_adv) > 1:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        for epoch in range(self.epoch_train_iters):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs_self = b_obs_self[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_voxel_grid = b_voxel_grid[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_act = b_act[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_logp_old = b_logp[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_adv = b_adv[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_ret = b_ret[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_val = b_val[mb_inds].to(self.device, dtype=torch.float32, non_blocking=True)

                _, new_logp, entropy, new_val = self.actor_critic.get_action_and_value(mb_obs_self, mb_voxel_grid, mb_act)

                logratio = new_logp - mb_logp_old
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()

                #policy loss
                pg_loss1 = - ratio * mb_adv
                pg_loss2 = - torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_adv
                pi_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value loss
                if self.clip_vloss:
                    v_loss_unclipped = (new_val - mb_ret) ** 2
                    v_clipped = mb_val + torch.clamp(new_val - mb_val, -self.clip_ratio, self.clip_ratio)
                    v_loss_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pi_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.ac_optimizer.zero_grad()
                loss.backward()

                if self.torch_dist:
                    average_gradients_torch(self.actor_critic)
                else:
                    mpi_avg_grads(self.actor_critic)

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.ac_optimizer.step()

                pi_l_old = pi_loss.item()
                v_l_old = v_loss.item()
                ent = entropy_loss.item()
            if self.torch_dist:
                global_kl = average_x_torch(approx_kl.item())
            else:
                global_kl = mpi_avg(approx_kl.item())
            if self.target_kl is not None and global_kl > self.target_kl:
                break

        return  pi_l_old, v_l_old, ent, approx_kl.item(), clipfrac.item()



    def save(self, epoch):
        torch.save(self.actor_critic.actor.state_dict(), os.path.join(self.save_dir, f"pi_net_{epoch}.pt"))
        torch.save(self.actor_critic.critic.state_dict(), os.path.join(self.save_dir, f"v_net_{epoch}.pt"))
        torch.save(self.actor_critic.network.state_dict(), os.path.join(self.save_dir, f"network_{epoch}.pt"))
        torch.save(self.actor_critic.voxel_mlp.state_dict(), os.path.join(self.save_dir, f"voxel_mlp_{epoch}.pt"))
        torch.save(self.actor_critic.fusion.state_dict(), os.path.join(self.save_dir, f"fusion_{epoch}.pt"))
        torch.save(self.ac_optimizer.state_dict(), os.path.join(self.save_dir, f"ac_optimizer_{epoch}.pt"))