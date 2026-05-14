
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from model.mlp_model import mlp, layer_init
import torch.nn.functional as F

from model.net_avoidance import Avoidance_Actor
from point_cloud import voxel_table_to_grid, build_point_features



def make_group_norm(num_channels: int, preferred_groups: int = 8):
    groups = min(preferred_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

def build_point_features_padded(local_pts_norm: torch.Tensor) -> torch.Tensor:
    """
    local_pts_norm: [..., 3]
    return:        [..., 6] = [x, y, z, r, r_xy, |z|]
    """
    x = local_pts_norm[..., 0:1]
    y = local_pts_norm[..., 1:2]
    z = local_pts_norm[..., 2:3]

    r = torch.norm(local_pts_norm, dim=-1, keepdim=True)
    r_xy = torch.norm(local_pts_norm[..., :2], dim=-1, keepdim=True)
    abs_z = torch.abs(z)

    return torch.cat([x, y, z, r, r_xy, abs_z], dim=-1)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()

        self.conv1 = layer_init(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.gn1 = make_group_norm(out_channels)

        self.conv2 = layer_init(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.gn2 = make_group_norm(out_channels)

        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer_init(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                ),
                make_group_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.dropout(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class VoxelPointMLP(nn.Module):
    """
    单体素输入版本：
        输入:  [N, 3]  一个体素内所有标准化局部点坐标
        输出:  [out_dim]
    """
    def __init__(self, mlp_feat_dim=64, out_dim=64, hidden_dim=(32, 32), in_dim=6):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for h in hidden_dim:
            layers.append(layer_init(nn.Linear(prev_dim, h)))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(layer_init(nn.Linear(prev_dim, mlp_feat_dim)))
        self.mlp = nn.Sequential(*layers)

        self.voxel_proj = nn.Sequential(
            layer_init(nn.Linear(mlp_feat_dim * 2 + 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, out_dim))
        )

        self.out_dim = out_dim
        self.mlp_feat_dim = mlp_feat_dim

    def forward(self, local_pts_norm: torch.Tensor) -> torch.Tensor:
        """
        local_pts_norm: [N, 3]
        return: [out_dim]
        """
        device = next(self.parameters()).device

        if local_pts_norm is None or local_pts_norm.numel() == 0:
            return torch.zeros(self.out_dim, dtype=torch.float32, device=device)

        if local_pts_norm.dim() != 2 or local_pts_norm.shape[-1] != 3:
            raise ValueError(f"local_pts_norm shape error: {local_pts_norm.shape}")

        local_pts_norm = local_pts_norm.to(device=device, dtype=torch.float32)

        point_feat = build_point_features(local_pts_norm)   # [N, 6]
        feat = self.mlp(point_feat)                         # [N, mlp_feat_dim]

        feat_mean = feat.mean(dim=0)                        # [mlp_feat_dim]
        feat_max = feat.max(dim=0).values                   # [mlp_feat_dim]

        occ_flag = torch.tensor([1.0], dtype=torch.float32, device=device)
        log_count = torch.tensor(
            [np.log1p(local_pts_norm.shape[0])],
            dtype=torch.float32,
            device=device
        )

        agg_feat = torch.cat([feat_mean, feat_max, occ_flag, log_count], dim=0)
        voxel_feat = self.voxel_proj(agg_feat)              # [out_dim]
        return voxel_feat



class Voxel3DCNN(nn.Module):
    def __init__(self, in_channels=64, out_dim=256, base_channels=32, dropout_p=0.0):
        super().__init__()

        self.stem = nn.Sequential(
            layer_init(nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)),
            make_group_norm(base_channels),
            nn.ReLU(inplace=True)
        )

        # [B, 32, 10, 20, 10]
        self.stage1 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels, stride=1, dropout_p=0.0),
            ResidualBlock3D(base_channels, base_channels, stride=1, dropout_p=0.0),
        )

        # [B, 32, 10, 20, 10] -> [B, 64, 5, 10, 5]
        self.stage2 = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels * 2, stride=2, dropout_p=0.0),
            ResidualBlock3D(base_channels * 2, base_channels * 2, stride=1, dropout_p=dropout_p),
        )

        # [B, 64, 5, 10, 5]
        self.stage3 = nn.Sequential(
            ResidualBlock3D(base_channels * 2, base_channels * 2, stride=1, dropout_p=dropout_p),
            ResidualBlock3D(base_channels * 2, base_channels * 2, stride=1, dropout_p=dropout_p),
        )

        # 保留粗空间结构
        self.avg_pool = nn.AdaptiveAvgPool3d((2, 4, 2))
        self.max_pool = nn.AdaptiveMaxPool3d((2, 4, 2))

        fc_in_dim = (base_channels * 2) * 2 * 4 * 2 * 2  # ×2 because avg/max concat

        self.fc = nn.Sequential(
            layer_init(nn.Linear(fc_in_dim, 512)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            layer_init(nn.Linear(512, out_dim))
        )

    def forward(self, x):
        # x: [B, C, nx, ny, nz]
        x = self.stem(x)        # [B, 32, 10, 20, 10]
        x = self.stage1(x)      # [B, 32, 10, 20, 10]
        x = self.stage2(x)      # [B, 64,  5, 10,  5]
        x = self.stage3(x)      # [B, 64,  5, 10,  5]

        x_avg = self.avg_pool(x)                # [B, 64, 2, 4, 2]
        x_max = self.max_pool(x)                # [B, 64, 2, 4, 2]
        x = torch.cat([x_avg, x_max], dim=1)    # [B, 128, 2, 4, 2]

        x = x.flatten(1)                        # [B, 2048]
        x = self.fc(x)                          # [B, out_dim]
        return x

class Dogfight_Actor(nn.Module):
    def __init__(self, feat_output, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feat_output, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def forward(self, x):
        return self.net(x)
    def distribution(self, pi):
        std = torch.exp(self.log_std)
        dist = Normal(pi, std)
        return dist
    def log_prob_from_distribution(self, distribution, action):
        return distribution.log_prob(action).sum(axis=-1)



class Dogfight_Critic(nn.Module):
    def __init__(self, feat_output):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feat_output, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class Avoidance_pi(nn.Module):
    def __init__(self, obs_self_dim, terrain_feat_dim, action_dim, share_mlp_output_dim, device = torch.device("cuda:0")):
        super().__init__()

        self.nx = 10
        self.ny = 20
        self.nz = 10
        self.num_voxels = self.nx * self.ny * self.nz
        self.voxel_out_dim = share_mlp_output_dim

        self.voxel_mlp = VoxelPointMLP(
            mlp_feat_dim=share_mlp_output_dim,
            out_dim=share_mlp_output_dim,
            hidden_dim=(32, 32),
            in_dim=6
        )

        self.network = Voxel3DCNN(
            in_channels=share_mlp_output_dim,
            out_dim=terrain_feat_dim
        )

        self.fusion = nn.Sequential(
            layer_init(nn.Linear(terrain_feat_dim + obs_self_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        self.actor = Avoidance_Actor(256, action_dim)

        # 加载模型
        self.voxel_mlp.load_state_dict(
            torch.load(f"./output/avoidance_model/voxel_mlp.pt",map_location=device))
        self.voxel_mlp.eval()

        self.network.load_state_dict(
            torch.load(f"./output/avoidance_model/network.pt",map_location=device))
        self.network.eval()

        self.fusion.load_state_dict(
            torch.load(f"./output/avoidance_model/fusion.pt",map_location=device))
        self.fusion.eval()

        self.actor.load_state_dict(
            torch.load(f"./output/avoidance_model/pi_net.pt",map_location=device))
        self.actor.eval()

    @torch.no_grad()
    def encode_terrain(self, terrain_voxel_dict):
        """
        terrain_voxel_dict[k]: [n_k, 3]，每个体素内全部标准化局部点
        return:
            voxel_grid: [C, nx, ny, nz]
        """
        device = next(self.parameters()).device

        voxel_feat_list = []
        for k in range(self.num_voxels):
            pts = terrain_voxel_dict.get(k, None)

            if pts is None:
                feat = torch.zeros(self.voxel_out_dim, dtype=torch.float32, device=device)
            else:
                if isinstance(pts, np.ndarray):
                    pts = torch.from_numpy(pts)
                pts = pts.to(device=device, dtype=torch.float32)

                if pts.numel() == 0:
                    feat = torch.zeros(self.voxel_out_dim, dtype=torch.float32, device=device)
                else:
                    feat = self.voxel_mlp(pts)  # [C]

            voxel_feat_list.append(feat)

        voxel_feat = torch.stack(voxel_feat_list, dim=0)  # [V, C]
        voxel_grid = voxel_table_to_grid(voxel_feat, self.nx, self.ny, self.nz)  # [C, nx, ny, nz]
        return voxel_grid

    @torch.no_grad()
    def get_features(self, obs_self, voxel_grid):
        """
        obs_self:
            [obs_dim] or [B, obs_dim]
        voxel_grid:
            [C, nx, ny, nz] or [B, C, nx, ny, nz]
        """
        if obs_self.dim() == 1:
            obs_self = obs_self.unsqueeze(0)

        if voxel_grid.dim() == 4:
            voxel_grid = voxel_grid.unsqueeze(0)

        voxel_grid = voxel_grid.to(obs_self.device, dtype=torch.float32)

        terrain_feat = self.network(voxel_grid)  # [B, terrain_feat_dim]
        fused = self.fusion(torch.cat([terrain_feat, obs_self], dim=-1))
        return fused

    @torch.no_grad()
    def forward(self, obs_self, voxel_grid):
        fused = self.get_features(obs_self, voxel_grid)
        return self.actor(fused)



class Dogfight_ActorCritic(nn.Module):
    def __init__(self, obs_self_dim, obs_target_dim, terrain_feat_dim, action_dim, share_mlp_output_dim, device = torch.device("cuda:0")):
        super().__init__()

        self.nx = 10
        self.ny = 20
        self.nz = 10
        self.num_voxels = self.nx * self.ny * self.nz
        self.voxel_out_dim = share_mlp_output_dim

        self.voxel_mlp = VoxelPointMLP(
            mlp_feat_dim=share_mlp_output_dim,
            out_dim=share_mlp_output_dim,
            hidden_dim=(32, 32),
            in_dim=6
        )

        self.network = Voxel3DCNN(
            in_channels=share_mlp_output_dim,
            out_dim=terrain_feat_dim
        )

        self.fusion = nn.Sequential(
            layer_init(nn.Linear(terrain_feat_dim + obs_self_dim + obs_target_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        # 加载模型
        self.voxel_mlp.load_state_dict(
            torch.load(f"./output/avoidance_model/voxel_mlp.pt",map_location=device))

        for p in self.voxel_mlp.parameters():
            p.requires_grad_(False)
        self.voxel_mlp.eval()

        self.network.load_state_dict(
            torch.load(f"./output/avoidance_model/network.pt",map_location=device))
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.network.eval()

        self.actor = Dogfight_Actor(256, action_dim)
        self.critic = Dogfight_Critic(256)

    @torch.no_grad()
    def encode_terrain(self, terrain_voxel_dict):
        """
        terrain_voxel_dict[k]: [n_k, 3]，每个体素内全部标准化局部点
        return:
            voxel_grid: [C, nx, ny, nz]
        """
        device = next(self.parameters()).device

        voxel_feat_list = []
        for k in range(self.num_voxels):
            pts = terrain_voxel_dict.get(k, None)

            if pts is None:
                feat = torch.zeros(self.voxel_out_dim, dtype=torch.float32, device=device)
            else:
                if isinstance(pts, np.ndarray):
                    pts = torch.from_numpy(pts)
                pts = pts.to(device=device, dtype=torch.float32)

                if pts.numel() == 0:
                    feat = torch.zeros(self.voxel_out_dim, dtype=torch.float32, device=device)
                else:
                    feat = self.voxel_mlp(pts)  # [C]

            voxel_feat_list.append(feat)

        voxel_feat = torch.stack(voxel_feat_list, dim=0)  # [V, C]
        voxel_grid = voxel_table_to_grid(voxel_feat, self.nx, self.ny, self.nz)  # [C, nx, ny, nz]
        return voxel_grid

    def get_features(self, obs_self, obs_target, voxel_grid):
        device = next(self.parameters()).device

        obs_self = torch.as_tensor(obs_self, dtype=torch.float32, device=device)
        obs_target = torch.as_tensor(obs_target, dtype=torch.float32, device=device)
        voxel_grid = torch.as_tensor(voxel_grid, dtype=torch.float32, device=device)

        if obs_self.dim() == 1:
            obs_self = obs_self.unsqueeze(0)

        if obs_target.dim() == 1:
            obs_target = obs_target.unsqueeze(0)

        if voxel_grid.dim() == 4:
            voxel_grid = voxel_grid.unsqueeze(0)

        with torch.no_grad():
            terrain_feat = self.network(voxel_grid)

        fused = self.fusion(torch.cat([terrain_feat, obs_self, obs_target], dim=-1))
        return fused


    def get_value(self, obs_self, obs_target, voxel_grid):
        fused = self.get_features(obs_self, obs_target, voxel_grid)
        return self.critic(fused)

    def get_action_and_value(self, obs_self, obs_target, voxel_grid, action=None):
        fused = self.get_features(obs_self, obs_target, voxel_grid)

        mu = self.actor(fused)
        pi = self.actor.distribution(mu)

        if action is None:
            action = pi.sample()

        log_p = self.actor.log_prob_from_distribution(pi, action)
        entropy = pi.entropy().sum(dim=-1)
        value = self.critic(fused)

        return action, log_p, entropy, value



