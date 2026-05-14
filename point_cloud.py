import math
import numpy as np
from WVRENV_PHD.utils.GNCData import ned_to_body
import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxel():
    def __init__(self, x_range = (0, 2000), y_range =(-2000, 2000), z_range =(-1000, 1000), nx = 10, ny = 20, nz = 10):
        """
        :param x_range: 机体系 x轴 范围
        :param y_range: 机体系 y轴 范围
        :param z_range: 机体系 z轴 范围
        :param nx: x 分辨率
        :param ny: y 分辨率
        :param nz: z 分辨率
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # 体素区域范围
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        self.region_min = np.array([x_min, y_min, z_min], dtype=np.float32)
        self.region_max = np.array([x_max, y_max, z_max], dtype=np.float32)

        # 每个体素尺寸
        self.voxel_size_x = (x_max - x_min) / self.nx
        self.voxel_size_y = (y_max - y_min) / self.ny
        self.voxel_size_z = (z_max - z_min) / self.nz
        self.voxel_size = np.array([self.voxel_size_x, self.voxel_size_y, self.voxel_size_z], dtype=np.float32)

        # 计算每个体素中心
        voxel_centers = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    cx = x_min + (ix + 0.5) * self.voxel_size_x
                    cy = y_min + (iy + 0.5) * self.voxel_size_y
                    cz = z_min + (iz + 0.5) * self.voxel_size_z
                    voxel_centers.append([cx, cy, cz])
        self.voxel_centers = np.asarray(voxel_centers, dtype=np.float32)
    def terrain_to_voxel(self, fighter, terrain_hits):
        '''
        :param fighter: 飞行器智能体
        :param terrain_hits: 障碍点，形状一般为 (N, 3)
        :return: 处理成放到神经网络的体素
        '''

        # 飞机坐标与姿态
        fighter_ned = np.asarray(fighter.state.ned_Pos, dtype=np.float32)
        yaw = fighter.fc_data.fYawAngle
        pitch = fighter.fc_data.fPitchAngle
        roll = fighter.fc_data.fRollAngle


        #初始化输出结果
        num_voxels = self.nx * self.ny * self.nz
        points_body = np.zeros((0, 3), dtype=np.float32)
        valid_mask = np.zeros((0,), dtype=bool)
        valid_points = np.zeros((0, 3), dtype=np.float32)
        voxel_indices = np.zeros((0, 3), dtype=np.int32)
        voxel_flat_indices = np.zeros((0,), dtype=np.int32)
        voxel_dict = {i: np.zeros((0, 3), dtype=np.float32) for i in range(num_voxels)}

        # 处理 terrain_hits
        terrain_hits = np.asarray(terrain_hits, dtype=np.float32)

        # 如果传感器没有扫到障碍，直接输出结果
        if terrain_hits.size == 0:
            return {
                "points_body": points_body,              # 所有点(机体系)
                "valid_mask": valid_mask,                # 是否在区域内
                "valid_points_body": valid_points,       # 落在区域内的点
                "voxel_indices": voxel_indices,          # (ix, iy, iz)
                "voxel_flat_indices": voxel_flat_indices,
                "voxel_dict": voxel_dict,                # 每个体素里的点
                "voxel_centers": self.voxel_centers,          # 每个体素中心
                "voxel_size": self.voxel_size,                # 每个体素尺寸
            }

        # 保证 shape 为 (N, 3)
        terrain_hits = terrain_hits.reshape(-1, 3)

        # 地形点从 NED 转到“相对飞机”的机体系
        rel_ned = terrain_hits - fighter_ned.reshape(1, 3)
        points_body = ned_to_body(rel_ned, yaw, pitch, roll).astype(np.float32)

        # 判断哪些点在体素范围内
        valid_mask = np.all(
            (points_body >= self.region_min) & (points_body < self.region_max),
            axis=1
        )
        valid_points = points_body[valid_mask]

        # 如果有障碍点，但都不在体素范围内
        if valid_points.shape[0] == 0:
            return {
                "points_body": points_body,
                "valid_mask": valid_mask,
                "valid_points_body": valid_points,
                "voxel_indices": voxel_indices,
                "voxel_flat_indices": voxel_flat_indices,
                "voxel_dict": voxel_dict,
                "voxel_centers": self.voxel_centers,
                "voxel_size": self.voxel_size,
            }


        # 计算每个有效点属于哪个体素
        shifted = valid_points - self.region_min.reshape(1, 3)

        ix = np.floor(shifted[:, 0] / self.voxel_size_x).astype(np.int32)
        iy = np.floor(shifted[:, 1] / self.voxel_size_y).astype(np.int32)
        iz = np.floor(shifted[:, 2] / self.voxel_size_z).astype(np.int32)

        # 防止边界数值误差
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)

        voxel_indices = np.stack([ix, iy, iz], axis=1)

        # 压平成一维编号
        voxel_flat_indices = ix * (self.ny * self.nz) + iy * self.nz + iz

        # 按体素收集点
        voxel_dict = {i: [] for i in range(num_voxels)}
        for p, vid in zip(valid_points, voxel_flat_indices):
            voxel_dict[int(vid)].append(p)

        for k in voxel_dict:
            if len(voxel_dict[k]) == 0:
                voxel_dict[k] = np.zeros((0, 3), dtype=np.float32)
            else:
                voxel_dict[k] = np.asarray(voxel_dict[k], dtype=np.float32)

        # 返回
        return {
            "points_body": points_body,              # 所有点(机体系)
            "valid_mask": valid_mask,                # 是否在区域内
            "valid_points_body": valid_points,       # 落在区域内的点
            "voxel_indices": voxel_indices,          # (ix, iy, iz)
            "voxel_flat_indices": voxel_flat_indices,
            "voxel_dict": voxel_dict,                # 每个体素里的点
            "voxel_centers": self.voxel_centers,          # 每个体素中心
            "voxel_size": self.voxel_size,                # 每个体素尺寸
        }


    def compute_voxel_local_coordinates(self, voxel_dict):
        """
        计算每个体素内点在该体素局部坐标系下的坐标

        参数
        ----
        voxel_dict : dict  terrain_to_voxel 返回的 voxel_dict
        返回
        ----
        local_coord_dict : dict 每个体素内点的局部坐标（未归一化） local_coord_dict[k].shape = (n_k, 3)
        local_coord_norm_dict : dict 每个体素内点的局部归一化坐标  坐标大致在 [-1, 1] local_coord_norm_dict[k].shape = (n_k, 3)
        """

        voxel_centers = np.asarray(self.voxel_centers, dtype=np.float32)
        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        voxel_half_size = voxel_size / 2.0

        # 防止除0
        voxel_half_size = np.maximum(voxel_half_size, 1e-6)

        local_coord_dict = {}
        local_coord_norm_dict = {}

        for k in voxel_dict:
            pts = np.asarray(voxel_dict[k], dtype=np.float32)   # shape=(n,3) 或 (0,3)
            center = voxel_centers[k]                           # shape=(3,)

            if pts.shape[0] == 0:
                local_coord_dict[k] = np.zeros((0, 3), dtype=np.float32)
                local_coord_norm_dict[k] = np.zeros((0, 3), dtype=np.float32)
            else:
                # 1. 局部坐标：以体素中心为原点
                local_coords = pts - center.reshape(1, 3)

                # 2. 归一化局部坐标：大致落到 [-1, 1]
                local_coords_norm = local_coords / voxel_half_size.reshape(1, 3)

                local_coord_dict[k] = local_coords.astype(np.float32)
                local_coord_norm_dict[k] = local_coords_norm.astype(np.float32)

        return local_coord_dict, local_coord_norm_dict

def build_point_features(local_pts_norm: torch.Tensor) -> torch.Tensor:
    """
    local_pts_norm: [N, 3]
    return: [N, 6] = [x, y, z, r, r_xy, |z|]
    """
    x = local_pts_norm[:, 0:1]
    y = local_pts_norm[:, 1:2]
    z = local_pts_norm[:, 2:3]

    r = torch.norm(local_pts_norm, dim=-1, keepdim=True)
    r_xy = torch.norm(local_pts_norm[:, :2], dim=-1, keepdim=True)
    abs_z = torch.abs(z)

    return torch.cat([x, y, z, r, r_xy, abs_z], dim=-1)



def encode_voxel_points_sum(local_coord_norm_dict, point_mlp):
    """
    local_coord_norm_dict[k]: [n_k, 3]
    return: [num_voxels, out_dim]
    """
    device = next(point_mlp.parameters()).device
    voxel_feature_table = []

    for k in range(len(local_coord_norm_dict)):
        pts = np.asarray(local_coord_norm_dict[k], dtype=np.float32)   # [n_k, 3]

        if pts.shape[0] == 0:
            pts_feat = torch.zeros((0, 6), dtype=torch.float32, device=device)
        else:
            pts_tensor = torch.from_numpy(pts).to(device)              # [n_k, 3]
            pts_feat = build_point_features(pts_tensor)                # [n_k, 6]

        voxel_feat = point_mlp(pts_feat)                               # [out_dim]
        voxel_feature_table.append(voxel_feat)

    voxel_feature_table = torch.stack(voxel_feature_table, dim=0)      # [num_voxels, out_dim]
    return voxel_feature_table


def voxel_table_to_grid(voxel_feature_table, nx = 10, ny =20, nz =10):
    '''
    :param voxel_feature_table:  经过mlp处理过后的体素   [num_vocels, 体素特征维度]
    :param nx:  x方向分辨率
    :param ny:  y方向分辨率
    :param nz:  z方向分辨率
    :return:  [特征维度, nx, ny, nz]
    '''
    num_voxels, C = voxel_feature_table.shape
    assert num_voxels == nx * ny * nz

    grid = voxel_feature_table.view(nx, ny, nz, C)  # [nx, ny, nz, C]
    grid = grid.permute(3, 0, 1, 2).contiguous()      # [C, nx, ny, nz]
    return grid

def encode_voxel_batch_to_grid(terrain_voxel_batch, point_mlp, nx=10, ny=20, nz=10):
    """
    terrain_voxel_batch:
        - 单样本: dict
        - 多样本: list[dict]

    return:
        voxel_batch: [B, C, nx, ny, nz]
    """
    # 单样本兼容
    if isinstance(terrain_voxel_batch, dict):
        terrain_voxel_batch = [terrain_voxel_batch]

    voxel_grid_list = []
    for voxel_dict in terrain_voxel_batch:
        voxel_feature_table = encode_voxel_points_sum(voxel_dict, point_mlp)   # [num_voxels, C]
        voxel_grid = voxel_table_to_grid(voxel_feature_table, nx, ny, nz)       # [C, nx, ny, nz]
        voxel_grid_list.append(voxel_grid)

    voxel_batch = torch.stack(voxel_grid_list, dim=0)                           # [B, C, nx, ny, nz]
    return voxel_batch


