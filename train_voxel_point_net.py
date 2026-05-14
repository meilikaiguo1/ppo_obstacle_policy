import os
import json
import glob
import math
import random
import argparse
from typing import List, Tuple
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def build_point_features(local_pts_norm: torch.Tensor) -> torch.Tensor:
    """
    local_pts_norm: [B, P, 3] or [N, 3]
    return: [..., 6] = [x, y, z, r, r_xy, |z|]
    """
    x = local_pts_norm[..., 0:1]
    y = local_pts_norm[..., 1:2]
    z = local_pts_norm[..., 2:3]

    r = torch.norm(local_pts_norm, dim=-1, keepdim=True)
    r_xy = torch.norm(local_pts_norm[..., :2], dim=-1, keepdim=True)
    abs_z = torch.abs(z)

    return torch.cat([x, y, z, r, r_xy, abs_z], dim=-1)


class VoxelCSVPretrainDataset(Dataset):
    """
    读取你收集的 voxel_data_*.csv
    需要包含列：
        points_json
        occupancy
        log_count
        mean_x mean_y mean_z
        std_x std_y std_z
        min_r max_r mean_abs_z
    """
    def __init__(self, csv_files: List[str]):
        super().__init__()
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)

        required_cols = [
            "points_json",
            "occupancy",
            "log_count",
            "mean_x", "mean_y", "mean_z",
            "std_x", "std_y", "std_z",
            "min_r", "max_r", "mean_abs_z",
        ]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pts = json.loads(row["points_json"])
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float32)

        target = {
            "occupancy": np.float32(row["occupancy"]),
            "log_count": np.float32(row["log_count"]),
            "mean_xyz": np.array([row["mean_x"], row["mean_y"], row["mean_z"]], dtype=np.float32),
            "std_xyz": np.array([row["std_x"], row["std_y"], row["std_z"]], dtype=np.float32),
            "min_r": np.float32(row["min_r"]),
            "max_r": np.float32(row["max_r"]),
            "mean_abs_z": np.float32(row["mean_abs_z"]),
        }

        return pts, target


def collate_fn(batch):
    """
    把变长点集 pad 成 [B, Pmax, 3]
    """
    points_list, target_list = zip(*batch)

    batch_size = len(points_list)
    max_points = max(p.shape[0] for p in points_list)

    points_pad = torch.zeros((batch_size, max_points, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_points), dtype=torch.bool)

    for i, pts in enumerate(points_list):
        n = pts.shape[0]
        if n > 0:
            points_pad[i, :n] = torch.from_numpy(pts)
            mask[i, :n] = True

    occupancy = torch.tensor([t["occupancy"] for t in target_list], dtype=torch.float32)
    log_count = torch.tensor([t["log_count"] for t in target_list], dtype=torch.float32)
    mean_xyz = torch.tensor(np.stack([t["mean_xyz"] for t in target_list], axis=0), dtype=torch.float32)
    std_xyz = torch.tensor(np.stack([t["std_xyz"] for t in target_list], axis=0), dtype=torch.float32)
    min_r = torch.tensor([t["min_r"] for t in target_list], dtype=torch.float32)
    max_r = torch.tensor([t["max_r"] for t in target_list], dtype=torch.float32)
    mean_abs_z = torch.tensor([t["mean_abs_z"] for t in target_list], dtype=torch.float32)

    target = {
        "occupancy": occupancy,
        "log_count": log_count,
        "mean_xyz": mean_xyz,
        "std_xyz": std_xyz,
        "min_r": min_r,
        "max_r": max_r,
        "mean_abs_z": mean_abs_z,
    }
    return points_pad, mask, target


class VoxelPointMLP(nn.Module):
    """
    - self.mlp
    - self.voxel_proj

    这样训练完可以直接：
        actor_critic.voxel_mlp.load_state_dict(...)
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

    def forward(self, points_xyz: torch.Tensor, mask: torch.Tensor):
        """
        points_xyz: [B, P, 3]
        mask:       [B, P]
        return:     [B, out_dim]
        """
        B, P, _ = points_xyz.shape

        point_feat = build_point_features(points_xyz)   # [B, P, 6]
        feat = self.mlp(point_feat)                     # [B, P, mlp_feat_dim]

        mask_f = mask.unsqueeze(-1).float()             # [B, P, 1]
        count = mask_f.sum(dim=1)                       # [B, 1]
        count_clamped = count.clamp_min(1.0)

        feat_sum = (feat * mask_f).sum(dim=1)
        feat_mean = feat_sum / count_clamped

        feat_for_max = feat.masked_fill(~mask.unsqueeze(-1), -1e9)
        feat_max = feat_for_max.max(dim=1).values
        empty_rows = (count.squeeze(-1) == 0)
        if empty_rows.any():
            feat_max[empty_rows] = 0.0

        occ_flag = (count > 0).float()                  # [B, 1]
        log_count = torch.log1p(count)                  # [B, 1]

        agg_feat = torch.cat([feat_mean, feat_max, occ_flag, log_count], dim=-1)
        voxel_feat = self.voxel_proj(agg_feat)          # [B, out_dim]
        return voxel_feat


class VoxelPointMLPPretrainModel(nn.Module):
    """
    预训练模型：
    encoder = VoxelPointMLP
    head    = 体素统计量预测头
    """
    def __init__(self, mlp_feat_dim=64, out_dim=64, hidden_dim=(32, 32)):
        super().__init__()
        self.encoder = VoxelPointMLP(
            mlp_feat_dim=mlp_feat_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            in_dim=6,
        )

        self.occ_head = nn.Sequential(
            layer_init(nn.Linear(out_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=0.01)
        )

        # 10维:
        # log_count(1), mean_xyz(3), std_xyz(3), min_r(1), max_r(1), mean_abs_z(1)
        self.reg_head = nn.Sequential(
            layer_init(nn.Linear(out_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 10), std=0.01)
        )

    def forward(self, points_xyz: torch.Tensor, mask: torch.Tensor):
        feat = self.encoder(points_xyz, mask)
        occ_logit = self.occ_head(feat).squeeze(-1)   # [B]
        reg = self.reg_head(feat)                     # [B, 10]
        return occ_logit, reg


def compute_losses(occ_logit, reg_pred, target, geom_loss_weight=1.0):
    """
    reg_pred:
      0      -> log_count
      1:4    -> mean_xyz
      4:7    -> std_xyz
      7      -> min_r
      8      -> max_r
      9      -> mean_abs_z
    """
    occ_target = target["occupancy"]
    nonempty = occ_target > 0.5

    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()

    loss_occ = bce(occ_logit, occ_target)

    reg_target = torch.cat([
        target["log_count"].unsqueeze(-1),
        target["mean_xyz"],
        target["std_xyz"],
        target["min_r"].unsqueeze(-1),
        target["max_r"].unsqueeze(-1),
        target["mean_abs_z"].unsqueeze(-1),
    ], dim=-1)

    loss_log_count = smooth_l1(reg_pred[:, 0], reg_target[:, 0])

    if nonempty.any():
        loss_geom = smooth_l1(reg_pred[nonempty, 1:], reg_target[nonempty, 1:])
    else:
        loss_geom = torch.zeros((), device=occ_logit.device)

    total_loss = loss_occ + 0.5 * loss_log_count + geom_loss_weight * loss_geom

    loss_dict = {
        "loss_total": total_loss,
        "loss_occ": loss_occ.detach(),
        "loss_log_count": loss_log_count.detach(),
        "loss_geom": loss_geom.detach(),
    }
    return total_loss, loss_dict


def run_one_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    stats = {
        "loss_total": 0.0,
        "loss_occ": 0.0,
        "loss_log_count": 0.0,
        "loss_geom": 0.0,
        "num_batches": 0,
    }

    for points_xyz, mask, target in loader:
        points_xyz = points_xyz.to(device)
        mask = mask.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        with torch.set_grad_enabled(train):
            occ_logit, reg_pred = model(points_xyz, mask)
            total_loss, loss_dict = compute_losses(occ_logit, reg_pred, target)

            if train:
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        stats["loss_total"] += float(loss_dict["loss_total"])
        stats["loss_occ"] += float(loss_dict["loss_occ"])
        stats["loss_log_count"] += float(loss_dict["loss_log_count"])
        stats["loss_geom"] += float(loss_dict["loss_geom"])
        stats["num_batches"] += 1

    for k in ["loss_total", "loss_occ", "loss_log_count", "loss_geom"]:
        stats[k] /= max(stats["num_batches"], 1)

    return stats


def main(args):
    set_seed(args.seed)


    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "voxel_data_*.csv")))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在 {args.data_dir} 下没有找到 voxel_data_*.csv")

    print(f"找到 {len(csv_files)} 个 CSV 文件")
    dataset = VoxelCSVPretrainDataset(csv_files)
    print(f"总样本数: {len(dataset)}")

    n_train = int(len(dataset) * args.train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoxelPointMLPPretrainModel(
        mlp_feat_dim=args.mlp_feat_dim,
        out_dim=args.out_dim,
        hidden_dim=tuple(args.hidden_dim),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "loss_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total",
            "val_total",
            "train_occ",
            "val_occ",
            "train_log_count",
            "val_log_count",
            "train_geom",
            "val_geom",
            "lr",
        ])

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_stats = run_one_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_stats['loss_total']:.6f}, "
            f"val_total={val_stats['loss_total']:.6f}, "
            f"train_occ={train_stats['loss_occ']:.6f}, "
            f"val_occ={val_stats['loss_occ']:.6f}, "
            f"train_geom={train_stats['loss_geom']:.6f}, "
            f"val_geom={val_stats['loss_geom']:.6f}"
        )
        current_lr = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats["loss_total"],
                val_stats["loss_total"],
                train_stats["loss_occ"],
                val_stats["loss_occ"],
                train_stats["loss_log_count"],
                val_stats["loss_log_count"],
                train_stats["loss_geom"],
                val_stats["loss_geom"],
                current_lr,
            ])

        # 保存 latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "args": vars(args),
        }, os.path.join(save_dir, "pretrain_full_latest.pt"))

        # 只保存 encoder，方便 PPO 直接加载
        torch.save(
            model.encoder.state_dict(),
            os.path.join(save_dir, "voxel_mlp_pretrained_latest.pt")
        )

        # 保存 best
        if val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "args": vars(args),
            }, os.path.join(save_dir, "pretrain_full_best.pt"))

            torch.save(
                model.encoder.state_dict(),
                os.path.join(save_dir, "voxel_mlp_pretrained_best.pt")
            )

    print("训练完成")
    print(f"最佳验证集损失: {best_val:.6f}")
    print(f"encoder 权重保存在: {os.path.join(save_dir, 'voxel_mlp_pretrained_best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./output/logfiles/410/voxel_data")
    parser.add_argument("--save_dir", type=str, default="./output/pretrain_voxel_mlp")
    parser.add_argument("--seed", type=int, default=410)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)

    # 和 PPO 里的 voxel_mlp 保持一致
    parser.add_argument("--mlp_feat_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[32, 32])

    args = parser.parse_args()
    main(args)