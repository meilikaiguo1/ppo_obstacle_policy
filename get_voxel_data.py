import os
import json
import argparse
import numpy as np
import torch
import csv

from Spinup.mpi_torch_utils import proc_id, mpi_fork
from Simset.sim_set import make_sim_env, Reset


def build_voxel_pretrain_target(pts: np.ndarray):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)

    if pts.shape[0] == 0:
        return {
            "occupancy": 0.0,
            "log_count": 0.0,
            "mean_x": 0.0,
            "mean_y": 0.0,
            "mean_z": 0.0,
            "std_x": 0.0,
            "std_y": 0.0,
            "std_z": 0.0,
            "min_r": 0.0,
            "max_r": 0.0,
            "mean_abs_z": 0.0,
        }

    r = np.linalg.norm(pts, axis=1)
    mean_xyz = pts.mean(axis=0)
    std_xyz = pts.std(axis=0)

    return {
        "occupancy": 1.0,
        "log_count": float(np.log1p(len(pts))),
        "mean_x": float(mean_xyz[0]),
        "mean_y": float(mean_xyz[1]),
        "mean_z": float(mean_xyz[2]),
        "std_x": float(std_xyz[0]),
        "std_y": float(std_xyz[1]),
        "std_z": float(std_xyz[2]),
        "min_r": float(r.min()),
        "max_r": float(r.max()),
        "mean_abs_z": float(np.abs(pts[:, 2]).mean()),
    }


def sample_random_action():
    """
    采样一个随机动作，范围和训练时动作定义一致
    a = [load_cmd, omega_cmd, rudder_cmd, thrust_cmd]
    """
    a0 = np.random.uniform(-0.6, 0.9)
    a1 = np.random.uniform(-0.6, 0.6)
    a2 = np.random.uniform(-0.3, 0.3)
    a3 = np.random.uniform(-1.0, 1.0)
    return np.array([a0, a1, a2, a3], dtype=np.float32)


def apply_action_to_env(sim_in_list, action):
    """
    把 [-1, 1] 动作映射到环境的 control_input
    """
    a0, a1, a2, a3 = action.tolist()

    load = 9 * a0 if a0 > 0 else 3 * a0
    omega = 300 * a1
    rudder = a2
    thrust = 0.75 + 0.25 * a3

    load = min(9.0, max(-3.0, load))
    omega = min(300.0, max(-300.0, omega))
    rudder = min(1.0, max(-1.0, rudder))
    thrust = min(1.0, max(0.1, thrust))

    ctrl = sim_in_list[0] if isinstance(sim_in_list, (list, tuple)) else sim_in_list
    ctrl.control_input = [
        thrust,
        (load / 9.0) if load > 0 else (load / 3.0),
        omega / 300.0,
        rudder,
    ]


def collect_current_frame_voxel_samples(env, remain_nonempty, remain_empty, empty_keep_prob=0.02):
    """
    收集当前帧体素样本：
    - points 存标准化后的局部坐标
    - target 存统计标签
    """
    fighter = env.world.fighters[0]

    terrain_hits = env.terrain_sensor.terrain_scan(fighter)
    terrain_voxels = env.voxel.terrain_to_voxel(fighter, terrain_hits)

    # 这里 normal_voxels 就是“标准化后的点坐标”
    _, normal_voxels = env.voxel.compute_voxel_local_coordinates(terrain_voxels["voxel_dict"])
    num_voxels = env.voxel.nx * env.voxel.ny * env.voxel.nz

    samples = []
    add_nonempty = 0
    add_empty = 0

    for k in range(num_voxels):
        pts = normal_voxels.get(k, np.zeros((0, 3), dtype=np.float32))
        pts = np.asarray(pts, dtype=np.float32)

        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)

        if pts.shape[0] > 0:
            if add_nonempty >= remain_nonempty:
                continue
            voxel_type = 1
            add_nonempty += 1
        else:
            if add_empty >= remain_empty:
                continue

            # 空体素随机下采样
            if np.random.rand() > empty_keep_prob:
                continue

            voxel_type = 0
            add_empty += 1

        target = build_voxel_pretrain_target(pts)

        samples.append({
            "voxel_id": int(k),
            "num_points": int(pts.shape[0]),
            "voxel_type": int(voxel_type),
            "points": pts.astype(np.float32),   # 直接把标准化后的点存下来
            "target": target,
        })

        if add_nonempty >= remain_nonempty and add_empty >= remain_empty:
            break

    return samples, add_nonempty, add_empty


def get_voxel_data(args):
    # 仿真环境
    env, sim_in_list = make_sim_env(args)

    # 随机种子
    p_id = proc_id()
    seed = args.seed + p_id * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 保存路径
    file_dir = os.path.join(".", "output", "logfiles", str(args.seed))
    data_dir = os.path.join(file_dir, "voxel_data")
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, f"voxel_data_{p_id}.csv")

    # 写表头
    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "voxel_id",
            "num_points",
            "voxel_type",
            "points_json",   # 新增：标准化后的点坐标字符串
            "occupancy",
            "log_count",
            "mean_x", "mean_y", "mean_z",
            "std_x", "std_y", "std_z",
            "min_r", "max_r", "mean_abs_z",
        ]
        writer.writerow(header)

    num_empty = 0
    num_nonempty = 0

    Reset(env)

    for step in range(args.per_steps):
        samples, add_nonempty, add_empty = collect_current_frame_voxel_samples(
            env,
            remain_nonempty=max(args.target_nonempty_total - num_nonempty, 0),
            remain_empty=max(args.target_empty_total - num_empty, 0),
            empty_keep_prob=args.empty_keep_prob,
        )

        num_empty += add_empty
        num_nonempty += add_nonempty

        with open(file_name, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for sample in samples:
                t = sample["target"]

                # 把标准化后的点坐标存成字符串
                points_json = json.dumps(sample["points"].tolist(), ensure_ascii=False)

                row = [
                    sample["voxel_id"],
                    sample["num_points"],
                    sample["voxel_type"],
                    points_json,
                    t["occupancy"],
                    t["log_count"],
                    t["mean_x"], t["mean_y"], t["mean_z"],
                    t["std_x"], t["std_y"], t["std_z"],
                    t["min_r"], t["max_r"], t["mean_abs_z"],
                ]
                writer.writerow(row)

        # 环境推进
        action = sample_random_action()
        apply_action_to_env(sim_in_list, action)

        terminal = env.update(sim_in_list, 100, 100)
        if terminal >= 0:
            Reset(env)

        if num_empty >= args.target_empty_total and num_nonempty >= args.target_nonempty_total:
            break

        if (step + 1) % 100 == 0 and p_id == 0:
            print(
                f"step={step + 1}, "
                f"nonempty={num_nonempty}/{args.target_nonempty_total}, "
                f"empty={num_empty}/{args.target_empty_total}"
            )

    print(
        f"[rank {p_id}] finished. "
        f"nonempty={num_nonempty}, empty={num_empty}, file={file_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--procs", type=int, default=20)

    # 每个进程自己的目标数量
    parser.add_argument("--target_nonempty_total", type=int, default=5000)
    parser.add_argument("--target_empty_total", type=int, default=2500)

    parser.add_argument("--per_steps", type=int, default=5000)
    parser.add_argument("--empty_keep_prob", type=float, default=0.02)

    # 环境参数
    parser.add_argument("--sim_max_steps", type=int, default=3000)

    parser.add_argument("--azimuth_range", nargs=3, type=float, default=[-30, 30, 2])
    parser.add_argument("--elevation_range", nargs=3, type=float, default=[-30, 30, 2])
    parser.add_argument("--step_m", type=float, default=400.0)
    parser.add_argument("--max_range", type=float, default=2000.0)
    parser.add_argument("--accuracy", type=float, default=20.0)
    parser.add_argument("--max_iter", type=int, default=10)

    pargs = parser.parse_args()

    pargs.azimuth_range = tuple(pargs.azimuth_range)
    pargs.elevation_range = tuple(pargs.elevation_range)

    mpi_fork(pargs.procs)
    get_voxel_data(pargs)