"""
Render the aggregated LiDAR point cloud as a bird's-eye-view image.

Standalone visualization helper, separate from the GT pipeline. It reuses the
same multi-sweep aggregation and sensor-to-ego transform the pipeline uses, so
the points shown are exactly the cloud the occupancy GT is built from, in the
same ego frame and extent as the occupancy bird's-eye view. Points are colored
by height to give the top-down view some structure.

This needs the nuScenes devkit (it reads raw sweeps). It is a visualization aid
only; nothing here is part of GT generation or training.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from occperc.gt.lidar_aggregation import aggregate_lidar
from occperc.gt.voxel_grid import VoxelGrid
from occperc.gt.voxelizer import apply_transform, sensor_to_ego_transform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the aggregated LiDAR cloud as a bird's-eye view."
    )
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes root.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=5,
                        help="Keyframes into the scene, matching the GT frame.")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--out", default="lidar_bev.png")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Walk into the scene to the chosen keyframe, stopping safely at the end.
    scene = nusc.scene[args.scene_index]
    sample_token = scene["first_sample_token"]
    for _ in range(args.frame):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt

    # Aggregate the same cloud the pipeline uses, and bring it into the ego
    # frame with the same transform the voxelizer applies.
    cloud = aggregate_lidar(nusc, sample_token, n_sweeps=args.n_sweeps)
    transform = sensor_to_ego_transform(nusc, sample_token)
    pts = apply_transform(cloud.points, transform)  # (N, 3) in ego frame

    # Clip to the occupancy grid extent so the view matches the occupancy BEV.
    grid = VoxelGrid()
    x_min, x_max = grid.x_range
    y_min, y_max = grid.y_range
    in_view = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
    )
    pts = pts[in_view]

    print(f"Sample: {sample_token}")
    print(f"Points in view: {pts.shape[0]} (from {cloud.num_points} aggregated)")

    fig, ax = plt.subplots(figsize=(6, 6))
    # Color by height (z) so the ground and taller structures are separable.
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=pts[:, 2], cmap="viridis", s=0.5, linewidths=0,
    )
    ax.plot(0, 0, marker="^", color="red", markersize=10, label="ego vehicle")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Aggregated LiDAR ({args.n_sweeps} sweeps), bird's-eye view")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("height z (m)")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved image to {args.out}")


if __name__ == "__main__":
    main()