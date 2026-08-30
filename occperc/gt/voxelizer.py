"""
Voxelization: aggregated point cloud to occupancy grid.

Takes the dense cloud produced by lidar_aggregation and reduces it to a boolean
occupancy grid over the VoxelGrid. Two steps matter here.

First, a frame correction. The aggregated cloud is expressed in the LIDAR_TOP
sensor frame, but the voxel grid is defined in the ego frame. The two differ by
the LiDAR's mounting extrinsic, chiefly a vertical offset of about 1.8 m since
the sensor sits on the roof. Voxelizing without applying this transform would
shift all occupancy up by several voxels. So we bring points into the ego frame
first, using the keyframe's calibrated_sensor record.

Second, reduction. Aggregation deliberately produces many points per voxel; we
collapse them to the set of occupied cells via the shared voxelize_reduce
helper. A voxel is occupied if it contains at least min_points_per_voxel points
(default 1). We favour completeness over noise-filtering: a missed obstacle is a
worse error than a spurious one for navigation, and multi-sweep aggregation
already suppresses most stray returns. The threshold is exposed for
noise-sensitive settings.

Reference: nuScenes devkit, nuscenes.utils.geometry_utils.transform_matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from occperc.gt.lidar_aggregation import LIDAR_CHANNEL, AggregatedCloud
from occperc.gt.voxel_grid import VoxelGrid, voxelize_reduce

# The sensor-to-ego extrinsic is a static property of a keyframe: the same
# sample_token always yields the same matrix, and a full GT build touches each
# keyframe more than once (binary voxelize plus label voxelize). We memoize by
# token. This is keyed on the token alone; running two nuScenes versions in one
# process would need a wider key.
_TRANSFORM_CACHE: dict[str, np.ndarray] = {}


@dataclass
class OccupancyGrid:
    """A boolean occupancy grid plus the indices of its occupied voxels.

    Attributes
    ----------
    occupied : (X, Y, Z) bool
        True where a voxel contains at least the required number of points.
    occupied_indices : (M, 3) int
        The integer indices of the occupied voxels. Redundant with the occupied
        array but convenient for downstream code (labeling, visibility) that
        iterates over occupied cells directly.
    grid : VoxelGrid
        The grid definition these results were computed against.
    """

    occupied: np.ndarray
    occupied_indices: np.ndarray
    grid: VoxelGrid

    @property
    def num_occupied(self) -> int:
        return int(self.occupied_indices.shape[0])

    @property
    def occupancy_rate(self) -> float:
        """Fraction of voxels that are occupied, a sparsity indicator."""
        return self.num_occupied / float(self.occupied.size)


def sensor_to_ego_transform(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """Return the 4x4 matrix taking LIDAR_TOP sensor-frame points to the ego frame.

    The keyframe's LiDAR sample_data references a calibrated_sensor record
    holding the sensor's translation and rotation relative to the ego body.
    transform_matrix assembles these into a homogeneous transform. Results are
    memoized by sample token, since every keyframe is voxelized more than once
    during a full GT build.
    """
    cached = _TRANSFORM_CACHE.get(sample_token)
    if cached is not None:
        return cached

    sample_rec = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample_rec["data"][LIDAR_CHANNEL])
    calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    transform = transform_matrix(
        translation=np.array(calib["translation"]),
        rotation=Quaternion(calib["rotation"]),
        inverse=False,
    )
    _TRANSFORM_CACHE[sample_token] = transform
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points and return (N, 3)."""
    n = points.shape[0]
    homogeneous = np.concatenate([points, np.ones((n, 1), dtype=points.dtype)], axis=1)
    transformed = homogeneous @ transform.T  # (N, 4)
    return transformed[:, :3]


def voxelize(
    cloud: AggregatedCloud,
    grid: VoxelGrid,
    nusc: NuScenes,
    sample_token: str,
    min_points_per_voxel: int = 1,
) -> OccupancyGrid:
    """Reduce an aggregated cloud to an occupancy grid.

    Parameters
    ----------
    cloud : AggregatedCloud
        Points in the LIDAR_TOP sensor frame (from aggregate_lidar).
    grid : VoxelGrid
        Target grid, defined in the ego frame.
    nusc, sample_token :
        Needed to look up the sensor-to-ego extrinsic for this keyframe.
    min_points_per_voxel : int, default 1
        A voxel is occupied once it holds at least this many points.

    Returns
    -------
    OccupancyGrid
    """
    # 1. Bring points from the sensor frame into the ego frame the grid uses.
    transform = sensor_to_ego_transform(nusc, sample_token)
    points_ego = apply_transform(cloud.points, transform)

    # 2. Convert to voxel indices and drop anything outside the grid. This
    # filter must run before the reduction, because ravel_multi_index inside
    # voxelize_reduce only accepts in-bounds indices.
    indices, mask = grid.points_to_valid_indices(points_ego)
    indices = indices[mask]

    occupied = np.zeros(grid.shape, dtype=bool)

    # 3. Reduce many points to occupied cells via the shared helper. The binary
    # case passes no per-point values, so it just gets occupied voxels and
    # counts back.
    vox_xyz, _, _ = voxelize_reduce(
        indices, grid.shape, values=None, min_points_per_voxel=min_points_per_voxel
    )

    if vox_xyz.shape[0] == 0:
        return OccupancyGrid(occupied=occupied, occupied_indices=vox_xyz, grid=grid)

    occupied[vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]] = True
    return OccupancyGrid(occupied=occupied, occupied_indices=vox_xyz, grid=grid)


if __name__ == "__main__":
    import argparse

    from occperc.gt.lidar_aggregation import aggregate_lidar

    parser = argparse.ArgumentParser(description="Voxelize one keyframe.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--min-points", type=int, default=1)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]
    for _ in range(5):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt

    cloud = aggregate_lidar(nusc, sample_token, n_sweeps=args.n_sweeps)
    occ = voxelize(cloud, VoxelGrid(), nusc, sample_token,
                   min_points_per_voxel=args.min_points)

    print(f"Aggregated points:  {cloud.num_points:>7d}")
    print(f"Occupied voxels:    {occ.num_occupied:>7d}")
    print(f"Grid shape:         {occ.occupied.shape}")
    print(f"Occupancy rate:     {occ.occupancy_rate * 100:.2f}% of voxels")
    print(f"min_points_per_voxel: {args.min_points}")

    # Sanity check: the sensor-to-ego transform should lift points by about
    # 1.8 m in z.
    z_before = cloud.points[:, 2].mean()
    z_after = apply_transform(cloud.points,
                              sensor_to_ego_transform(nusc, sample_token))[:, 2].mean()
    print(f"Mean z: sensor {z_before:+.2f} m -> ego {z_after:+.2f} m "
          f"(shift {z_after - z_before:+.2f} m)")