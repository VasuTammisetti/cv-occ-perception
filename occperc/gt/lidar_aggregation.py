"""
LiDAR loading and multi-sweep aggregation.

A single nuScenes LiDAR sweep (32-beam Velodyne) is sparse: at range the returns
are thin and many truly-occupied voxels receive no points. To build usable
occupancy ground truth we aggregate several consecutive sweeps into a denser
cloud.

The sweeps cannot simply be concatenated. Between sweeps the ego vehicle moves,
and each sweep's points are recorded in that sweep's own sensor frame at a
different pose and time. Naively stacking them smears the world. Points must
therefore be transformed into a single common frame before merging. That
transform is the sensor to ego to global chain (and back), which the nuScenes
devkit helper LidarPointCloud.from_file_multisweep performs for us; this module
wraps it with the conventions the rest of the pipeline uses.

Frame convention: the aggregated cloud comes back in the keyframe's LIDAR_TOP
sensor frame. That is not the same as the ego frame, since the two differ by the
calibrated_sensor extrinsic, so downstream code must apply that extrinsic before
feeding these points into ego-frame structures such as VoxelGrid.

Reference: nuScenes devkit, nuscenes.utils.data_classes.LidarPointCloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Sensor we read sweeps from, and the reference sensor the merged cloud is
# expressed in. Both are the top LiDAR.
LIDAR_CHANNEL = "LIDAR_TOP"


@dataclass
class AggregatedCloud:
    """Result of aggregating several sweeps for one keyframe.

    Attributes
    ----------
    points : (N, 3) float32
        Point coordinates (x, y, z) in the keyframe's LIDAR_TOP sensor frame.
        Not the ego frame; see the module docstring.
    times : (N,) float32
        Time offset of each point relative to the reference sweep, in seconds.
        Kept because it is a natural per-point feature and because it is what
        later lets us reason about motion if we choose to.
    """

    points: np.ndarray
    times: np.ndarray

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


def aggregate_lidar(
    nusc: NuScenes,
    sample_token: str,
    n_sweeps: int = 10,
    min_distance: float = 1.0,
) -> AggregatedCloud:
    """Aggregate n_sweeps LiDAR sweeps for a keyframe into one cloud.

    Parameters
    ----------
    nusc : NuScenes
        An initialised devkit handle.
    sample_token : str
        Token of the keyframe (a sample) to aggregate around.
    n_sweeps : int, default 10
        How many sweeps to combine, counting back from the keyframe. More
        sweeps give a denser cloud but smear moving objects across a longer
        time window; dynamic objects are handled in a later module, so a higher
        value is preferred here for density.
    min_distance : float, default 1.0
        Points closer than this (in metres) to the sensor are dropped. This
        removes returns off the ego vehicle itself, such as the roof and mounts.

    Returns
    -------
    AggregatedCloud
        Points in the keyframe's sensor frame, with per-point time offsets.

    Notes
    -----
    Near the start of a scene the sweep linked list runs out before n_sweeps
    sweeps have been consumed. The devkit handles this by simply returning fewer
    sweeps, so the point count is not guaranteed to be about n_sweeps times a
    single sweep. At the very first keyframe of a scene there is no history at
    all and aggregation returns just one sweep.
    """
    sample_rec = nusc.get("sample", sample_token)

    # from_file_multisweep walks the sweep linked list backwards from the
    # keyframe, loads each sweep, transforms every sweep's points into the
    # reference sweep's frame using that sweep's calibrated_sensor and ego_pose,
    # and concatenates them. It returns the merged cloud plus a (1, N) array of
    # per-point time offsets in seconds.
    cloud, times = LidarPointCloud.from_file_multisweep(
        nusc,
        sample_rec,
        chan=LIDAR_CHANNEL,
        ref_chan=LIDAR_CHANNEL,
        nsweeps=n_sweeps,
        min_distance=min_distance,
    )

    # cloud.points is (4, N): rows are x, y, z, intensity. We keep xyz and hand
    # intensity off; occupancy is a geometric label and does not need it.
    points = cloud.points[:3, :].T.astype(np.float32)  # (N, 3)
    times = times.reshape(-1).astype(np.float32)       # (N,)

    return AggregatedCloud(points=points, times=times)


def load_single_sweep(
    nusc: NuScenes,
    sample_token: str,
    min_distance: float = 1.0,
) -> AggregatedCloud:
    """Load just the keyframe sweep, with no aggregation.

    Useful as a baseline to show the density gain from aggregation, and as a
    fallback when a denser cloud is not wanted. Implemented as the n_sweeps=1
    case of the aggregator so the two share a code path.
    """
    return aggregate_lidar(nusc, sample_token, n_sweeps=1, min_distance=min_distance)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate LiDAR for one keyframe.")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes root.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--n-sweeps", type=int, default=10)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Step several keyframes into the scene before aggregating. The first
    # keyframe has no prior sweeps to walk back to, so aggregation there returns
    # a single sweep; a mid-scene sample has a full history behind it and shows
    # the real density gain.
    first_scene = nusc.scene[0]
    sample_token = first_scene["first_sample_token"]
    for _ in range(5):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt

    single = load_single_sweep(nusc, sample_token)
    multi = aggregate_lidar(nusc, sample_token, n_sweeps=args.n_sweeps)

    print(f"Single sweep:      {single.num_points:>7d} points")
    print(f"{args.n_sweeps}-sweep aggregate: {multi.num_points:>7d} points")
    print(f"Density gain:      {multi.num_points / max(single.num_points, 1):.1f}x")
    print("Point cloud shape:", multi.points.shape)
    print("Time range (s):    "
          f"[{multi.times.min():.3f}, {multi.times.max():.3f}]")

    # Aggregation must actually add points. A silent no-op, such as aggregating
    # at the first keyframe of a scene, would otherwise pass unnoticed.
    assert multi.num_points > single.num_points, (
        "aggregation returned no extra points; check the target sample has "
        "sweep history behind it"
    )

    # Convention check: from_file_multisweep reports offsets as
    # (reference_time - sweep_time), so sweeps from the past have positive
    # offsets and the reference sweep sits at zero. Confirm nothing lands in the
    # future, which would signal a frame or time bookkeeping error.
    assert multi.times.min() >= 0.0, "unexpected negative time offset"

    print("Sanity checks passed (density gain and time convention).")