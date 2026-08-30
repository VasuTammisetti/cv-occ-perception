"""
Visibility via ray casting: free, occupied, unobserved.

A binary occupancy grid conflates two very different kinds of empty. A voxel
with no points might be genuinely free, meaning a LiDAR beam passed through it
on the way to a hit beyond, or it might be occluded, with no beam ever reaching
it. For a learning target the distinction is essential: labeling occluded space
as free trains a model to hallucinate free space it cannot see. So we resolve
three states, following Occ3D-nuScenes and OpenOccupancy:

    UNOBSERVED (0) : no beam reached this voxel (occluded or out of range).
    FREE       (1) : a beam demonstrably passed through this voxel.
    OCCUPIED   (2) : a beam terminated here (a point landed in it).

We recover these by ray casting. Every LiDAR return is a beam from the sensor
origin to the hit point. Voxels the ray passes through are FREE; the endpoint
voxel is OCCUPIED; voxels no ray touches stay UNOBSERVED.

Traversal is a batched Amanatides and Woo (1987) DDA: every ray carries its own
per-axis t_max and t_delta, and in each iteration all rays advance one voxel
along whichever axis crosses a boundary soonest. Unlike fixed-step sampling
this is exact, since a ray cannot skip a voxel it passes through no matter how
grazing the angle, while still being fully vectorised across rays.

Ray origin: the beams start at the sensor, not the ego origin. In the ego frame
that is the LiDAR mount position, which is the translation of the sensor-to-ego
transform (about 0.94 m forward and 1.84 m up). Casting from the ego origin
instead would push all free space about 1.8 m too low.

Known approximation for multi-sweep origins: the aggregated cloud spans about 10
sweeps of ego motion, but every ray is cast from the keyframe sensor position.
For points returned several sweeps ago the true beam origin was metres away, so
free space along older sweeps' rays is biased by the ego displacement in
between. Occ3D handles this by casting each sweep with its own sensor origin;
doing the same here would require lidar_aggregation to keep a per-point
sweep-origin array. Until that lands, treat the free/unobserved boundary near
fast motion as approximate.

Cost: the aggregated cloud has hundreds of thousands of points and each ray
crosses roughly 50 to 150 voxels, so full casting is tens of millions of
updates. A subsample parameter casts from every Nth point, which keeps the demo
fast; turn it up for denser free space. The DDA itself is correct at any
subsample.

Reference: Occ3D-nuScenes and OpenOccupancy for the three-state convention;
Amanatides and Woo (1987), "A fast voxel traversal algorithm for ray tracing".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nuscenes.nuscenes import NuScenes

from occperc.gt.lidar_aggregation import AggregatedCloud
from occperc.gt.voxel_grid import VoxelGrid
from occperc.gt.voxelizer import apply_transform, sensor_to_ego_transform

# Visibility state codes. UNOBSERVED is 0 so a zero-filled grid starts as
# "we know nothing", which is the correct prior before any ray is cast.
UNOBSERVED = 0
FREE = 1
OCCUPIED = 2

# Sentinel standing in for infinity in the t_max and t_delta tables. Rays with
# a near-zero direction component along an axis never cross that axis's
# boundaries; a large value keeps argmin away from such axes.
_INF = np.float64(1e18)


@dataclass
class VisibilityGrid:
    """Three-state visibility grid over the voxel grid.

    Attributes
    ----------
    state : (X, Y, Z) uint8
        Per-voxel visibility: UNOBSERVED, FREE, or OCCUPIED.
    grid : VoxelGrid
        The grid definition these results were computed against.
    """

    state: np.ndarray
    grid: VoxelGrid

    def counts(self) -> dict[str, int]:
        hist = np.bincount(self.state.ravel(), minlength=3)
        return {
            "unobserved": int(hist[UNOBSERVED]),
            "free": int(hist[FREE]),
            "occupied": int(hist[OCCUPIED]),
        }

    @property
    def observed_mask(self) -> np.ndarray:
        """Boolean grid: True where a voxel was observed (free or occupied).

        This is exactly the mask the DL loss uses to ignore unobserved voxels
        during supervision.
        """
        return self.state != UNOBSERVED


def _dda_free_marks(
    origin: np.ndarray,
    unit: np.ndarray,
    lengths: np.ndarray,
    grid: VoxelGrid,
    state: np.ndarray,
) -> None:
    """Mark FREE every voxel each ray passes through, in place on state.

    Batched Amanatides and Woo: all rays advance one voxel per iteration, each
    along its own next-boundary axis. This is exact, with no fixed-step
    sampling, so no voxel can be skipped by a grazing chord.

    origin is (3,), unit is (N, 3) unit direction vectors, lengths is (N,) ray
    lengths in metres. Rays whose length is not positive must be excluded by
    the caller.
    """
    size = float(grid.voxel_size)

    # Starting voxel of every ray. It may lie outside the grid, but marking is
    # masked and the DDA arithmetic is happy with out-of-range indices.
    p = np.floor((origin - grid.min_bound.astype(np.float64)) / size)
    p = np.broadcast_to(p, unit.shape).astype(np.int64).copy()  # (N, 3)

    # Step direction per axis: +1, -1, or 0 depending on the ray's direction.
    step_dir = np.sign(unit)
    step_dir[step_dir == 0] = 1.0  # axis unused by this ray; value irrelevant

    # t_delta: ray length needed to cross one voxel along each axis.
    abs_u = np.abs(unit)
    t_delta = np.where(abs_u > 1e-12, size / np.maximum(abs_u, 1e-12), _INF)

    # t_max: ray parameter at which each axis next crosses a boundary. For a
    # positive direction the next boundary is the upper face of the current
    # voxel (p + 1); for a negative direction it is the lower face (p).
    next_face = np.where(step_dir > 0, p + 1.0, p) * size \
        + grid.min_bound.astype(np.float64)
    t_max = np.where(abs_u > 1e-12, (next_face - origin) / unit, _INF)

    n_rays = lengths.shape[0]
    rows = np.arange(n_rays)
    active = np.ones(n_rays, dtype=bool)

    # Each iteration: every still-active ray crosses its nearest boundary and
    # enters the neighbouring voxel, which the beam passes through.
    while True:
        axis = np.argmin(t_max, axis=1)          # (N,) next boundary axis
        t = t_max[rows, axis]                    # (N,) entry time into next voxel
        still = active & (t < lengths)           # ray continues past boundary
        if not np.any(still):
            break
        p[still, axis[still]] += step_dir[still, axis[still]].astype(int)
        t_max[still, axis[still]] += t_delta[still, axis[still]]

        idx = p[still]
        valid = np.all((idx >= 0) & (idx < grid.dims), axis=1)
        if np.any(valid):
            v = idx[valid]
            state[v[:, 0], v[:, 1], v[:, 2]] = FREE


def cast_visibility(
    cloud: AggregatedCloud,
    grid: VoxelGrid,
    nusc: NuScenes,
    sample_token: str,
    subsample: int = 4,
) -> VisibilityGrid:
    """Ray-cast the aggregated cloud into a three-state visibility grid.

    Parameters
    ----------
    cloud : AggregatedCloud
        Points in the LIDAR_TOP sensor frame.
    grid : VoxelGrid
        Target grid, in the ego frame.
    nusc, sample_token :
        Used for the sensor-to-ego transform, which positions both the points
        and the ray origin.
    subsample : int, default 4
        Cast from every subsample-th point. A value of 1 casts from all points
        (slowest, densest free space); larger is faster and sparser.

    Returns
    -------
    VisibilityGrid

    Notes
    -----
    All rays are cast from the keyframe's sensor position; see the module
    docstring for the multi-sweep-origin approximation this implies.
    """
    transform = sensor_to_ego_transform(nusc, sample_token)

    # Endpoints (hit points) in the ego frame. OCCUPIED is marked from the full
    # cloud regardless of subsample, matching the binary voxelizer.
    endpoints_all = apply_transform(cloud.points, transform)  # (N, 3)
    endpoints = endpoints_all[::subsample] if subsample > 1 else endpoints_all
    endpoints = endpoints.astype(np.float64)

    state = np.full(grid.shape, UNOBSERVED, dtype=np.uint8)

    # Empty cloud: nothing to cast, everything stays UNOBSERVED.
    if endpoints.shape[0] == 0:
        return VisibilityGrid(state=state, grid=grid)

    # Ray origin: the sensor position in the ego frame is the translation column
    # of the transform. All beams emanate from this single point.
    origin = transform[:3, 3].astype(np.float64)

    directions = endpoints - origin
    lengths = np.linalg.norm(directions, axis=1)
    nonzero = lengths > 1e-6  # drop degenerate zero-length rays
    unit = directions[nonzero] / lengths[nonzero, None]
    ray_lengths = lengths[nonzero]

    # Mark FREE along each ray with the exact batched DDA.
    _dda_free_marks(origin, unit, ray_lengths, grid, state)

    # Mark OCCUPIED at the endpoints, overriding FREE where they coincide.
    end_idx, end_mask = grid.points_to_valid_indices(endpoints_all)
    end_idx = end_idx[end_mask]
    if end_idx.shape[0]:
        state[end_idx[:, 0], end_idx[:, 1], end_idx[:, 2]] = OCCUPIED

    return VisibilityGrid(state=state, grid=grid)


if __name__ == "__main__":
    import argparse

    from occperc.gt.lidar_aggregation import aggregate_lidar

    parser = argparse.ArgumentParser(description="Visibility ray-casting demo.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=4)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]
    # Walk a few keyframes into the scene, but stop safely at the end.
    for _ in range(5):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt

    cloud = aggregate_lidar(nusc, sample_token, n_sweeps=args.n_sweeps)
    vis = cast_visibility(cloud, VoxelGrid(), nusc, sample_token,
                          subsample=args.subsample)

    c = vis.counts()
    total = sum(c.values())
    print(f"Rays cast from:     {cloud.num_points // args.subsample:>7d} points "
          f"(subsample {args.subsample})")
    print(f"Occupied voxels:    {c['occupied']:>7d} ({100*c['occupied']/total:.2f}%)")
    print(f"Free voxels:        {c['free']:>7d} ({100*c['free']/total:.2f}%)")
    print(f"Unobserved voxels:  {c['unobserved']:>7d} "
          f"({100*c['unobserved']/total:.2f}%)")

    # Sanity: occupied count here should be close to the binary voxelizer's
    # occupied count, since both mark endpoint voxels. Free space should be a
    # larger volume than occupied, because beams sweep out a lot of empty air.
    assert c["free"] > c["occupied"], "expected more free than occupied voxels"
    assert c["occupied"] > 0, "no occupied voxels, endpoints not marked"
    print("Sanity checks passed (free > occupied, occupied > 0).")