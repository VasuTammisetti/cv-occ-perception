"""
Voxel grid definition and coordinate transforms.

This module owns the mapping between continuous 3D space (LiDAR and ego
coordinates, in metres) and discrete voxel indices. Everything downstream in
the GT pipeline (aggregation, labeling, visibility) relies on these
conventions, so they are defined once here and reused everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelGrid:
    """A fixed-resolution voxel grid defined in the ego-vehicle frame.

    The grid is an axis-aligned box. Its extent along each axis is given by a
    (min, max) pair in metres, and every voxel is a cube of side voxel_size
    metres.

    Defaults follow the Occ3D-nuScenes convention: an 80 m by 80 m region
    around the ego vehicle, a tight vertical band from just below the ground to
    a few metres up, and 0.4 m voxels. With these values the grid is
    200 x 200 x 16 voxels.
    """

    # (min, max) bounds in metres, in the ego frame.
    x_range: tuple[float, float] = (-40.0, 40.0)
    y_range: tuple[float, float] = (-40.0, 40.0)
    z_range: tuple[float, float] = (-1.0, 5.4)
    voxel_size: float = 0.4

    def __post_init__(self) -> None:
        self.min_bound = np.array(
            [self.x_range[0], self.y_range[0], self.z_range[0]], dtype=np.float32
        )
        self.max_bound = np.array(
            [self.x_range[1], self.y_range[1], self.z_range[1]], dtype=np.float32
        )

        # Number of voxels along each axis. Round to guard against float error
        # (for example 80 / 0.4 landing at 199.9999) before casting to int, but
        # refuse anything that is not cleanly divisible, otherwise the top slab
        # of the grid silently falls outside every voxel.
        extent = self.max_bound - self.min_bound
        self.dims = np.round(extent / self.voxel_size).astype(int)  # (X, Y, Z)
        residual = extent - self.dims * self.voxel_size
        if np.any(np.abs(residual) > 1e-4):
            raise ValueError(
                f"grid extent {extent} is not a multiple of "
                f"voxel_size {self.voxel_size}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid dimensions as a plain (X, Y, Z) tuple."""
        return tuple(int(d) for d in self.dims)

    def points_to_indices(self, points: np.ndarray) -> np.ndarray:
        """Convert (N, 3) point coordinates to (N, 3) integer voxel indices.

        A point at coordinate p maps to index floor((p - min_bound) / size).
        We shift so the grid corner sits at the origin, scale by voxel size,
        and floor to get the containing cell. Extra columns such as intensity
        are ignored.

        Indices are not range-checked here; use valid_mask to filter points
        that fall outside the grid. The math is done in float64 so that points
        sitting exactly on a voxel boundary do not flip into the neighbouring
        cell due to float32 rounding.
        """
        shifted = points[:, :3].astype(np.float64) - self.min_bound
        return np.floor(shifted / self.voxel_size).astype(int)

    def indices_to_centers(self, indices: np.ndarray) -> np.ndarray:
        """Convert (N, 3) voxel indices to (N, 3) cell-centre coordinates.

        The inverse of points_to_indices, up to quantisation. The 0.5 offset
        returns the centre of each cell rather than its lower corner, which is
        what you want when treating a voxel as a single 3D point. No range
        check is performed; see valid_mask.
        """
        return self.min_bound + (indices + 0.5) * self.voxel_size

    def valid_mask(self, indices: np.ndarray) -> np.ndarray:
        """Boolean (N,) mask: True where an index lies inside the grid.

        Out-of-range points are reported as invalid rather than clamped.
        Clamping would collapse many distant points onto the boundary voxels
        and create spurious occupancy there, so callers filter these points out
        instead.
        """
        in_lower = np.all(indices >= 0, axis=1)
        in_upper = np.all(indices < self.dims, axis=1)
        return in_lower & in_upper

    def points_to_valid_indices(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper: voxel indices plus an in-grid mask.

        Every consumer of this class (aggregation, labeling, visibility) needs
        the same two steps, convert then filter, so they are bundled here.
        Returns (indices, mask); callers keep only indices[mask].
        """
        indices = self.points_to_indices(points)
        return indices, self.valid_mask(indices)


def voxelize_reduce(
    indices: np.ndarray,
    grid_shape: tuple[int, int, int],
    values: np.ndarray | None = None,
    min_points_per_voxel: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reduce per-point voxel indices to per-voxel results.

    Shared by the binary voxelizer and the semantic labeler so the flatten,
    group, and unflatten logic lives in exactly one place. Points are grouped
    by voxel through a single linear index, which is cheaper than a
    lexicographic sort over 3-column rows and scales to the full dataset.

    Parameters
    ----------
    indices : (N, 3) int
        In-grid voxel indices, one per point. Must already be filtered to the
        grid bounds, since flattening assumes every index is valid and
        ravel_multi_index raises on out-of-range input.
    grid_shape : (X, Y, Z)
        Dimensions used to flatten and unflatten linear indices.
    values : (N,) int, optional
        A per-point integer label. When given, each occupied voxel's majority
        label is returned. When None, only occupancy and counts are returned
        (the binary case).
    min_points_per_voxel : int, default 1
        A voxel is kept only if it holds at least this many points.

    Returns
    -------
    vox_xyz : (M, 3) int
        Indices of the kept (occupied) voxels.
    counts : (M,) int
        Number of points in each kept voxel.
    majority : (M,) int or None
        Majority label per kept voxel (ties go to the smallest class id, which
        is numpy's argmax default), or None when values was None.
    """
    if indices.shape[0] == 0:
        empty_i = np.empty((0, 3), dtype=int)
        empty_c = np.empty((0,), dtype=int)
        empty_m = None if values is None else np.empty((0,), dtype=int)
        return empty_i, empty_c, empty_m

    flat = np.ravel_multi_index(indices.T, grid_shape)
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)

    keep = counts >= min_points_per_voxel
    vox_xyz = np.stack(np.unravel_index(uniq[keep], grid_shape), axis=1)
    kept_counts = counts[keep]

    majority = None
    if values is not None:
        # Per-voxel majority label via a compact (voxel, class) count table.
        # Only occupied voxels appear as rows, so this stays small regardless
        # of grid size.
        n_classes = int(values.max()) + 1 if values.size else 1
        table = np.zeros((uniq.shape[0], n_classes), dtype=np.int64)
        np.add.at(table, (inv, values), 1)
        majority_all = table.argmax(axis=1)
        majority = majority_all[keep]

    return vox_xyz, kept_counts, majority


if __name__ == "__main__":
    # Minimal self-check: grid size, a round-trip through a known point, and the
    # boundary points that should or should not survive valid_mask.
    grid = VoxelGrid()
    print("Grid shape (X, Y, Z):", grid.shape)
    print("Total voxels:", int(np.prod(grid.shape)))

    sample = np.array([[12.34, -5.67, 0.89]], dtype=np.float32)
    idx, mask = grid.points_to_valid_indices(sample)
    center = grid.indices_to_centers(idx)
    print("Point:", sample[0], "-> index:", idx[0], "-> centre:", center[0])
    assert mask[0]
    assert np.all(np.abs(center - sample) <= grid.voxel_size / 2 + 1e-6)

    # A point exactly on the upper edge of the grid belongs to no voxel.
    outside = np.array([[40.0, 0.0, 0.0]], dtype=np.float32)
    _, outside_mask = grid.points_to_valid_indices(outside)
    assert not outside_mask[0]

    print("Round-trip and boundary checks passed.")