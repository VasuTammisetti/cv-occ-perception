"""
GT generation orchestrator: one sample token -> complete occupancy GT.

Ties the five GT modules into a single call. Given a keyframe, it aggregates
LiDAR sweeps, labels the points by 3D box, voxelizes the labels, and ray-casts
visibility, returning both grids together. This is the entry point the GT
generation script and the walkthrough demo call; the per-module scripts remain
useful for inspecting each stage in isolation.

The two grids answer different questions and are kept separate (see the
labeling and visibility modules): the label grid says what class occupies a
voxel, the visibility grid says whether it was observed and, if so, whether it
is free or occupied. Binary occupancy is not stored; it is exactly
`visibility == OCCUPIED`, so it is derived on load rather than duplicated.
Note that this derivation implicitly assumes occupancy at
`min_points_per_voxel=1`; with a higher threshold the binary voxelizer and
`visibility == OCCUPIED` would disagree. Occupancy is also unaffected by the
ray-cast `subsample`, which trades free-space completeness only.

Labels outside observed voxels are meaningless (label 0 may be empty or unseen
background); use `SampleGT.semantic_grid()` for a loss-ready array that
disambiguates this, or mask with `visibility != UNOBSERVED` directly.

One known accuracy limitation: labels are derived from the keyframe's 3D boxes,
while points come from `n_sweeps` sweeps spanning up to ~0.5 s of ego and
object motion. Points from older sweeps are labeled at their recorded
positions, so points on dynamic objects lag their keyframe-time boxes. This is
the standard tradeoff for multi-sweep occupancy GT; set `n_sweeps=1` to avoid
it at the cost of point density.

SampleGT itself is defined in the serialization module (imported here), so the
training side can load it without pulling in the nuScenes devkit; this module
only produces it.
"""

from __future__ import annotations

from nuscenes.nuscenes import NuScenes

from occperc.gt.lidar_aggregation import aggregate_lidar
from occperc.gt.labeling import (
    build_category_map,
    label_points_by_boxes,
    voxelize_labels,
)
from occperc.gt.serialization import SampleGT, save_sample
from occperc.gt.visibility import FREE, UNOBSERVED, cast_visibility
from occperc.gt.voxel_grid import VoxelGrid

# Occupancy threshold this module's contract depends on. `visibility ==
# OCCUPIED` equals "voxel contains >= 1 labelled point", which is how the label
# voxelizer (voxelize_labels) and the binary voxelizer both behave by default.
# Documented here as the assumption occupancy-on-load relies on; raising it
# would make occupancy and `visibility == OCCUPIED` diverge.
MIN_POINTS_PER_VOXEL = 1


def generate_sample_gt(
    nusc: NuScenes,
    sample_token: str,
    grid: VoxelGrid,
    category_map: dict[str, int],
    n_sweeps: int = 10,
    subsample: int = 4,
) -> SampleGT:
    """Run the full GT pipeline for one keyframe.

    Parameters
    ----------
    nusc, sample_token :
        The dataset handle and the keyframe to process.
    grid : VoxelGrid
        Target grid.
    category_map : dict
        Category-name to class-id mapping (from build_category_map, built
        against the same dataset as nusc).
    n_sweeps : int, default 10
        Sweeps to aggregate for density. Note that points from older sweeps
        are labeled with the keyframe's boxes, so dynamic-object points lag
        slightly (see module docstring).
    subsample : int, default 4
        Ray-casting subsample factor (speed vs. free-space completeness).
        Does not affect occupancy: endpoints are always marked from the full
        cloud.

    Returns
    -------
    SampleGT
    """
    cloud = aggregate_lidar(nusc, sample_token, n_sweeps=n_sweeps)

    labelled = label_points_by_boxes(cloud, nusc, sample_token, category_map)
    label_grid = voxelize_labels(labelled, grid, nusc, sample_token)

    vis = cast_visibility(cloud, grid, nusc, sample_token, subsample=subsample)

    return SampleGT(
        sample_token=sample_token,
        labels=label_grid,
        visibility=vis.state,
        grid=grid,
    )


def iter_scene_samples(nusc: NuScenes, scene_index: int = 0):
    """Yield (index, token) for every keyframe in a scene, in order.

    The index is the keyframe's position within the scene (0, 1, 2, and so on),
    handy for progress reporting and for naming per-frame output files without
    extra bookkeeping.
    """
    scene = nusc.scene[scene_index]
    token = scene["first_sample_token"]
    index = 0
    while token:
        yield index, token
        token = nusc.get("sample", token)["next"]
        index += 1


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate complete GT for a few keyframes."
    )
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Which scene to process.")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="If given, save each SampleGT as a .npz here "
                             "(also exercises the serialization path).")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    category_map = build_category_map(nusc)
    grid = VoxelGrid()

    for index, token in iter_scene_samples(nusc, scene_index=args.scene_index):
        if index >= args.num_frames:
            break

        gt = generate_sample_gt(
            nusc, token, grid, category_map,
            n_sweeps=args.n_sweeps, subsample=args.subsample,
        )

        n_occ = int(gt.occupancy.sum())
        n_free = int((gt.visibility == FREE).sum())
        n_unobs = int((gt.visibility == UNOBSERVED).sum())
        n_labelled = int((gt.semantic_grid() > 0).sum())

        print(f"[{index:>3d}] {token}")
        print(f"      occupied: {n_occ:>7d}   free: {n_free:>7d}   "
              f"unobserved: {n_unobs:>7d}   "
              f"observed+labelled: {n_labelled:>7d}")

        if args.output_dir is not None:
            save_sample(args.output_dir, gt)