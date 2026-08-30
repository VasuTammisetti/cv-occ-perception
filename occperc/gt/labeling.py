"""
Semantic labeling: assign a class to each occupied voxel.

The binary occupancy grid says where space is occupied; this module says what
occupies it. Labels come from the nuScenes 3D bounding boxes: each foreground
object (car, pedestrian, truck, and so on) has an oriented 3D box, and a point
falling inside a box takes that box's class. Points inside no box are assigned a
generic occupied-but-unlabeled class. In practice this is mostly background
(road, vegetation, buildings), which boxes do not describe. A lidarseg-based
refinement for those background points is a natural extension and is left as a
plug-in point rather than baked in here.

Frame convention: get_sample_data returns boxes already expressed in the
LIDAR_TOP sensor frame, which is the frame the aggregated cloud is in before the
sensor-to-ego transform. So point-in-box tests happen in the sensor frame, and
the sensor-to-ego transform is applied afterwards during voxelization, exactly
as in the binary voxelizer. This keeps every stage on one frame.

Coupling to the binary grid: LABEL_UNLABELLED (0) is written for empty voxels
too, so empty and occupied-background are not distinguishable from this grid
alone. The binary occupancy grid is the authority on emptiness; read this grid
only where OccupancyGrid.occupied is True. The demo at the bottom shows the
masking pattern consumers should copy.

Category ids: build_category_map derives ids from the loaded dataset's category
table. These are stable for a given nuScenes version, but if label grids are
persisted to disk, save the mapping alongside them so a rebuild against a
different table cannot silently change class ids.

Reference: nuScenes devkit, NuScenes.get_sample_data and
nuscenes.utils.geometry_utils.points_in_box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box

from occperc.gt.lidar_aggregation import LIDAR_CHANNEL, AggregatedCloud
from occperc.gt.voxel_grid import VoxelGrid, voxelize_reduce
from occperc.gt.voxelizer import apply_transform, sensor_to_ego_transform

# Reserved label for occupied space that no bounding box explains. Kept at 0 so
# it doubles as the empty-or-background default when a grid is zero-filled.
LABEL_UNLABELLED = 0


def build_category_map(nusc: NuScenes) -> dict[str, int]:
    """Map nuScenes category names to contiguous integer class ids.

    Ids start at 1; 0 is reserved for LABEL_UNLABELLED. The mapping is built
    from the dataset's own category table so it stays in step with whatever
    split is loaded, rather than hard-coding a class list. See the module
    docstring for the persistence caveat.
    """
    names = sorted(cat["name"] for cat in nusc.category)
    return {name: i + 1 for i, name in enumerate(names)}


@dataclass
class LabelledCloud:
    """Aggregated points plus a per-point integer class label."""

    points: np.ndarray  # (N, 3) in the LIDAR_TOP sensor frame
    labels: np.ndarray  # (N,) int class ids; LABEL_UNLABELLED where no box

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


def label_points_by_boxes(
    cloud: AggregatedCloud,
    nusc: NuScenes,
    sample_token: str,
    category_map: dict[str, int],
) -> LabelledCloud:
    """Label each aggregated point by the 3D box that contains it.

    Boxes are retrieved in the sensor frame, matching the cloud. For each box we
    test which points fall inside and stamp them with the box's class id. Points
    in no box keep LABEL_UNLABELLED. If two boxes overlap, the later box in
    iteration wins; box overlap is rare and this is not worth resolving more
    cleverly for GT at this resolution. Note that points_in_box treats boundary
    points as inside, so boxes sharing a face will trade points along it.

    The cloud must have been aggregated around this sample_token. The label
    lookup itself is frame-safe, but the pairing is assumed, not checked.
    """
    sample_rec = nusc.get("sample", sample_token)
    lidar_token = sample_rec["data"][LIDAR_CHANNEL]
    # get_sample_data returns (path, boxes, intrinsics); the boxes are in the
    # sensor frame of the given sample_data.
    _, boxes, _ = nusc.get_sample_data(lidar_token)

    points = cloud.points
    labels = np.full(points.shape[0], LABEL_UNLABELLED, dtype=np.int32)

    # points_in_box expects (3, N); transpose once up front.
    points_t = points.T  # (3, N)

    for box in boxes:
        class_id = category_map.get(box.name, LABEL_UNLABELLED)
        if class_id == LABEL_UNLABELLED:
            continue
        inside = points_in_box(box, points_t)  # (N,) bool
        labels[inside] = class_id

    return LabelledCloud(points=points, labels=labels)


def voxelize_labels(
    labelled: LabelledCloud,
    grid: VoxelGrid,
    nusc: NuScenes,
    sample_token: str,
) -> np.ndarray:
    """Voxelize labeled points into an (X, Y, Z) int label grid.

    Reuses the sensor-to-ego transform and the shared reduction helper. Where
    several points share a voxel, the voxel takes the majority label among them
    (ties broken by smallest class id). Empty voxels stay LABEL_UNLABELLED (0);
    see the module docstring on why the binary grid must be consulted to tell
    empty from background.

    Returns
    -------
    (X, Y, Z) int32 grid of class ids.
    """
    transform = sensor_to_ego_transform(nusc, sample_token)
    points_ego = apply_transform(labelled.points, transform)

    indices, mask = grid.points_to_valid_indices(points_ego)
    indices = indices[mask]
    point_labels = labelled.labels[mask]

    label_grid = np.full(grid.shape, LABEL_UNLABELLED, dtype=np.int32)
    if indices.shape[0] == 0:
        return label_grid

    vox_xyz, _, majority = voxelize_reduce(
        indices, grid.shape, values=point_labels
    )
    label_grid[vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]] = majority
    return label_grid


if __name__ == "__main__":
    import argparse

    from occperc.gt.lidar_aggregation import aggregate_lidar

    parser = argparse.ArgumentParser(description="Semantic labeling demo.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--n-sweeps", type=int, default=10)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    category_map = build_category_map(nusc)
    id_to_name = {v: k for k, v in category_map.items()}

    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]
    # Walk a few keyframes into the scene, but stop safely at the end.
    for _ in range(5):
        nxt = nusc.get("sample", sample_token)["next"]
        if not nxt:
            break
        sample_token = nxt

    cloud = aggregate_lidar(nusc, sample_token, n_sweeps=args.n_sweeps)
    labelled = label_points_by_boxes(cloud, nusc, sample_token, category_map)
    label_grid = voxelize_labels(labelled, VoxelGrid(), nusc, sample_token)

    n_labelled_pts = int((labelled.labels != LABEL_UNLABELLED).sum())
    pct = 100 * n_labelled_pts / max(labelled.num_points, 1)
    print(f"Total points:        {labelled.num_points:>7d}")
    print(f"Points inside a box: {n_labelled_pts:>7d} ({pct:.1f}%)")

    # Per-class voxel counts, most common first.
    vox_ids, vox_counts = np.unique(label_grid, return_counts=True)
    print("\nVoxel label breakdown:")
    order = np.argsort(-vox_counts)
    for i in order:
        cid = int(vox_ids[i])
        name = "UNLABELLED (empty or background)" if cid == 0 else id_to_name[cid]
        print(f"  {cid:>2d}  {name:<40s} {int(vox_counts[i]):>7d}")