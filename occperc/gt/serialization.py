"""
Serialization: persist per-sample GT to disk, decoupling generation from use.

GT generation is the expensive stage (sweep aggregation and ray casting); the
DL pipeline should not pay that cost every epoch. So we write each sample's GT
once and let the dataset load ready-made arrays. This module defines the
on-disk contract, and deliberately imports nothing from the generation pipeline
beyond VoxelGrid: the training side must stay free of the nuScenes devkit, and
generation depends on serialization, never the reverse. The SampleGT container
lives here for that reason.

Layout for an output directory:

    out_dir/
        meta.json              run-level metadata: grid config, category map,
                               and a manifest of written samples
        <sample_token>.npz     one file per keyframe: labels and visibility

Storage choice: dense arrays via savez_compressed. The grids are about 640k
voxels but overwhelmingly zero (occupancy is a few percent), and compression
squashes that redundancy to a small fraction of the raw size. This is far
simpler than a sparse format, and the DL pipeline gets dense arrays it can use
directly. Sparse storage is noted as an optimization but not needed at this
scale. The category map lives once in meta.json rather than in every npz, so
class ids are self-describing and cannot silently drift between a save and a
later rebuild (see the labeling module's persistence caveat). Each npz also
records its grid dims, so a file generated against a mismatched grid config is
detectable on load instead of silently misread.

Dtype contract: labels are int32, visibility is uint8. Class id semantics
(0 = background or empty; visibility 0/1/2 = UNOBSERVED/FREE/OCCUPIED) live in
the labeling and visibility modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from occperc.gt.voxel_grid import VoxelGrid

FORMAT_VERSION = 1


@dataclass
class SampleGT:
    """Complete ground truth for one keyframe.

    Attributes
    ----------
    sample_token : str
        The keyframe this GT was generated for.
    labels : (X, Y, Z) int32
        Semantic class id per voxel (0 = unlabelled, background, or empty).
        Only meaningful where visibility is not 0 (UNOBSERVED).
    visibility : (X, Y, Z) uint8
        UNOBSERVED (0), FREE (1), or OCCUPIED (2) per voxel.
    grid : VoxelGrid
        The grid definition used.
    """

    sample_token: str
    labels: np.ndarray
    visibility: np.ndarray
    grid: VoxelGrid

    @property
    def occupancy(self) -> np.ndarray:
        """Binary occupancy, derived from the visibility grid (state OCCUPIED)."""
        return self.visibility == 2

    @property
    def observed_mask(self) -> np.ndarray:
        """Boolean grid: True where a voxel was observed (free or occupied)."""
        return self.visibility != 0

    def semantic_grid(self) -> np.ndarray:
        """Labels with unobserved voxels forced to -1, ready for a masked loss."""
        out = self.labels.copy()
        out[~self.observed_mask] = -1
        return out


def write_meta(
    out_dir: Path,
    grid: VoxelGrid,
    category_map: dict[str, int],
    samples: list[dict] | None = None,
) -> None:
    """Write run-level metadata as meta.json.

    Parameters
    ----------
    out_dir, grid, category_map :
        Output directory, grid definition, and category-name to class-id
        mapping. The grid config lets a reader reconstruct the exact
        VoxelGrid; the category map makes class ids self-describing.
    samples : list of dicts with keys token, file, seconds, optional
        Manifest of every successfully generated keyframe in this directory.
        Written after generation by the GT script, so the manifest always
        matches the files present; it also enables resuming an interrupted
        run, since tokens already listed are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": FORMAT_VERSION,
        "grid": {
            "x_range": list(grid.x_range),
            "y_range": list(grid.y_range),
            "z_range": list(grid.z_range),
            "voxel_size": grid.voxel_size,
            "shape": list(grid.shape),
        },
        "category_map": category_map,
        "num_samples": len(samples) if samples is not None else 0,
        "samples": samples or [],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def save_sample(out_dir: Path, gt: SampleGT) -> Path:
    """Write one sample's GT to out_dir / sample_token.npz.

    Stores the label and visibility grids plus the sample token and the grid
    dims (so a mismatched-grid load can be detected). Binary occupancy is
    derived from visibility on load, not duplicated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{gt.sample_token}.npz"
    np.savez_compressed(
        path,
        labels=gt.labels.astype(np.int32, copy=False),
        visibility=gt.visibility.astype(np.uint8, copy=False),
        sample_token=gt.sample_token,
        grid_dims=np.array(gt.grid.shape),
    )
    return path


def load_meta(out_dir: Path) -> dict:
    """Read meta.json back into a dict."""
    return json.loads((Path(out_dir) / "meta.json").read_text())


def grid_from_meta(meta: dict) -> VoxelGrid:
    """Reconstruct the VoxelGrid described by a meta.json dict."""
    g = meta["grid"]
    return VoxelGrid(
        x_range=tuple(g["x_range"]),
        y_range=tuple(g["y_range"]),
        z_range=tuple(g["z_range"]),
        voxel_size=g["voxel_size"],
    )


def load_sample(
    out_dir: Path,
    sample_token: str,
    grid: VoxelGrid | None = None,
) -> SampleGT:
    """Load one sample's GT back into a SampleGT.

    This is the training-time entry point: no nuScenes devkit involved.

    Parameters
    ----------
    out_dir, sample_token :
        Directory and token; reads out_dir / sample_token.npz.
    grid : VoxelGrid, optional
        If given, the file's stored dims are checked against it, catching a
        file generated with a different grid config before it can be silently
        misread.

    Returns
    -------
    SampleGT
    """
    path = Path(out_dir) / f"{sample_token}.npz"
    with np.load(path) as data:
        labels = data["labels"]
        visibility = data["visibility"]
        stored_token = str(data["sample_token"])
        stored_dims = tuple(int(d) for d in data["grid_dims"])

    if stored_token != sample_token:
        raise ValueError(
            f"{path.name}: token mismatch ({stored_token} != {sample_token})"
        )
    if grid is not None and stored_dims != grid.shape:
        raise ValueError(
            f"{path.name}: grid dims {stored_dims} != expected {grid.shape}. "
            f"Was this file generated with a different grid config?"
        )
    if grid is None:
        # Reconstruct a default grid when the stored dims match it; the per-file
        # record only holds dims, not ranges, so consult meta.json for the full
        # definition in the general case.
        grid = VoxelGrid() if stored_dims == (200, 200, 16) else None

    return SampleGT(
        sample_token=sample_token,
        labels=labels,
        visibility=visibility,
        grid=grid,  # type: ignore[arg-type]
    )


def list_samples(out_dir: Path) -> list[str]:
    """Return the sample tokens listed in the manifest, in written order."""
    return [s["token"] for s in load_meta(out_dir).get("samples", [])]