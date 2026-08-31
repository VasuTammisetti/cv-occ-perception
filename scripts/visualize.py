"""
Visualize serialized occupancy GT as bird's-eye-view images.

Standalone helper, separate from the pipeline: it loads a saved .npz through the
serialization module (no nuScenes devkit needed) and renders two top-down views
side by side: the three-state visibility grid (occupied, free, unobserved) and
the semantic label grid (class per column). This is for the design report and
the walkthrough, not part of GT generation or training.

Both views collapse the grid along z with a priority order so the most
informative value wins per column. For visibility that is occupied over free
over unobserved. For labels it is the most common non-background class in the
column, so foreground objects remain visible from above.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from occperc.gt.serialization import list_samples, load_meta, load_sample

# Visibility state codes (match the visibility module).
UNOBSERVED = 0
FREE = 1
OCCUPIED = 2

# Label 0 is background or empty (match the labeling module).
LABEL_BACKGROUND = 0


def bev_from_visibility(visibility: np.ndarray) -> np.ndarray:
    """Collapse an (X, Y, Z) visibility grid to an (X, Y) top-down map.

    Priority per column: occupied wins over free, free wins over unobserved.
    So a column is marked occupied if any voxel in it is occupied, else free if
    any voxel is free, else unobserved.
    """
    occupied_col = np.any(visibility == OCCUPIED, axis=2)
    free_col = np.any(visibility == FREE, axis=2)

    bev = np.full(visibility.shape[:2], UNOBSERVED, dtype=np.uint8)
    bev[free_col] = FREE
    bev[occupied_col] = OCCUPIED  # applied last so occupied takes priority
    return bev


def bev_from_labels(labels: np.ndarray) -> np.ndarray:
    """Collapse an (X, Y, Z) label grid to an (X, Y) top-down class map.

    Each column takes the most common non-background class among its voxels, so
    foreground objects stay visible from above. Columns with only background
    stay background. This is for display only, not a substitute for the 3D
    labels.
    """
    x, y, z = labels.shape
    flat = labels.reshape(-1, z)  # (X*Y, Z)
    out = np.full(flat.shape[0], LABEL_BACKGROUND, dtype=labels.dtype)

    # For each column, find the most frequent class among foreground voxels.
    for i in range(flat.shape[0]):
        col = flat[i]
        fg = col[col != LABEL_BACKGROUND]
        if fg.size:
            vals, counts = np.unique(fg, return_counts=True)
            out[i] = vals[counts.argmax()]
    return out.reshape(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render occupancy GT as bird's-eye-view images."
    )
    parser.add_argument("--root", default="gt_out",
                        help="Directory of serialized GT.")
    parser.add_argument("--index", type=int, default=1,
                        help="Which sample in the manifest to render.")
    parser.add_argument("--out", default="occupancy_bev.png",
                        help="Output image path.")
    args = parser.parse_args()

    root = Path(args.root)
    tokens = list_samples(root)
    if not tokens:
        raise SystemExit(f"no samples in {root}; run scripts/generate_gt.py first")

    if not 0 <= args.index < len(tokens):
        raise SystemExit(
            f"--index {args.index} out of range: manifest has {len(tokens)} "
            f"samples (valid indices 0..{len(tokens) - 1})"
        )
    token = tokens[args.index]
    gt = load_sample(root, token)
    meta = load_meta(root)

    grid = meta.get("grid")
    if grid is None:
        raise SystemExit(
            "meta file has no 'grid' entry; cannot map voxels to metric extent"
        )
    x_min, x_max = grid["x_range"]
    y_min, y_max = grid["y_range"]
    extent = [x_min, x_max, y_min, y_max]

    # --- Visibility BEV ---
    vis_bev = bev_from_visibility(gt.visibility)
    counts = {
        "occupied": int((vis_bev == OCCUPIED).sum()),
        "free": int((vis_bev == FREE).sum()),
        "unobserved": int((vis_bev == UNOBSERVED).sum()),
    }
    total = vis_bev.size
    print(f"Sample: {token}")
    print("Visibility (columns):")
    for name, n in counts.items():
        print(f"  {name:<11s} {n:>6d} ({100 * n / total:.1f}%)")

    # --- Semantic BEV ---
    label_bev = bev_from_labels(gt.labels)
    id_to_name = {int(v): k for k, v in meta["category_map"].items()}
    present = sorted(int(v) for v in np.unique(label_bev) if v != LABEL_BACKGROUND)
    print(f"Foreground classes present (top-down): {len(present)}")

    fig, (ax_vis, ax_sem) = plt.subplots(1, 2, figsize=(13, 6))

    # Left panel: visibility. Grey unobserved, light blue free, red occupied.
    vis_cmap = ListedColormap(["#d9d9d9", "#9ecae1", "#e34a33"])
    ax_vis.imshow(vis_bev.T, origin="lower", cmap=vis_cmap, vmin=0, vmax=2,
                  extent=extent)
    ax_vis.plot(0, 0, marker="^", color="black", markersize=10)
    ax_vis.set_xlabel("x (m)")
    ax_vis.set_ylabel("y (m)")
    ax_vis.set_title("Visibility (occupied / free / unobserved)")
    ax_vis.legend(handles=[
        Patch(facecolor="#e34a33", label="occupied"),
        Patch(facecolor="#9ecae1", label="free"),
        Patch(facecolor="#d9d9d9", label="unobserved"),
    ], loc="upper right", framealpha=0.9)

    # Right panel: semantic labels. Background light grey, foreground classes
    # from a qualitative colour map. We remap present class ids to 0..K so the
    # colours are distinct regardless of the raw id values.
    remap = {cid: i + 1 for i, cid in enumerate(present)}
    sem_display = np.zeros_like(label_bev, dtype=np.int32)
    for cid, slot in remap.items():
        sem_display[label_bev == cid] = slot

    base = plt.get_cmap("tab20")
    sem_colors = ["#eeeeee"] + [base(i % 20) for i in range(len(present))]
    sem_cmap = ListedColormap(sem_colors)
    ax_sem.imshow(sem_display.T, origin="lower", cmap=sem_cmap,
                  vmin=0, vmax=len(present), extent=extent)
    ax_sem.plot(0, 0, marker="^", color="black", markersize=10)
    ax_sem.set_xlabel("x (m)")
    ax_sem.set_ylabel("y (m)")
    ax_sem.set_title("Semantic labels (foreground classes)")
    ax_sem.legend(handles=[
        Patch(facecolor=sem_colors[slot],
              label=id_to_name.get(cid, str(cid)).replace("vehicle.", "")
                    .replace("human.pedestrian.", "ped."))
        for cid, slot in remap.items()
    ], loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(f"Occupancy ground truth (bird's-eye view)  |  sample {token[:12]}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved image to {args.out}")


if __name__ == "__main__":
    main()