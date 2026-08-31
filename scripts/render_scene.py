"""
Render a raw nuScenes camera scene with projected 3D bounding boxes.

Standalone visualization helper, separate from the GT pipeline. Unlike
scripts/visualize.py (which reads serialized .npz with no devkit), this one uses
the nuScenes devkit to render the original camera image with the annotated 3D
boxes drawn on it. It is meant to sit next to the occupancy bird's-eye view in
the design report, showing the real scene the pipeline consumes.

This is a visualization aid only; nothing here is part of GT generation or
training, and the pipeline never imports it.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

# Front camera is the most legible single view for a report figure.
CAMERA_CHANNEL = "CAM_FRONT"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a nuScenes camera image with 3D boxes."
    )
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes root.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Which scene to render from.")
    parser.add_argument("--frame", type=int, default=5,
                        help="How many keyframes into the scene to render. "
                             "Matches the mid-scene frame used elsewhere.")
    parser.add_argument("--out", default="scene_rgb.png",
                        help="Output image path.")
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

    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][CAMERA_CHANNEL]

    # Render the camera image with annotated 3D boxes onto our own axis so we
    # control the output file and size. with_anns=True draws the 3D boxes.
    fig, ax = plt.subplots(figsize=(9, 5))
    nusc.render_sample_data(
        cam_token,
        with_anns=True,
        ax=ax,
        verbose=False,
    )
    ax.set_title(f"{CAMERA_CHANNEL} with 3D bounding boxes")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved image to {args.out}")
    print(f"Scene: {scene['name']}  sample: {sample_token}")


if __name__ == "__main__":
    main()