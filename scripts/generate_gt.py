"""
Generate occupancy GT for a subset of nuScenes keyframes.

Walks a scene, runs the full GT pipeline per keyframe, and writes each result to
an output directory as a compressed .npz, plus a single meta.json. This is the
GT generation half of the project made runnable end-to-end.

Robustness for long runs: a manifest of written samples is recorded in
meta.json (so a directory can be validated and an interrupted run resumed),
already-generated keyframes are skipped on rerun, and a per-frame failure is
logged and skipped rather than aborting the batch (toggle with --keep-going;
failing fast remains the default so the short demo surfaces errors loudly).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from occperc.gt.generator import (
    generate_sample_gt,
    iter_scene_samples,
)
from occperc.gt.labeling import build_category_map
from occperc.gt.serialization import save_sample, write_meta
from occperc.gt.voxel_grid import VoxelGrid

META_NAME = "meta.json"


def _load_manifest(out_dir: Path) -> dict:
    """Return the existing meta.json contents, or an empty manifest."""
    meta_path = out_dir / META_NAME
    if not meta_path.exists():
        return {}
    with meta_path.open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate occupancy GT.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--out", default="gt_out", help="Output directory.")
    parser.add_argument("--scene", type=int, default=0, help="Scene index.")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Cap on keyframes to process (0 = whole scene).")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--keep-going", action="store_true",
                        help="Log per-frame failures and continue instead of "
                             "aborting the batch.")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    grid = VoxelGrid()
    category_map = build_category_map(nusc)

    if not 0 <= args.scene < len(nusc.scene):
        parser.error(f"scene index {args.scene} out of range "
                     f"(dataset has {len(nusc.scene)} scenes)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip keyframes already present in a prior manifest. The
    # manifest only covers successful frames, so a frame that failed last run is
    # retried this run.
    manifest = _load_manifest(out_dir)
    done = {s["token"] for s in manifest.get("samples", [])}
    if done:
        print(f"Resuming: {len(done)} keyframe(s) already present, will skip.")

    tokens = [token for _, token in iter_scene_samples(nusc, args.scene)]
    if args.max_samples > 0:
        tokens = tokens[: args.max_samples]
    tokens = [t for t in tokens if t not in done]

    if not tokens:
        print("Nothing to do (all requested keyframes already generated).")
        return

    print(f"Generating GT for {len(tokens)} keyframe(s) -> {out_dir}/")

    # Each record holds token, file, and seconds. Records accumulate across
    # resume runs so the final manifest describes the whole directory.
    written = list(manifest.get("samples", []))
    failures: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    for i, token in enumerate(tokens):
        t_frame = time.perf_counter()
        try:
            gt = generate_sample_gt(
                nusc, token, grid, category_map,
                n_sweeps=args.n_sweeps, subsample=args.subsample,
            )
            path = save_sample(out_dir, gt)
        except Exception as e:  # noqa: BLE001, record and continue if asked
            msg = f"{type(e).__name__}: {e}"
            if not args.keep_going:
                raise
            print(f"  [{i+1}/{len(tokens)}] {token[:12]}...  FAILED: {msg}")
            failures.append((token, msg))
            continue

        elapsed = time.perf_counter() - t_frame
        written.append({"token": token, "file": path.name,
                        "seconds": round(elapsed, 2)})
        occ = int(gt.occupancy.sum())
        print(f"  [{i+1}/{len(tokens)}] {token[:12]}...  "
              f"occupied={occ:>6d}  {elapsed:.1f}s  -> {path.name}")

    # Write meta last: the manifest is only complete once all files exist, so a
    # meta.json in the directory always describes exactly what is there.
    write_meta(out_dir, grid, category_map, samples=written)

    total = time.perf_counter() - t_start
    print(f"Done. {len(written)} file(s) + meta.json in {out_dir}/ "
          f"({total:.1f}s total, "
          f"{total / max(len(tokens), 1):.1f}s/frame this run).")
    if failures:
        print(f"WARNING: {len(failures)} keyframe(s) failed:")
        for token, msg in failures:
            print(f"  {token}: {msg}")


if __name__ == "__main__":
    main()