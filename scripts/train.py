"""
Run the DL pipeline: one command, end to end.

Loads serialized GT, builds the placeholder model and masked loss from config,
and runs the training loop. This is the DL pipeline inference and loss
computation half of the project made runnable, the command shown in the
walkthrough video. Config comes from defaults, an optional --config YAML, and
optional command-line overrides layered on top.

Canonical invocation (from the project root):
    python -m scripts.train --config configs/train.yaml
Running as a module keeps imports clean without sys.path manipulation. Device is
auto-detected by default, so no device flag is needed; pass --device cpu or
--device cuda to force one.
"""

from __future__ import annotations

import argparse

from occperc.engine.config import TrainConfig
from occperc.engine.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the occupancy pipeline.")
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument("--gt-root", help="Override: GT directory.")
    parser.add_argument("--epochs", type=int, help="Override: number of epochs.")
    parser.add_argument("--max-steps", type=int, help="Override: cap total steps.")
    parser.add_argument("--lr", type=float, help="Override: learning rate.")
    parser.add_argument("--device", help="Override: auto, cpu, or cuda.")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()

    # Command-line overrides take precedence over the file and the defaults.
    # These mutations happen after __post_init__, so we re-validate to keep the
    # guarantee that a fully-built config is always a valid config.
    if args.gt_root is not None:
        cfg.gt_root = args.gt_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    cfg.validate()

    print("Config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print()

    losses = train(cfg)

    if losses:
        print(f"\nFirst step loss: {losses[0]:.4f}")
        print(f"Last step loss:  {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print("Loss decreased, pipeline trains end-to-end. [OK]")
        else:
            print("Loss did not decrease over this short run "
                  "(expected occasionally with so few steps).")


if __name__ == "__main__":
    main()