"""
Training loop for the occupancy pipeline.

Wires the pieces built separately (dataset, model, masked loss) into a standard
forward, loss, backward, step loop. The point is to demonstrate a training-ready
pipeline end to end, not to train a model to convergence: the placeholder
network and tiny dataset only need to show that data flows, the forward pass
produces correctly-shaped logits, the masked loss reduces to a scalar, and
gradients update the weights (loss should visibly decrease as the model begins
to memorise the handful of samples).

The loop is deliberately framework-free (plain PyTorch) and device-agnostic; it
returns the per-step loss history so callers can assert the pipeline actually
trains.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from occperc.data.occ_dataset import OccupancyDataset
from occperc.engine.config import TrainConfig
from occperc.losses.occupancy_loss import OccupancyLoss
from occperc.models.placeholder_net import PlaceholderOccNet


def build_components(cfg: TrainConfig):
    """Construct dataset, model, loss, and optimizer from a config.

    The model's class count is read from the dataset, so data and model stay in
    sync automatically: the output head always matches the label space.
    """
    dataset = OccupancyDataset(cfg.gt_root)

    model = PlaceholderOccNet(
        num_classes=dataset.num_classes,
        encoder=cfg.encoder,
        neck=cfg.neck,
        head=cfg.head,
        base=cfg.base_channels,
    ).to(cfg.device)

    loss_fn = OccupancyLoss(num_classes=dataset.num_classes).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    return dataset, model, loss_fn, optimizer


def train(cfg: TrainConfig) -> list[float]:
    """Run the training loop; return the list of per-step loss values."""
    torch.manual_seed(cfg.seed)

    dataset, model, loss_fn, optimizer = build_components(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    model.train()
    losses: list[float] = []
    step = 0

    for epoch in range(cfg.epochs):
        for batch in loader:
            inputs = batch["input"].to(cfg.device)   # (B, 1, X, Y, Z)
            target = batch["target"].to(cfg.device)  # (B, X, Y, Z)

            # Forward, masked loss, backward, optimizer step.
            logits = model(inputs)                   # (B, C, X, Y, Z)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % cfg.log_every == 0:
                print(f"epoch {epoch + 1:>2d}  step {step:>3d}  "
                      f"loss {loss.item():.4f}")

            if cfg.max_steps and step >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}, stopping.")
                return losses

    return losses