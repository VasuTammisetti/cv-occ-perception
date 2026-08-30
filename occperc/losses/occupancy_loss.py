"""
Masked per-voxel loss for occupancy prediction.

Cross-entropy between per-voxel class logits and the semantic target, computed
only over observed voxels. Unobserved voxels carry the ignore label -1 (set by
SampleGT.semantic_grid) and are excluded from the loss: the visibility grid
exists precisely so the model is never supervised on space the sensor could not
see, which would otherwise train it to hallucinate behind occluders.

The ignore index is -1, not PyTorch's default -100, so CrossEntropyLoss is
constructed with ignore_index=-1 explicitly. With the default, a -1 target is an
out-of-range class index and fails with a device-side assert on CUDA, a
confusing failure that is avoided by being explicit.

Shape contract:
    logits : (B, C, X, Y, Z)   per-voxel class scores
    target : (B, X, Y, Z)      class id per voxel (int64), or -1 = ignore
This is exactly the (B, C, spatial) and (B, spatial) layout that torch's
cross-entropy expects, so no reshaping is needed.

Degenerate batch handling: if a batch contains no observed voxels at all (for
example an empty scan), mean reduction has nothing to average over and torch
returns NaN, which would poison gradients. This module returns a zero loss that
is still connected to the graph instead.

Optional class weighting addresses the extreme imbalance, since background and
free-adjacent voxels vastly outnumber any object class, but it is off by default
to keep the placeholder simple. A real system would also consider Lovasz-softmax
or geometric and semantic scaled losses; those are noted as extensions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Must match SampleGT.semantic_grid's ignore marker for unobserved voxels.
IGNORE_INDEX = -1


class OccupancyLoss(nn.Module):
    """Masked cross-entropy over observed voxels.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes C. Used only to validate inputs.
    class_weights : Tensor, optional
        Per-class weights (length num_classes) to counter class imbalance. If
        provided, move this module to the training device with .to(device), or
        construct the weights on the device, since a CPU-side weight tensor
        against CUDA logits raises at the first forward pass.
    label_smoothing : float, optional
        Passed through to CrossEntropyLoss. Often helpful under extreme class
        imbalance; default 0.0 (off).
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None and class_weights.numel() != num_classes:
            raise ValueError(
                f"class_weights has {class_weights.numel()} entries, "
                f"expected num_classes={num_classes}"
            )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the masked loss.

        Parameters
        ----------
        logits : (B, C, X, Y, Z) float
        target : (B, X, Y, Z) int64, with -1 at ignored voxels

        Returns
        -------
        Scalar loss tensor. Zero (but differentiable) if no voxels are observed
        in the batch.
        """
        if logits.dim() != 5:
            raise ValueError(f"expected 5D logits (B,C,X,Y,Z), got {logits.shape}")
        if target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(
                f"target {tuple(target.shape)} does not match logits "
                f"{tuple(logits.shape)} at (B, X, Y, Z)"
            )
        if target.dtype != torch.int64:
            raise ValueError(
                f"target must be int64 (class ids or {IGNORE_INDEX}), "
                f"got {target.dtype}"
            )
        if target.min().item() < -1 or target.max().item() >= logits.shape[1]:
            raise ValueError(
                f"target values out of range [{IGNORE_INDEX}, {logits.shape[1] - 1}]"
            )

        # All-ignored batch: mean reduction over zero voxels would give NaN.
        # Return a zero that keeps the graph connected so backward is a no-op.
        if not (target != IGNORE_INDEX).any():
            return logits.sum() * 0.0

        return self.criterion(logits, target)


if __name__ == "__main__":
    import torch.nn.functional as F

    # Numeric check on synthetic data, no dataset needed.
    torch.manual_seed(0)
    B, C, X, Y, Z = 2, 24, 8, 8, 4  # small grid for a fast check
    logits = torch.randn(B, C, X, Y, Z, requires_grad=True)
    target = torch.randint(0, C, (B, X, Y, Z))

    # Mark some voxels unobserved (-1); they must not contribute to the loss.
    target[:, 0, 0, 0] = IGNORE_INDEX

    # Unweighted: the exclusion and independence proofs below assume
    # class_weights is None, since torch's weighted mean uses a different
    # normalization that would break the manual comparison.
    loss_fn = OccupancyLoss(num_classes=C)
    loss = loss_fn(logits, target)
    print(f"Loss: {loss.item():.4f}")
    assert loss.item() > 0, "loss should be positive on random logits"

    loss.backward()
    assert logits.grad is not None, "no gradient, loss is not differentiable"
    print("Gradient flows.")

    # Degenerate-batch path: an all-ignored batch must yield exactly 0.0 loss
    # with a connected graph (not NaN), so it cannot poison training.
    logits2 = torch.randn(B, C, X, Y, Z, requires_grad=True)
    all_ignored = torch.full((B, X, Y, Z), IGNORE_INDEX)
    zero_loss = loss_fn(logits2, all_ignored)
    print(f"All-ignored loss: {abs(zero_loss.item())} (0.0 expected, nothing to supervise)")
    assert zero_loss.item() == 0.0, "all-ignored batch should give zero loss, not NaN"
    zero_loss.backward()
    assert (logits2.grad == 0).all(), "all-ignored batch must produce zero gradients"
    print("Degenerate-batch path verified (0 loss, zero grads, graph intact).")

    # Exclusion proof: the module's loss must equal cross-entropy computed over
    # only the observed voxels. This proves -1 voxels are excluded, not clamped.
    obs = target != IGNORE_INDEX                       # (B, X, Y, Z) bool
    logits_flat = logits.permute(0, 2, 3, 4, 1)[obs]   # (N, C)
    target_flat = target[obs]                          # (N,)
    manual = F.cross_entropy(logits_flat, target_flat)  # mean over observed only
    module_loss = loss_fn(logits, target)
    assert torch.isclose(manual, module_loss, atol=1e-6), \
        f"module loss {module_loss.item()} != manual observed-only loss {manual.item()}"
    print("Exclusion verified (loss over observed voxels only, -1 truly ignored).")

    # Independence proof: perturbing logits only at ignored voxels must not
    # change the loss. This is the strongest masking check, since even if
    # indexing were subtly wrong somewhere, ignored-voxel values cannot move the
    # loss.
    perturbed = logits.detach().clone()
    perturbed[:, :, 0, 0, 0] += 100.0  # the voxel marked IGNORE_INDEX above
    loss_pert = loss_fn(perturbed, target)
    assert torch.isclose(loss, loss_pert, atol=1e-6), \
        "ignored-voxel logits influenced the loss"
    print("Independence verified (ignored-voxel logits have zero effect).")