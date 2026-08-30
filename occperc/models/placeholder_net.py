"""
Placeholder occupancy network: encoder -> neck -> head.

The assignment grades pipeline design, not accuracy, so the blocks here are
deliberately small, just a handful of 3D convolutions, while the structure
mirrors how real occupancy networks are organised. Each block is pulled from a
registry by name, so any one can be replaced (a deeper 3D backbone, an FPN neck,
a different head) via config with no change to the assembly below.

Shape contract, end to end:
    input  : (B, 1, X, Y, Z)
    output : (B, C, X, Y, Z)   with C = num_classes, logits per voxel
Output spatial dims equal input spatial dims, so predictions align voxel for
voxel with the target grid. The encoder halves resolution; the head upsamples
back by the same factor. Input dims must be even for the reversal to be exact.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from occperc.models.registry import (
    ENCODERS,
    HEADS,
    NECKS,
    EncoderBase,
    HeadBase,
    NeckBase,
)


@ENCODERS.register("simple3d")
class SimpleEncoder3D(EncoderBase):
    """Two 3D conv stages; the second halves spatial resolution.

    Kept intentionally shallow. A real backbone (3D ResNet, sparse conv net)
    would slot in here under its own registry name.
    """

    def __init__(self, in_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, base, kernel_size=3, padding=1),
            nn.BatchNorm3d(base),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(x))


@NECKS.register("identity")
class IdentityNeck(NeckBase):
    """A one-conv bottleneck at the encoder's resolution.

    Placeholder for a real multi-scale fusion neck, for example an FPN.
    Preserves channel count and resolution so the head's expectations are
    simple.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@HEADS.register("occupancy")
class OccupancyHead(HeadBase):
    """Upsamples back to input resolution and predicts per-voxel logits.

    The transposed conv reverses the encoder's stride-2 downsample, so the
    output grid matches the input grid exactly.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.classifier = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.up(x))


class PlaceholderOccNet(nn.Module):
    """Assembles encoder -> neck -> head from registry names.

    Parameters
    ----------
    num_classes : int
        Output channels; must match the dataset's class count.
    encoder, neck, head : str
        Registry keys selecting each block. Defaults assemble the simple
        placeholder network; swap any one to experiment.
    base : int
        Encoder width.
    """

    def __init__(
        self,
        num_classes: int,
        encoder: str = "simple3d",
        neck: str = "identity",
        head: str = "occupancy",
        base: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = ENCODERS.build(encoder, in_channels=1, base=base)
        self.neck = NECKS.build(neck, channels=self.encoder.out_channels)
        self.head = HEADS.build(
            head, in_channels=self.neck.out_channels, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.encoder(x)))


if __name__ == "__main__":
    # Shape check: a random input must produce logits at the same spatial
    # resolution, with num_classes channels. No data needed.
    num_classes = 24
    model = PlaceholderOccNet(num_classes=num_classes)

    print("Registered blocks:")
    print("  encoders:", ENCODERS.available())
    print("  necks:   ", NECKS.available())
    print("  heads:   ", HEADS.available())

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    x = torch.randn(2, 1, 200, 200, 16)  # (B, 1, X, Y, Z)
    y = model(x)
    print(f"\nInput:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")

    assert y.shape == (2, num_classes, 200, 200, 16), \
        f"output shape {tuple(y.shape)} does not match input grid"
    print("Shape contract holds (output aligns with input grid).")