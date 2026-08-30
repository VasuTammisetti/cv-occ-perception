"""
Training configuration.

A plain dataclass holds every knob the pipeline exposes (data paths, model
shape, and optimization settings) so a run is fully described by one object
rather than scattered constants. Values can be overridden from a YAML file with
TrainConfig.from_yaml, giving config-file flexibility without a heavyweight
config framework or an extra runtime dependency beyond PyYAML.

Only keys present in the YAML override the defaults; anything omitted keeps its
default here. Unknown keys are rejected so a typo in a config file fails loudly
rather than being silently ignored.

Validation happens in __post_init__ (so also on programmatic construction, not
just from_yaml): registry component names are checked against the live
registries, device is resolved against actual hardware, and basic numeric sanity
is enforced. This means a bad name, an unavailable GPU, or a bad value fails at
config-build time rather than deep inside model assembly or step 1 of training.
validate imports the model package itself, so the registries are populated
regardless of import order, and the config validates standalone.

Device handling: the default auto resolves to cuda when a CUDA-enabled torch
build and a GPU are present, else cpu. Resolution happens in place, so after
validation device is always a concrete cpu or cuda value for .to(device). The
same config therefore runs unchanged on either kind of machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    # --- data ---
    gt_root: str = "gt_out"          # directory of serialized GT (.npz + meta)
    num_workers: int = 0             # dataloader workers (0 = main process)

    # --- model ---
    encoder: str = "simple3d"
    neck: str = "identity"
    head: str = "occupancy"
    base_channels: int = 16          # encoder width

    # --- optimization ---
    batch_size: int = 2
    lr: float = 1e-3
    epochs: int = 5
    max_steps: int = 0               # 0 = run full epochs; >0 caps total steps
    seed: int = 0

    # --- runtime ---
    device: str = "auto"             # auto resolves to cuda if available else cpu
    log_every: int = 1               # print loss every N steps

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail fast on bad names or values, at config-build time.

        Importing the model package here runs the registration decorators, so
        the registries are populated no matter what was imported before this
        config was built. Validation is therefore self-sufficient rather than
        depending on external import order. Already-imported modules are cached,
        so the import is effectively free on repeat calls.
        """
        import occperc.models  # noqa: F401  (populates the registries)
        from occperc.models.registry import ENCODERS, NECKS, HEADS

        for field_name, reg in (
            ("encoder", ENCODERS),
            ("neck", NECKS),
            ("head", HEADS),
        ):
            key = getattr(self, field_name)
            if key not in reg:
                raise ValueError(
                    f"unknown {field_name} {key!r}; "
                    f"available: {reg.available()}"
                )

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.max_steps < 0:
            raise ValueError(
                f"max_steps must be >= 0 (0 = full epochs), got {self.max_steps}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")

        # Resolve device. auto picks CUDA when available, else CPU, so the same
        # config runs unchanged on either. An explicit cuda is checked against
        # availability so a misconfigured run fails with a clear message here
        # rather than a torch stack trace mid-training.
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' requested but CUDA is not available "
                "(this torch build may be CPU-only, or no GPU is present). "
                "Use device='auto' to fall back to CPU automatically."
            )
        elif self.device not in ("cpu", "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu', or 'cuda', got {self.device!r}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Build a config from a YAML file, overriding only the keys present.

        Unknown keys raise, so a mistyped field is caught immediately instead of
        silently having no effect.
        """
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"unknown config keys in {path}: {sorted(unknown)}; "
                f"valid keys: {sorted(valid)}"
            )
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)