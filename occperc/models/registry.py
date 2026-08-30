"""
A minimal component registry.

Maps string names to component classes (or factory callables) so the model's
blocks (encoder, neck, head) can be selected by name from config rather than
imported and wired directly. This is the seam that makes the pipeline
extensible: a new encoder is added by decorating it with
@ENCODERS.register("my_encoder"), and selected with a one-line config change,
with no edit to the assembly code.

IMPORTANT: decorators only run when the module containing them is imported.
The package's models/__init__.py imports placeholder_net so the registries are
populated before anything builds from them; a new component module must be
imported there too, or build will not find it.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Generic, TypeVar

import torch.nn as nn

T = TypeVar("T")


class EncoderBase(nn.Module):
    """Base class for encoders.

    Subclasses set out_channels and implement
    forward: (B, C_in, X, Y, Z) -> (B, out_channels, X', Y', Z').
    """
    out_channels: int


class NeckBase(nn.Module):
    """Base class for necks.

    Subclasses set out_channels and implement forward at the encoder's output
    resolution, preserving spatial dims.
    """
    out_channels: int


class HeadBase(nn.Module):
    """Base class for heads.

    Subclasses implement
    forward: (B, C_in, X', Y', Z') -> (B, num_classes, X, Y, Z),
    upsampling back to the input grid resolution.
    """


class Registry(Generic[T]):
    """A name to factory (class or function) registry with a decorator."""

    def __init__(self, name: str, *, allow_overwrite: bool = False) -> None:
        self.name = name
        self.allow_overwrite = allow_overwrite
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: register a class or factory function under key.

        Validates immediately that the entry is callable; full signature
        validation happens at build time with a clear error message.

        Re-registering an object with the same qualified name under the same
        key is treated as a no-op. This tolerates a module being imported twice
        under different module identities, for example when a module is both
        imported by its package and run with python -m, which creates distinct
        class objects that share a name. A genuinely different component
        claiming an already-registered key still raises.
        """
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            existing = self._entries.get(key)
            same_name = (
                existing is not None
                and getattr(existing, "__qualname__", None)
                == getattr(fn, "__qualname__", None)
            )
            if existing is not None and not same_name and not self.allow_overwrite:
                raise KeyError(
                    f"{key!r} already registered in {self.name} "
                    f"(-> {existing!r}); "
                    f"set allow_overwrite=True to replace it"
                )
            if not callable(fn):
                raise TypeError(
                    f"Registry entries must be callable, got {fn!r} for {key!r}"
                )
            self._entries[key] = fn
            return fn
        return deco

    def build(self, key: str, **kwargs: Any) -> T:
        """Instantiate the entry registered under key with kwargs.

        Signature mismatches surface as a clear error naming the key and the
        accepted parameters, instead of a bare TypeError from the constructor.
        """
        fn = self._get(key)
        sig = inspect.signature(fn)
        try:
            sig.bind(**kwargs)
        except TypeError as e:
            params = ", ".join(sig.parameters)
            raise TypeError(
                f"Invalid kwargs for {self.name}[{key!r}]: {e}. "
                f"Accepted parameters: ({params})"
            ) from e
        return fn(**kwargs)

    def _get(self, key: str) -> Callable[..., T]:
        if key not in self._entries:
            raise KeyError(
                f"{key!r} not in {self.name} registry; "
                f"available: {self.available()}"
            )
        return self._entries[key]

    def available(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, keys={self.available()})"


# The three registries the model assembles from. Each is typed to its component
# base class, documenting the invariant that only that kind of component belongs
# in it and that build returns that base type.
ENCODERS: Registry[EncoderBase] = Registry("encoders")
NECKS: Registry[NeckBase] = Registry("necks")
HEADS: Registry[HeadBase] = Registry("heads")