"""Rlix: Ray-based multi-pipeline GPU time-sharing."""

from __future__ import annotations

__all__ = [
    "init",
    "__version__",
    "PipelineCoordinator",
    "RollFullFinetunePipeline",
    "RollMultiLoraPipeline",
]

__version__ = "0.0.0"

from rlix.init import init  # noqa: E402


# Lazy imports to avoid circular dependency: rlix.pipeline imports roll.pipeline
# which imports rlix.protocol, causing a circular chain if loaded eagerly here.
def __getattr__(name: str) -> object:
    if name in ("PipelineCoordinator", "RollFullFinetunePipeline", "RollMultiLoraPipeline"):
        from rlix.pipeline import PipelineCoordinator, RollFullFinetunePipeline, RollMultiLoraPipeline
        _lazy_exports = {
            "PipelineCoordinator": PipelineCoordinator,
            "RollFullFinetunePipeline": RollFullFinetunePipeline,
            "RollMultiLoraPipeline": RollMultiLoraPipeline,
        }
        return _lazy_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
