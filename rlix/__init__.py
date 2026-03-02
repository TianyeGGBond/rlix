"""Rlix: Ray-based multi-pipeline GPU time-sharing (ENG-123).

Phase 1 provides the core package skeleton + protocol contracts + Library Mode discovery.
"""

from __future__ import annotations

__all__ = [
    "init",
    "__version__",
    "RlixCoordinator",
    "RlixFullFinetunePipeline",
    "RlixMultiLoraPipeline",
]

__version__ = "0.0.0"

from rlix.init import init  # noqa: E402
from rlix.pipeline import RlixCoordinator, RlixFullFinetunePipeline, RlixMultiLoraPipeline  # noqa: E402
