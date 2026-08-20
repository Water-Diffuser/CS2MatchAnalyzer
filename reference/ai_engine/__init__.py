"""Pluggable AI engine for the gameplay mechanics analyzer.

Public surface is intentionally small — the rest of the app only ever touches
`get_provider`, `ClipRequest`, and `build_result`.
"""
from .budget import BudgetExceeded, BudgetGuard, CostEstimate, ResponseCache, TokenBucket
from .framegrid import FrameGrid, build_frame_grid
from .keystore import KeyFingerprint, KeyStoreUnavailable, delete_key, load_key, save_key, scrub
from .providers import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    AIProvider,
    ClipRequest,
    build_result,
    get_provider,
)

__all__ = [
    "AIProvider", "ClipRequest", "get_provider", "build_result",
    "BudgetGuard", "BudgetExceeded", "CostEstimate", "ResponseCache", "TokenBucket",
    "FrameGrid", "build_frame_grid",
    "KeyFingerprint", "KeyStoreUnavailable", "save_key", "load_key", "delete_key", "scrub",
    "PROMPT_VERSION", "SCHEMA_VERSION",
]
