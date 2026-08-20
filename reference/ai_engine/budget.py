"""Spend ceiling, rate limiting, and response caching.

Every one of these exists because of a specific way hobby projects generate
surprise bills:

  * BudgetGuard   - a runaway loop analyzing a 3-hour VOD clip by clip.
  * TokenBucket   - a retry storm turning one 429 into four hundred requests.
  * ResponseCache - a user reopening the same dashboard tab twenty times.

The ceiling is enforced *before* the call, not reconciled after it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass

# Prices move and differ per tier, so they are configuration, not code. Populate
# this from the vendors' current pricing pages at build time and ship it as a
# data file the app can refresh; never hard-code stale numbers into a binary.
# USD per 1M tokens: {"model_id": {"input": x, "output": y}}
PRICE_TABLE_PATH = "config/model_prices.json"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    usd: float

    def human(self, n_clips: int, model: str) -> str:
        return f"~${self.usd:.3f} · {n_clips} clips · {model}"


class BudgetGuard:
    """Hard per-session and per-month ceilings, checked before each request.

    Defaults are deliberately low. A user who wants to spend more can raise
    them explicitly; a user who did not think about it is protected.
    """

    def __init__(self, session_limit_usd: float = 0.50, monthly_limit_usd: float = 10.00):
        self.session_limit = session_limit_usd
        self.monthly_limit = monthly_limit_usd
        self.session_spent = 0.0
        self.month_spent = 0.0
        self._lock = threading.Lock()

    def check(self, estimate: CostEstimate) -> None:
        with self._lock:
            if self.session_spent + estimate.usd > self.session_limit:
                raise BudgetExceeded(
                    f"session ceiling ${self.session_limit:.2f} would be exceeded "
                    f"(spent ${self.session_spent:.3f}, this call ~${estimate.usd:.3f})"
                )
            if self.month_spent + estimate.usd > self.monthly_limit:
                raise BudgetExceeded(f"monthly ceiling ${self.monthly_limit:.2f} reached")

    def record(self, actual_usd: float) -> None:
        with self._lock:
            self.session_spent += actual_usd
            self.month_spent += actual_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.session_limit - self.session_spent)


class TokenBucket:
    """Classic token bucket. Blocks rather than dropping, since these calls are
    user-initiated batch work where waiting is preferable to failing."""

    def __init__(self, rate_per_sec: float = 8 / 60, burst: int = 3):
        self.rate = rate_per_sec
        self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            if time.monotonic() + wait > deadline:
                raise TimeoutError("rate limiter timed out")
            time.sleep(min(wait, 1.0))


class ResponseCache:
    """Content-addressed cache. Re-analyzing identical input must cost $0.

    `prompt_version` and `schema_version` are part of the key so that improving
    a prompt correctly invalidates old entries, rather than silently mixing two
    rubrics into one dashboard.
    """

    def __init__(self, db_path: str = "analysis.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS ai_cache ("
            " key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.conn.commit()

    @staticmethod
    def key(clip_sha256: str, prompt_version: str, model: str, schema_version: str) -> str:
        return hashlib.sha256(
            "|".join((clip_sha256, prompt_version, model, schema_version)).encode()
        ).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT payload FROM ai_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ai_cache VALUES (?,?,?)",
            (key, json.dumps(payload), time.time()),
        )
        self.conn.commit()


def is_retryable(status_code: int) -> bool:
    """Retry only what can plausibly succeed on a second attempt.

    Retrying a 400 or a 401 just spends money on a request that cannot work,
    and a 402/403 loop is how a rate-limit incident becomes a billing incident.
    """
    return status_code in (408, 409, 425, 429, 500, 502, 503, 504)
