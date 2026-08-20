"""Pluggable AI provider adapters.

One interface, three vendors. Nothing above this layer knows or cares which is
configured; swapping providers is a dropdown, not a refactor.

All three can be pinned to our JSON schema, which is what makes a swappable
adapter realistic rather than aspirational:

    Google     response_mime_type="application/json" + response_schema
    OpenAI     response_format={"type": "json_schema", ..., "strict": True}
    Anthropic  a single tool with our input_schema + forced tool_choice

Because OpenAI's strict mode is the most restrictive of the three, the shared
schema is authored to its rules (every property required, additionalProperties
false, optionality as a nullable type union) and the other two accept it
unchanged. See schemas/clip_analysis.v1.schema.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .budget import BudgetExceeded, BudgetGuard, CostEstimate, ResponseCache, TokenBucket, is_retryable
from .framegrid import FrameGrid, build_frame_grid, estimate_image_tokens
from .keystore import load_key

PROMPT_VERSION = "clip_v1"
SCHEMA_VERSION = "clip_analysis/v1"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "clip_analysis.v1.schema.json"

# Model choice is configuration. Per-clip work goes to the cheap tier because
# cost here is dominated by image tokens; the single session-synthesis call
# reads only compact JSON and can afford the strong tier.
DEFAULT_MODELS = {
    "google":    {"clip": "gemini-2.0-flash",              "synthesis": "gemini-2.5-pro"},
    "openai":    {"clip": "gpt-4o-mini",                   "synthesis": "gpt-4o"},
    "anthropic": {"clip": "claude-haiku-4-5-20251001",     "synthesis": "claude-sonnet-5"},
}

SYSTEM_PROMPT = """\
You are an FPS aim coach analyzing a single 3-second engagement.

INPUT
  1. `measured`: frame-accurate metrics from computer vision. GROUND TRUTH.
     Never re-estimate, contradict, or recompute these values.
  2. A 3x3 grid of frames sampled from the clip, read left-to-right,
     top-to-bottom. The relative timestamp (seconds from clip start) is
     printed in each tile's top-left corner.

YOUR JOB
  Explain WHY the measured numbers came out as they did, using visual evidence,
  and give one specific corrective drill.

RULES
  - Cite the tile timestamp for every visual claim: "at 1.33s the crosshair
    sits at chest height on the doorway".
  - If the frames do not support a claim, omit it. Set `confidence` low and
    list what you could not determine in `not_determinable`. An honest
    "insufficient visual evidence" is more useful than a plausible guess.
  - Never comment on game sense, positioning, or economy. Mechanics only.
  - No praise-sandwiching. Lead with the single highest-leverage correction.
  - `drill` must name a real, specific, repeatable exercise with a target
    number, not "practice more".
  - Output ONLY JSON matching the provided schema.\
"""


def _assessed_schema() -> dict[str, Any]:
    """The `assessed` sub-schema is what the model fills in.

    The model is never shown the `measured` schema as an output target: if it
    can write those fields it will, and a hallucinated reaction time that
    overwrites an instrumented one is the worst failure this system can have.
    """
    full = json.loads(SCHEMA_PATH.read_text())
    schema = full["properties"]["assessed"]
    # `ai_status` is bookkeeping the app owns (ok / failed / skipped / cached).
    # Leaving it in the model's output target invites it to report "ok" on a
    # response we are about to reject.
    schema["properties"].pop("ai_status", None)
    schema["required"] = [f for f in schema["required"] if f != "ai_status"]
    return schema


@dataclass
class ClipRequest:
    """Everything needed to analyze one engagement."""

    video_path: str
    start_ms: int
    end_ms: int
    clip_sha256: str
    game: str
    event_type: str
    measured: dict[str, Any]
    reference_bands: dict[str, Any]
    clip_index: int = 1
    clip_total: int = 1
    sensitivity_note: str = "unknown"
    _grid: FrameGrid | None = field(default=None, repr=False)

    def grid(self) -> FrameGrid:
        if self._grid is None:
            self._grid = build_frame_grid(self.video_path, self.start_ms, self.end_ms)
        return self._grid

    def user_prompt(self) -> str:
        """Assemble the text half of the request.

        Two details carry most of the weight. Labeling `measured` as ground
        truth stops the model announcing "your reaction looks like about 400ms"
        from nine JPEGs and contradicting the instrumented 214ms. And the
        reference bands turn an unanswerable question ("is 0.31 jitter bad?")
        into a factual comparison it gets right every time.
        """
        return (
            f"CONTEXT\n"
            f"  game: {self.game} · clip {self.clip_index:02d} of {self.clip_total:02d}"
            f" · event: {self.event_type}\n"
            f"  player sensitivity: {self.sensitivity_note}\n\n"
            f"MEASURED (ground truth — do not recompute)\n"
            f"{json.dumps(self.measured, indent=2)}\n\n"
            f"REFERENCE BANDS (this skill tier)\n"
            f"{json.dumps(self.reference_bands, indent=2)}\n\n"
            f"The attached image is a 3x3 frame grid; tile timestamps (seconds from "
            f"clip start) are {self.grid().tile_timestamps_s}.\n\n"
            f"Analyze per the schema."
        )


class AIProvider(Protocol):
    name: str

    def analyze_clip(self, req: ClipRequest) -> dict[str, Any]: ...
    def estimate_cost(self, req: ClipRequest) -> CostEstimate: ...
    def validate_key(self) -> bool: ...


class _BaseProvider:
    """Shared plumbing: cost gate, rate limit, cache, retry, validation."""

    name = "base"

    def __init__(
        self,
        model: str | None = None,
        budget: BudgetGuard | None = None,
        cache: ResponseCache | None = None,
        limiter: TokenBucket | None = None,
    ):
        self.model = model or DEFAULT_MODELS[self.name]["clip"]
        self.budget = budget or BudgetGuard()
        self.cache = cache or ResponseCache()
        self.limiter = limiter or TokenBucket()
        self._api_key = load_key(self.name)
        if not self._api_key:
            raise RuntimeError(f"no API key configured for {self.name}")

    def estimate_cost(self, req: ClipRequest) -> CostEstimate:
        img = estimate_image_tokens(req.grid())
        text_in = len(req.user_prompt()) // 4 + len(SYSTEM_PROMPT) // 4
        out = 400  # bounded by the schema's maxLength constraints
        # Populate rates from config/model_prices.json at build time; the zeroed
        # default makes a missing price table obvious rather than silently free.
        rate_in, rate_out = self._rates()
        usd = (img + text_in) / 1e6 * rate_in + out / 1e6 * rate_out
        return CostEstimate(input_tokens=img + text_in, output_tokens=out, usd=usd)

    def _rates(self) -> tuple[float, float]:
        try:
            prices = json.loads(Path("config/model_prices.json").read_text())
            p = prices[self.model]
            return p["input"], p["output"]
        except (OSError, KeyError, json.JSONDecodeError):
            return 0.0, 0.0

    def analyze_clip(self, req: ClipRequest) -> dict[str, Any]:
        cache_key = ResponseCache.key(req.clip_sha256, PROMPT_VERSION, self.model, SCHEMA_VERSION)
        if (hit := self.cache.get(cache_key)) is not None:
            hit["ai_status"] = "cached"
            return hit

        estimate = self.estimate_cost(req)
        self.budget.check(estimate)          # raises BudgetExceeded before spending
        self.limiter.acquire()

        last_error: Exception | None = None
        for attempt in range(2):             # one corrective retry, no more
            try:
                raw, usage = self._call(req, repair_hint=str(last_error) if attempt else None)
                assessed = json.loads(raw)
                self._validate(assessed)
                assessed["ai_status"] = "ok"
                self.budget.record(estimate.usd)
                self.cache.put(cache_key, assessed)
                return assessed | {"_usage": usage}
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc            # feed the validator error back in
            except Exception as exc:        # transport
                status = getattr(exc, "status_code", 500)
                if not is_retryable(status) or attempt:
                    raise
                time.sleep(2 ** attempt + 0.3)
                last_error = exc

        # The dashboard must stay fully usable with zero successful AI calls;
        # that is what makes this tier genuinely optional rather than load-bearing.
        return {"ai_status": "failed", "primary_weakness": None, "severity": None,
                "confidence": None, "summary": None, "evidence": [], "drill": None,
                "not_determinable": [f"model returned unusable output: {last_error}"]}

    @staticmethod
    def _validate(assessed: dict[str, Any]) -> None:
        """Local re-validation. Provider-side schema enforcement is good, not perfect."""
        required = {"primary_weakness", "severity", "confidence", "summary",
                    "evidence", "drill", "not_determinable"}
        if missing := required - assessed.keys():
            raise ValueError(f"missing required fields: {sorted(missing)}")

    def _call(self, req: ClipRequest, repair_hint: str | None) -> tuple[str, dict]:
        raise NotImplementedError


class GoogleProvider(_BaseProvider):
    """Gemini via the google-genai SDK."""

    name = "google"

    def __init__(self, **kw):
        super().__init__(**kw)
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=self._api_key)

    def _call(self, req: ClipRequest, repair_hint: str | None) -> tuple[str, dict]:
        from google.genai import types

        prompt = req.user_prompt()
        if repair_hint:
            prompt += f"\n\nYour previous response was rejected: {repair_hint}\nReturn valid JSON."

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=req.grid().jpeg, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_assessed_schema(),
                temperature=0.2,   # this is analysis, not creative writing
                max_output_tokens=600,
            ),
        )
        usage = {
            "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", None),
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", None),
        }
        return resp.text, usage

    def validate_key(self) -> bool:
        try:
            next(iter(self.client.models.list()), None)
            return True
        except Exception:
            return False


class OpenAIProvider(_BaseProvider):
    """GPT via the openai SDK, using strict structured outputs."""

    name = "openai"

    def __init__(self, **kw):
        super().__init__(**kw)
        from openai import OpenAI
        self.client = OpenAI(api_key=self._api_key)

    def _call(self, req: ClipRequest, repair_hint: str | None) -> tuple[str, dict]:
        import base64

        b64 = base64.b64encode(req.grid().jpeg).decode()
        prompt = req.user_prompt()
        if repair_hint:
            prompt += f"\n\nYour previous response was rejected: {repair_hint}\nReturn valid JSON."

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "clip_assessment", "strict": True,
                                "schema": _assessed_schema()},
            },
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content, {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

    def validate_key(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


class AnthropicProvider(_BaseProvider):
    """Claude via forced tool use, which is how you pin its output to a schema."""

    name = "anthropic"

    def __init__(self, **kw):
        super().__init__(**kw)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self._api_key)

    def _call(self, req: ClipRequest, repair_hint: str | None) -> tuple[str, dict]:
        import base64

        b64 = base64.b64encode(req.grid().jpeg).decode()
        prompt = req.user_prompt()
        if repair_hint:
            prompt += f"\n\nYour previous response was rejected: {repair_hint}\nReturn valid JSON."

        tool = {
            "name": "submit_assessment",
            "description": "Submit the mechanical assessment for this engagement.",
            "input_schema": _assessed_schema(),
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_assessment"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return json.dumps(block.input), {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
        raise ValueError("model did not call the submit_assessment tool")

    def validate_key(self) -> bool:
        try:
            self.client.messages.create(
                model=self.model, max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False


PROVIDERS: dict[str, type[_BaseProvider]] = {
    "google": GoogleProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str, **kw) -> AIProvider:
    """The only construction path the rest of the app uses."""
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kw)


def build_result(req: ClipRequest, assessed: dict[str, Any], provider: str,
                 model: str, cv_version: str = "0.3.1") -> dict[str, Any]:
    """Assemble the full clip_analysis/v1 record.

    Note that `measured` is copied straight through from CV and `assessed` is
    dropped in beside it. They are never merged: that separation is what makes
    the AI output auditable against the numbers, lets you re-run analysis with a
    different model without recomputing CV, and guarantees a schema violation in
    the AI block can never corrupt your metrics history.
    """
    usage = assessed.pop("_usage", {}) or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": req.clip_sha256[:32],
        "session_id": req.clip_sha256[32:64] or req.clip_sha256[:32],
        "source": {
            "file": req.video_path, "game": req.game,
            "start_ms": req.start_ms, "end_ms": req.end_ms,
            "fps": 60, "resolution": "1920x1080",
            "content_sha256": req.clip_sha256,
        },
        "event": {
            "type": req.event_type, "weapon": None,
            "detection_confidence": req.measured.get("cv_confidence", 0.0),
            "detected_by": ["ammo_decrement", "kill_feed_ocr"],
        },
        "measured": req.measured,
        "assessed": assessed,
        "provenance": {
            "cv_version": cv_version,
            "prompt_version": PROMPT_VERSION,
            "provider": provider,
            "model": model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "estimated_cost_usd": None,
            "cached": assessed.get("ai_status") == "cached",
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


__all__ = ["AIProvider", "ClipRequest", "GoogleProvider", "OpenAIProvider",
           "AnthropicProvider", "get_provider", "build_result",
           "BudgetExceeded", "BudgetGuard", "PROMPT_VERSION", "SCHEMA_VERSION"]
