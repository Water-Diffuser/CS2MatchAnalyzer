# Gameplay Mechanics Analyzer — Architectural Blueprint (MVP)

> Analyze local gameplay / aim-trainer recordings and return actionable mechanical feedback.
> Local-first CV for measurement, user-supplied LLM/VLM APIs for interpretation.

**Status:** design blueprint · **Schema version:** `clip_analysis/v1` · **Target:** 6–8 week MVP

---

## 0. The one design decision everything else follows from

**Traditional CV measures. The AI model explains. Never swap those roles.**

A VLM cannot reliably tell you that a reaction time was 214 ms — it sees a
handful of sampled frames, it has no frame-accurate clock, and asking it to
count produces confident garbage. OpenCV can measure that to ±1 frame, for
free, on the user's own CPU.

Conversely, OpenCV cannot tell you *"you pre-aimed the wrong side of the box
because you never cleared the angle behind you, so every duel starts with a
180° flick you didn't need."* That is the feedback users actually pay for.

So the pipeline is: **deterministic numbers first, AI narrative second, with the
numbers passed into the prompt as ground truth.** This is also what keeps the
API bill small — the model gets ~8 short clips per match, not 40 minutes of
video, and it never has to "look harder" to derive something we already know.

Every downstream choice (schema shape, cost model, provider abstraction, even
the UI) falls out of that split.

---

## 1. Architecture Overview

### 1.1 Stack recommendation

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Shell | **Tauri 2** (Rust) | ~8 MB bundle vs Electron's ~150 MB; uses the OS webview. Critically, Tauri gives a *privileged Rust side* the webview cannot read — that is where API keys live. Electron's main process is still JS and a single `nodeIntegration` slip exposes everything. |
| UI | **React + TypeScript + Vite** | Boring on purpose. `uPlot` for dense motion traces (100k points, 40 kB), `Recharts` for summary bars. |
| CV worker | **Python 3.11 sidecar** (FastAPI on loopback), bundled via PyInstaller | OpenCV + NumPy + SciPy + PyAV has no real competitor. Ships as a Tauri sidecar binary — the user never installs Python. |
| Store | **SQLite** (WAL) + content-addressed blob dir | A match is ~50 rows and a few MB of JSON. Postgres/Docker for an MVP desktop app is self-harm. |
| Decode | **PyAV** (libav bindings), `ffprobe` for metadata | Direct frame access with real PTS timestamps. `cv2.VideoCapture` lies about timestamps on VFR files — and OBS output is very often VFR. |
| AI | **User-supplied keys**, provider-adapter pattern | See §2. |

### 1.2 Why local-first, and why that *is* the cost strategy

The naive design uploads video to a server that decodes it. That server needs
CPU (or GPU) proportional to your user count, plus egress, plus storage — the
exact bill the MVP is trying to avoid. Local-first inverts it: **the user's
machine already has the file and an idle CPU, and decoding it there costs you
$0.00.** The only money that moves is the user's own API spend, against their
own key, which never touches your infrastructure.

Concretely, the marginal cost of one more user is zero. You can ship this on a
$0 budget and a static download link.

### 1.3 Pipeline

```
┌── LOCAL (Tauri app, user's machine) ──────────────────────────────────────┐
│                                                                            │
│  [1] INGEST        drag-drop mp4/webm → ffprobe → normalize → proxy        │
│        │           · reject/repair VFR · detect game + resolution          │
│        ▼                                                                   │
│  [2] SAMPLE        adaptive decode: 2 fps coarse scan over whole file      │
│        │           full-fps decode only inside candidate windows           │
│        ▼                                                                   │
│  [3] DETECT        HUD-ROI CV — the cheap tier  (§3)                       │
│        │           kill feed · ammo counter · hitmarker · HP · score       │
│        │           → raw event candidates with frame-accurate timestamps   │
│        ▼                                                                   │
│  [4] DERIVE        motion telemetry — the signal-processing tier (§3.4)    │
│        │           optical-flow camera trace → velocity/jerk/SPARC         │
│        │           → reaction time, overshoot, jitter, placement error     │
│        ▼                                                                   │
│  [5] SELECT        rank moments by coachability score, keep top N (§4.2)   │
│        │           ← THE COST VALVE. Everything above is free; below isn't │
│        ▼                                                                   │
│  [6] PACKAGE       trim clip · build annotated frame grid · attach metrics │
│        │                                                                   │
└────────┼───────────────────────────────────────────────────────────────────┘
         │  HTTPS, user's own key, direct to vendor — never via your servers
         ▼
┌── EXTERNAL AI (Gemini / OpenAI / Anthropic — user's choice) ──────────────┐
│  [7] per-clip analysis  (cheap model, N calls, parallel, cached)          │
│  [8] session synthesis  (strong model, 1 call, map-reduce over [7])       │
└────────┼──────────────────────────────────────────────────────────────────┘
         ▼
┌── LOCAL ─────────────────────────────────────────────────────────────────┐
│  [9] PERSIST  SQLite ← validated JSON (schema-checked before it's stored) │
│ [10] RENDER   dashboard · timeline scrubber · overlay of crosshair trace  │
└──────────────────────────────────────────────────────────────────────────┘
```

Stages 1–6 and 9–10 are free and offline. Only 7–8 cost money, and §4 caps them.

### 1.4 Process topology & the loopback-port problem

Tauri spawns the Python sidecar as a child process on an **ephemeral port bound
to `127.0.0.1`**. Two details that are easy to get wrong:

1. **Any local process can reach a loopback port**, including a browser tab via
   DNS rebinding. So the sidecar requires a `Bearer` token: Rust generates 32
   random bytes at launch, passes them to Python via env var (not argv — argv is
   world-readable in `/proc` on Linux), and the webview receives it through an
   IPC command. Also validate the `Origin`/`Host` header to kill rebinding.
2. **The sidecar never sees an API key.** Key material stays in Rust. When the
   Python worker needs to make a model call it asks Rust to make it, or — simpler
   for the MVP — Rust performs the HTTP call itself and hands Python the response.
   Model calls are I/O, not CV, so this costs nothing architecturally.

```
Tauri (Rust)  ──spawn──▶  Python sidecar :PORT   (CV only, no secrets)
   │  owns keychain                │
   │  makes all vendor HTTPS calls │  loopback + bearer token
   ▼                               ▼
 Webview (React) ── IPC ──────────┘   never holds a plaintext key
```

### 1.5 What "lightweight" costs you, honestly

- First-run decode of a 40-minute 1080p60 capture is **~3–6 minutes** on a
  mid-range CPU for the coarse pass. Show a real progress bar and let stage 3
  stream results as they're found — do not block the UI on a full pass.
- PyInstaller-bundled OpenCV adds ~90 MB to the installer. Total ~120 MB, still
  under Electron-with-nothing.
- No GPU is required. If one is present, PyAV can use hardware decode
  (`h264_nvdec` / `videotoolbox`) for a 3–5× ingest speedup — detect and use it,
  never require it.

---

## 2. Secure API Configuration

### 2.1 Where the key lives

**`.env` is for your development machine and nowhere else.** In a shipped
desktop app a dotenv file is plaintext on disk, gets swept into crash dumps,
syncs to backup services, and shows up in screen-shares. Ship the OS keystore:

| OS | Backend | Access via |
|---|---|---|
| macOS | Keychain Services | `security-framework` (Rust) / `keyring` (Py) |
| Windows | Credential Manager (DPAPI-backed) | `windows-sys` / `keyring` |
| Linux | Secret Service (GNOME Keyring, KWallet) | `secret-service` / `keyring` |

One crate covers all three: the Rust **`keyring`** crate, service name
`com.yourorg.gameplayanalyzer`, account `provider:{gemini|openai|anthropic}`.

Linux caveat worth planning for: on a headless box or a minimal WM there may be
no Secret Service daemon at all. Detect that, tell the user plainly, and fall
back to an explicitly-consented encrypted file (age/XChaCha20 with an
app-password-derived key) — **not** to silent plaintext.

### 2.2 Key handling rules

1. **The key never crosses into the webview.** The frontend sends the key *in*
   once (an IPC `save_api_key` command) and from then on only ever receives a
   fingerprint: `{provider, last4, sha256_prefix, added_at, last_used_at}`.
   Render `sk-…••••…8fQ2`. If your renderer can read the key, one malicious npm
   postinstall in your dependency tree exfiltrates it.
2. **Validate on entry, cheaply.** Regex the shape (`sk-ant-`, `sk-`, `AIza`)
   to catch paste errors, then make the smallest possible live call (list-models
   or a 1-token completion) to confirm it works. Store only after it succeeds —
   a key that silently fails 20 minutes into a match analysis is a terrible
   first-run experience.
3. **Scrub logs at the sink, not the call site.** A single regex filter on the
   log writer — `(sk-ant-|sk-|AIza)[A-Za-z0-9_\-]{16,}` → `[REDACTED]` — because
   somebody will eventually `println!("{:?}", request)` and you want that caught
   structurally rather than by code review.
4. **Never bundle a fallback key of your own.** Any key shipped in a binary is
   a public key. If a user hasn't configured one, the AI tier is simply disabled
   and the CV metrics still work — which is why the CV tier must stand alone.
5. **Delete means delete**, plus a link to the vendor's revocation page, since
   deleting locally does not revoke anything.

### 2.3 Cost control — treat the budget as a first-class object

This is where hobby projects generate horror stories. Five layers, all client-side:

**L1 — Selection (the big one).** Stage 5 caps how many clips are ever eligible.
Default: **8 moments per match, 3 s each**. This is a 50–100× reduction versus
"analyze the whole VOD" and it is the reason the whole thing is affordable.

**L2 — Frame grids instead of video.** Do not upload video by default. Sample
9 frames from the 3 s window, tile them into one 3×3 JPEG at 512 px per tile,
and **burn the relative timestamp into each tile's corner**. One image ≈ one
image's worth of tokens instead of nine, and the model can still reason about
ordering because the timestamps are legible. Native video upload (Gemini Files
API) is an opt-in "deep analysis" toggle — note that Gemini samples video at
~1 fps by default, which is far too coarse for flick analysis, so the frame grid
is often *more* informative as well as cheaper.

**L3 — Model tiering / map-reduce.** Per-clip calls go to the cheap tier
(Gemini Flash, GPT-4o-mini, Haiku). The single session-synthesis call, which
reads N compact JSON summaries and no images at all, goes to the strong tier.
Cost is dominated by images, so this is close to free.

**L4 — Pre-flight estimate + hard ceiling.** Before any batch: compute
`estimated_tokens` from image count and tile size, price it from a local table,
and show *"~$0.04 · 8 clips · Gemini 2.0 Flash — Analyze?"*. Enforce a hard
per-session ceiling in code (default $0.50) and a monthly one. A token-bucket
limiter (e.g. 8 requests / 60 s, burst 3) prevents a retry loop from becoming a
$200 incident. Exponential backoff with jitter on 429; **never retry a 4xx that
isn't 429/408** — retrying a 400 just burns money on a request that cannot work.

**L5 — Content-addressed caching.** Cache key =
`sha256(clip_bytes) + prompt_version + model_id + schema_version`. Re-analyzing
the same match, or reopening a dashboard, must cost exactly $0. Tag the cache
with `prompt_version` so improving a prompt correctly invalidates it.

### 2.4 Privacy — the part that gets skipped

Gameplay footage carries the user's Steam/Riot name, their friends' names, their
Discord overlay, sometimes their desktop. Sending it to a third party is a real
disclosure, so:

- Show, per provider, exactly what leaves the machine and where it goes, once,
  before the first call — with a link to that vendor's data-retention policy.
  Free-tier Gemini in particular may use submitted data for product improvement;
  paid tiers generally do not. Say so.
- **Strip audio by default** (voice comms are other people's data, and you never
  need audio for mechanics).
- Offer a one-click **HUD-name blur** — you already know the kill-feed and
  scoreboard ROIs from §3, so blurring them is a `cv2.GaussianBlur` on a known
  rect, not an ML problem.
- Default to local-only mode. AI enrichment is opt-in, per session.

---

## 3. Computer Vision Strategy

The job of this layer: turn pixels into a timestamped, frame-accurate event log
plus a continuous motion trace — **without any game API, memory reading, or
network capture** (all of which are anti-cheat suicide in Valorant/CS2).

### 3.1 Bootstrapping: know the game before you look at it

Resolution + aspect ratio + a small set of HUD template matches on 3 sampled
frames identifies the title. Ship a **`profiles/{game}.yaml`** per supported
game — this is what makes the system extensible without touching code:

```yaml
game: valorant
detect: { templates: [ui/spike_icon.png, ui/ult_orb.png], min_score: 0.82 }
rois:                      # normalized 0..1, resolution independent
  kill_feed:   [0.62, 0.06, 0.38, 0.22]
  ammo:        [0.86, 0.88, 0.11, 0.08]
  health:      [0.04, 0.88, 0.14, 0.07]
  crosshair:   [0.46, 0.44, 0.08, 0.12]
  minimap:     [0.02, 0.02, 0.20, 0.28]   # excluded from optical flow
fov_degrees: 103           # horizontal, for px→degrees conversion
shot_signal: ammo_decrement
```

Aim trainers are *much* easier and should be MVP milestone one: fixed camera,
high-contrast targets, a clean score counter. Ship KovaaK's/Aimlabs first, get
the metrics pipeline correct against near-ground-truth, then take on Valorant
where the CV is genuinely hard.

### 3.2 Event detection — kill feed

The kill feed is the single richest signal. Approach, in order of robustness:

1. **Crop the ROI, upscale 3–4× (Lanczos), then threshold on team colors.**
   Valorant/CS2 kill-feed text is drawn in a small palette; an HSV color-key
   isolates it far better than generic binarization, and it also tells you
   *which side* each name is on (= who killed whom).
2. **Template-match weapon icons — don't OCR them.** There are ~20 weapons and
   the sprites are pixel-identical every time. `cv2.matchTemplate` with
   `TM_CCOEFF_NORMED` at ≥0.85 is faster and dramatically more accurate than
   asking Tesseract to read a picture of a rifle.
3. **OCR only the names**, with Tesseract `--psm 7` and a charset whitelist.
   You are not trying to read arbitrary text: you have the roster from the
   scoreboard, so **match OCR output to the nearest roster name by Levenshtein
   distance** and reject above a threshold. This converts a hard open-vocabulary
   OCR problem into an easy closed-set classification problem.
4. **Temporal dedup is mandatory.** A kill-feed row persists ~5 s, so a naive
   per-frame reader reports one kill 300 times. Compute a perceptual hash
   (`pHash`) of each row rect; emit an event only when a *new* hash appears in
   the top slot, and record the timestamp of its **first** appearance — which is
   also, conveniently, the frame-accurate kill time.

Ownership: the user enters their in-game name once during setup. Left slot =
their kill, right slot = their death.

### 3.3 Event detection — shots, hits, and whiffs

| Signal | Method | Notes |
|---|---|---|
| **Shot fired** | Ammo-counter OCR decrement in the ammo ROI | The most reliable shot proxy in any FPS. Digits only, fixed font, tiny ROI → near-100% with a 7-segment-style template classifier. Handles spray (27→26→25…) for free. |
| Shot fired (fallback) | Muzzle-flash luminance spike in the center-bottom ROI | Frame-to-frame ΔV in HSV; cheap but false-positives on flashbangs/utility. Use only when ammo OCR is unavailable. |
| **Hit registered** | Template match on the hitmarker sprite | Fixed sprite at fixed position; ~0.9 correlation threshold. |
| **Headshot** | Hitmarker variant / skull icon in kill feed | Both are template matches. |
| **Whiff** | `shot_fired ∧ ¬hit_registered` within 2 frames | Derived, not detected. This composition is the whole trick — you never need a "whiff detector". |
| Death | Own name in the right kill-feed slot, or HP→0 | HP ROI OCR is a useful cross-check. |
| Target spawn (trainers) | HSV threshold + `SimpleBlobDetector` | Trainer targets are deliberately high-contrast. Near-perfect. |

### 3.4 The motion trace — how you measure aim without game telemetry

This is the part with real engineering content, and it is what separates this
product from a highlight clipper.

**Key insight:** in an FPS the crosshair is nailed to screen center, so *aim
movement is camera movement*, and camera movement is recoverable from the pixels.

```
For each consecutive frame pair in a candidate window:
  1. Mask out: HUD ROIs, minimap, and the central 15% (moving enemies/effects
     would otherwise dominate and drag the estimate toward the target).
  2. Shi-Tomasi corners on the masked border region (~200 points).
  3. Lucas–Kanade sparse optical flow → per-point displacement.
  4. RANSAC-fit a global translation (or affine, if you want roll/zoom).
     RANSAC is essential — it rejects the moving players that survived masking.
  5. Global (dx, dy) px → (Δyaw, Δpitch) degrees using profile FOV.
```

That yields a per-frame angular velocity trace, which is the substrate for
every mechanical metric:

| Metric | Computation | Why this formulation |
|---|---|---|
| **Reaction time** | `t(first shot) − t(stimulus onset)` | Report with an explicit error bar: at 60 fps you cannot resolve better than **±16.7 ms**. Say so in the UI. Recommend 60 fps+ captures; refuse to report reaction time on 30 fps footage rather than reporting a number that is half noise. |
| **Smoothness** | **SPARC** (spectral arc length) of the velocity profile | Take this from motor-control research rather than inventing a metric. SPARC is amplitude- and duration-normalized, so a fast flick and a slow track are directly comparable — normalized-jerk is not, and will just tell you that fast movements are "worse". |
| **Jitter** | Fraction of velocity PSD energy above ~8 Hz | Human voluntary aim is <5 Hz. Energy above 8 Hz is tremor, mouse sensor noise, or a death-grip. This cleanly separates "bad aim" from "too much sensitivity/grip tension". |
| **Overshoot** | Sign changes in velocity during the 150 ms before the shot, + peak angular error past target center | Counts corrective micro-adjustments — the classic high-sens failure mode. |
| **Target-switch speed** | `t(crosshair settles within θ of target B) − t(kill A)`, settle = |ω| < 15°/s for 3 frames | Define "settled" explicitly or the number is meaningless. |
| **Crosshair placement** | Pitch deviation from the running-median horizon + Euclidean distance from center to the detected head box at engagement start | MVP-honest: without depth you cannot compute true head-level error. Report the 2D proxy and *label it as a proxy*. |
| **Counter-strafe accuracy** | Correlation of shot timing with the world-motion minimum from the same flow field | Falls out of machinery you already built. |

**Calibration matters more than sophistication.** Ship a 30-second setup where
the user does a known 360° in-game; that single measurement pins the px→degree
conversion and the effective sensitivity, and turns every angular metric from
"relative" into "absolute and comparable across users."

### 3.5 Stage 5 — coachability ranking (the cost valve)

Not every event deserves an API call. Score each candidate and keep the top N:

```
score = 2.0 · |z(reaction_time)|          # unusually fast or slow
      + 1.5 · overshoot_count
      + 1.5 · z(jitter_ratio)
      + 1.0 · whiff_ratio_in_window
      + 1.0 · is_death
      + 0.5 · is_multikill
      − 2.0 · similarity_to_already_selected   # diversity, not 8 copies of one mistake
```

That last term is what stops the AI budget going on eight near-identical clips.
Guarantee coverage: force at least one *good* rep into the set — users need to
see what their own correct mechanics look like, not only a list of failures.

---

## 4. AI Engine — the pluggable tier

### 4.1 Adapter contract

One interface, three implementations. Nothing above this layer knows which
vendor is in use:

```python
class AIProvider(Protocol):
    name: str
    def analyze_clip(self, req: ClipRequest) -> ClipAnalysis: ...   # → validated JSON
    def synthesize(self, clips: list[ClipAnalysis]) -> SessionReport: ...
    def estimate_cost(self, req: ClipRequest) -> CostEstimate: ...
    def validate_key(self) -> bool: ...
```

Structured output per vendor — all three can be pinned to your schema, which is
the whole reason a swappable adapter is realistic:

| Provider | Mechanism |
|---|---|
| Google Gemini | `response_mime_type="application/json"` + `response_schema` |
| OpenAI | `response_format={"type":"json_schema", "json_schema":{..., "strict":True}}` |
| Anthropic | Tool-use with an `input_schema` + `tool_choice` forcing that tool |

Because of OpenAI strict mode, the schema must be written to the **intersection**
of the three dialects: every property listed in `required`,
`additionalProperties: false` everywhere, no top-level `oneOf`, optionality
expressed as a nullable type union. §5 obeys this.

### 4.2 Prompt engineering strategy

Send **four things** with every clip call:

1. **System prompt** — role, rubric, and hard constraints.
2. **The measured telemetry as JSON** — explicitly labeled ground truth.
3. **The annotated 3×3 frame grid** — with per-tile relative timestamps.
4. **The output schema** — enforced by the API, and restated in the prompt.

The single most important instruction is the one that stops the model
hallucinating measurements:

> *The `measured` block is ground truth from computer vision. Do not re-derive,
> re-estimate, or contradict any value in it. Your job is to explain **why** the
> measured values came out the way they did, using the visual evidence.*

Without that line, models will cheerfully announce "your reaction time looks
like about 400 ms" from nine JPEGs, contradicting your instrumented 214 ms, and
the user will believe the confident sentence over the correct number.

**System prompt (v1):**

```
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
  - If the frames do not support a claim, omit it. Set `confidence` low and say
    what you could not determine. An honest "insufficient visual evidence" is
    more useful than a plausible guess.
  - Never comment on game sense, positioning, or economy. Mechanics only.
  - No praise-sandwiching. Lead with the single highest-leverage correction.
  - `drill` must name a real, specific, repeatable exercise with a target
    number — not "practice more".
  - Output ONLY JSON matching the provided schema.
```

**User message assembly:**

```
CONTEXT
  game: valorant · clip 04 of 08 · event: whiff_then_death
  player sensitivity: 0.42 @ 800 DPI (eDPI 336) · FOV 103°

MEASURED (ground truth — do not recompute)
{ "reaction_time_ms": 214, "reaction_time_error_ms": 16.7,
  "smoothness_sparc": -2.41, "jitter_ratio": 0.31,
  "overshoot_count": 3, "peak_angular_error_deg": 8.2,
  "shots_fired": 5, "shots_hit": 1, "crosshair_placement_error_deg": 6.4 }

REFERENCE BANDS (this rank/skill tier)
{ "reaction_time_ms": [180, 250], "jitter_ratio": [0.05, 0.15],
  "overshoot_count": [0, 1], "crosshair_placement_error_deg": [0, 3] }

[IMAGE: 3x3 frame grid, timestamps burned in]

Analyze per the schema.
```

Four techniques doing the real work here:

- **Reference bands turn the model into a comparator, not an oracle.** "Is 0.31
  jitter bad?" is unanswerable from priors and invites invention. "0.31 vs an
  expected 0.05–0.15" is a factual comparison it will get right every time.
- **Timestamps burned into tiles** give the model a citable coordinate system,
  which makes its claims checkable by the user — and makes hallucinated claims
  obvious rather than plausible.
- **Forbidding the adjacent domain** (game sense, positioning) keeps it from
  drifting into the generic esports-commentary voice that VLMs default to and
  that users find worthless.
- **One correction, not five.** Models over-enumerate; a coach who lists eight
  problems has given no advice at all.

**Synthesis prompt (session level)** takes the N clip JSONs and *no images*,
and is asked for cross-clip pattern extraction only: "which weakness recurs in
≥3 clips, ranked by frequency × severity". Cheap, and it is where the product's
perceived intelligence actually comes from — a per-clip note is a fact, a
recurring pattern is an insight.

**Version your prompts.** `prompt_version: "clip_v1"` is stored on every result
and participates in the cache key, so a prompt improvement invalidates stale
analyses instead of silently mixing two rubrics in one dashboard.

### 4.3 Handling bad output

Even with schema enforcement: validate against the JSON Schema locally, and on
failure retry **once** with the validator's error appended. If it fails twice,
store the clip with `ai_status: "failed"` and render the CV metrics alone. The
dashboard must be fully usable with zero successful AI calls — that is what
makes the AI tier genuinely optional rather than load-bearing.

---

## 5. Data Schema

Canonical machine-readable version: [`schemas/clip_analysis.v1.schema.json`](../schemas/clip_analysis.v1.schema.json).

The structural decision: **`measured` and `assessed` are separate sibling
objects.** `measured` is written only by CV, `assessed` only by the model.
Nothing merges them. This gives you three things: the AI response is auditable
against the numbers; you can re-run analysis with a different model without
recomputing CV; and a schema violation in the AI block can never corrupt your
metrics history.

```json
{
  "schema_version": "clip_analysis/v1",
  "clip_id": "3f2a9c7e-51b8-4d0a-9c33-71b5d2e8a410",
  "session_id": "a1c4e8d2-77f0-4b19-8e6a-2d9f1b3c5a88",
  "source": {
    "file": "valorant_2026-08-19_ascent.mp4",
    "game": "valorant",
    "start_ms": 412300, "end_ms": 415300,
    "fps": 60, "resolution": "1920x1080",
    "content_sha256": "9b1e...c4a7"
  },
  "event": {
    "type": "whiff_then_death",
    "weapon": "vandal",
    "detection_confidence": 0.91,
    "detected_by": ["kill_feed_ocr", "ammo_decrement", "hitmarker_template"]
  },

  "measured": {
    "reaction_time_ms": 214,
    "reaction_time_error_ms": 16.7,
    "time_to_first_shot_ms": 231,
    "shots_fired": 5,
    "shots_hit": 1,
    "headshot_pct": 0.0,
    "smoothness_sparc": -2.41,
    "jitter_ratio": 0.31,
    "jitter_band_hz": [8, 30],
    "overshoot_count": 3,
    "peak_angular_error_deg": 8.2,
    "crosshair_placement_error_deg": 6.4,
    "crosshair_placement_method": "horizon_proxy_2d",
    "target_switch_ms": null,
    "path_efficiency": 0.62,
    "trace": {
      "t_ms":       [0, 16, 33, 50],
      "yaw_deg":    [0.0, -1.2, -4.8, -9.1],
      "pitch_deg":  [0.0, 0.3, 0.9, 1.4],
      "speed_dps":  [0.0, 72.1, 216.4, 258.0]
    },
    "cv_confidence": 0.88,
    "cv_warnings": ["fps=60 limits reaction_time resolution to +/-16.7ms"]
  },

  "assessed": {
    "ai_status": "ok",
    "primary_weakness": "overaiming",
    "severity": 4,
    "confidence": 0.78,
    "summary": "Three corrective micro-adjustments between 1.10s and 1.35s: the first flick overshoots the target by ~8 degrees and the correction oscillates instead of settling.",
    "evidence": [
      { "t_rel_s": 1.10, "observation": "crosshair passes right of the target head box" },
      { "t_rel_s": 1.33, "observation": "crosshair reverses direction, still above head level" }
    ],
    "drill": {
      "name": "KovaaK's 1w6ts Reload",
      "target": "3 sets x 5 min, aim for overshoot count <= 1 per rep",
      "rationale": "Forces a single decisive flick with a hard stop, penalizing the oscillating correction seen here."
    },
    "not_determinable": ["whether the target was already visible before clip start"]
  },

  "provenance": {
    "cv_version": "0.3.1",
    "prompt_version": "clip_v1",
    "provider": "google",
    "model": "gemini-2.0-flash",
    "input_tokens": 1482,
    "output_tokens": 311,
    "estimated_cost_usd": 0.00041,
    "cached": false,
    "analyzed_at": "2026-08-20T14:22:08Z"
  }
}
```

Notes on specific choices:

- `reaction_time_error_ms` sits next to the value, not in documentation. A
  metric without its uncertainty invites over-interpretation of frame noise.
- `crosshair_placement_method` is recorded because the MVP ships a 2D proxy and
  a later version will ship something better; without this field, old and new
  rows silently become incomparable in the trend chart.
- `assessed.not_determinable` gives the model a legitimate place to put "I
  couldn't tell", which measurably reduces the urge to fill the gap.
- `trace` is parallel arrays, not an array of objects — ~4× smaller in SQLite
  and it feeds `uPlot` without a transform.
- `provenance` carries model + prompt version + cost so the dashboard can show
  spend, and so a rubric change is traceable in the data rather than a mystery.

---

## 6. Dashboard

- **Timeline scrubber** — the whole match as a strip: kills (green), deaths
  (red), whiffs (amber), AI-analyzed clips (outlined). Click seeks the `<video>`
  to `start_ms`. Scrub the proxy transcode, not the source file.
- **Crosshair-trace overlay** — the `measured.trace` drawn on a canvas over the
  video, fading tail, overshoot points marked. This is the screenshot that sells
  the product: it makes an invisible mistake visible in one glance.
- **Four metric cards** — reaction time, smoothness, jitter, placement — each
  with the value, the reference band, and a session sparkline.
- **Weakness ranking** — from the synthesis call, ordered by frequency ×
  severity, each row expanding into the clips that evidence it.
- **Spend meter** — session cost so far against the ceiling, always visible.
  Users trust a tool that shows the meter running.

---

## 7. Build order

| Phase | Scope | Proves |
|---|---|---|
| 1 (wk 1–2) | Tauri shell, ingest, ffprobe, proxy transcode, scrubber, SQLite | The plumbing works on real OBS captures, including VFR ones |
| 2 (wk 2–4) | **Aim-trainer** CV: blob targets, score OCR, motion trace, all metrics | Metrics are correct where near-ground-truth is available |
| 3 (wk 4–5) | Keychain, provider adapters, budget guard, cache, per-clip prompt | The AI tier and, more importantly, the cost ceiling |
| 4 (wk 5–6) | Dashboard, trace overlay, synthesis call | The product is demoable |
| 5 (wk 6–8) | Valorant/CS2 profile: kill feed, ammo OCR, hitmarkers | The hard CV, on a foundation already validated |

Do phase 2 before phase 5. Aim trainers give you a near-ground-truth
environment to validate the metric math in; debugging SPARC and kill-feed OCR
simultaneously is how this project stalls.

### Risks worth naming now

- **Kill-feed OCR across patches.** Games restyle HUDs. Profiles are data files
  precisely so a patch is a YAML edit, not a release. Ship a "HUD looks wrong?"
  reporter.
- **Anti-cheat.** Read files, never touch a running game process. No overlay, no
  hooks, no memory reads, no packet capture. Post-hoc analysis of a saved video
  file is unambiguously safe; anything else risks a user's account.
- **VFR captures.** OBS output is often variable-frame-rate; `cv2.VideoCapture`
  frame indices will be wrong. Use PyAV PTS, and if VFR is detected, offer to
  remux to CFR before analysis.
- **Metric validity.** SPARC and jitter thresholds are borrowed from motor
  control; validate them against a handful of known-skill players before showing
  users a number that implies more precision than you've earned.
