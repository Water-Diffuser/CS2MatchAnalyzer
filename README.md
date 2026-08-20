# Gameplay Mechanics Analyzer

Analyze local gameplay and aim-trainer recordings (Valorant, CS2, Overwatch,
Aimlabs, KovaaK's) and return actionable feedback on aim mechanics.

**Local-first computer vision does the measuring. A user-supplied LLM/VLM does
the explaining.** That split is the core design decision — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

> Status: architectural blueprint + reference implementation of the AI engine.
> The CV pipeline is specified but not yet built; see the build order in §7.

## Why this shape

A vision model cannot tell you a reaction time was 214 ms — it sees sampled
frames and has no frame-accurate clock. OpenCV can, on the user's own CPU, for
free. Equally, OpenCV cannot tell you *why* your flick overshot. So the pipeline
measures deterministically first, then passes those numbers to the model as
labeled ground truth and asks only for the explanation.

Two consequences fall out of that:

- **No cloud GPU bill.** Video never leaves the user's machine except as ~8
  short frame grids per match, sent to their own API key. Marginal cost per
  user: $0.
- **The AI tier is optional.** With no key configured, every metric and chart
  still works. Nothing load-bearing sits behind an API call.

## Layout

| Path | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The blueprint: pipeline, CV strategy, key security, cost control, prompt engineering, build order |
| [`schemas/clip_analysis.v1.schema.json`](schemas/clip_analysis.v1.schema.json) | Canonical record for one analyzed engagement |
| [`reference/ai_engine/`](reference/ai_engine/) | Provider adapters, keychain storage, budget guard, frame-grid packaging |
| [`reference/analyze_clip.py`](reference/analyze_clip.py) | End-to-end example: timestamp in, structured analysis out |
| [`tests/test_schema.py`](tests/test_schema.py) | Contract tests between CV, AI, and storage |
| [`config/model_prices.json`](config/model_prices.json) | Pricing table for the pre-flight cost estimate — populate at build time |

## Quick start

```bash
pip install -r requirements.txt
python tests/test_schema.py

# One engagement at 412.3s into a recording, analyzed by the configured provider
python reference/analyze_clip.py match.mp4 412300 --provider google --budget 0.10
```

Keys are read from the OS keychain (macOS Keychain / Windows Credential Manager
/ Secret Service), falling back to `GEMINI_API_KEY` / `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` **for development only**. A shipped build stores keys in the
keychain via the Rust side, where the webview cannot reach them.

## Cost control

Five layers, all client-side, all enforced before a request is sent:
event selection (~8 clips/match), 3×3 frame grids instead of video, cheap model
per clip with one strong-model synthesis, a pre-flight estimate plus a hard
ceiling (default $0.50/session), and content-addressed caching so re-analysis is
free. Details in §2.3 of the architecture doc.

## Anti-cheat safety

This tool reads saved video files and nothing else. No overlay, no process
hooks, no memory reads, no packet capture. Post-hoc analysis of a recording is
unambiguously safe; anything else risks a user's account.
