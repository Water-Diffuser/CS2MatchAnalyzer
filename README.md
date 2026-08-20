# Gameplay Mechanics Analyzer

Analyze local gameplay and aim-trainer recordings (Valorant, CS2, Overwatch,
Aimlabs, KovaaK's) and return actionable feedback on aim mechanics.

**Local computer vision does the measuring. A user-supplied LLM/VLM does the
explaining.** That split is the core design decision — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why this shape

A vision model cannot tell you a reaction time was 214 ms — it sees sampled
frames and has no frame-accurate clock. OpenCV can, on the user's own CPU, for
free. Equally, OpenCV cannot tell you *why* your flick overshot. So the pipeline
measures deterministically first, then passes those numbers to the model as
labeled ground truth and asks only for the explanation.

Two consequences follow:

- **No cloud GPU bill.** Video never leaves the machine except as ~8 short
  frame grids per match, sent to the user's own API key. Marginal cost per
  user: $0.
- **The AI tier is optional.** With no key configured every metric still
  computes and the record is still schema-valid. Nothing load-bearing sits
  behind an API call.

## Status

| Stage | State |
|---|---|
| 3 · Event detection — shots, hits, whiffs, kill feed, targets | built, tested |
| 4 · Motion trace + mechanical metrics | built, tested |
| 5 · Coachability ranking and clip selection | built, tested |
| 6/9 · Packaging and schema-valid records | built, tested |
| 10 · Trace overlay rendering | built |
| Packaging · standalone binary, CI matrix | built, tested frozen |
| 7/8 · AI engine — provider adapters, keychain, budget guard | reference implementation |
| 1/10 · Tauri shell, ingest UI, dashboard | not started |

78 tests pass in ~1 minute. The pipeline runs end to end on a video file
today; what it does not yet have is a UI.

## Measured accuracy

The motion estimator is the one component whose output cannot be eyeballed — a
plausible trace and a correct trace look identical. So
[`analyzer/synthetic.py`](analyzer/synthetic.py) renders footage along a known
angular path, and [`tests/test_motion.py`](tests/test_motion.py) scores the
recovered trace against it. At 1280×720, 103° FOV, 60 fps:

| Case | Peak error |
|---|---|
| 24° steady pan | 0.073° |
| 45° flick in 120 ms | 0.040° |
| yaw + pitch combined | 0.064° / 0.438° |
| with a counter-moving distractor | 0.092° (RANSAC cost: +0.014°) |
| stationary camera, 40 frames | 0.008° drift |
| scale linearity, 10°–45° | 0.9894× – 0.9996× |
| untrackable footage (smoke, flash) | confidence 0.00, no invented motion |

Ammo-counter reading is exact across 0–100. Kill-feed dedup collapses 120
frames of one row into one event timed at first appearance. On aim-trainer
footage the full reaction-time loop — target spawn detected, shot detected,
interval measured — returns 216.7 ± 16.7 ms against a ground truth of
216.7 ms.

## Download

Standalone executables — no Python, no install, no account — are built for
Linux, Windows, and both macOS architectures by
[`.github/workflows/build.yml`](.github/workflows/build.yml). Grab one from the
latest run's artifacts, or from a release if the commit is tagged.

```
gameplay-analyzer selftest                   # verify the build works on your machine
gameplay-analyzer analyze match.mp4 --profile valorant --max-clips 8
gameplay-analyzer overlay match.mp4 --start-ms 412300 --out flick.png
gameplay-analyzer doctor                     # versions and bundled resources
```

`selftest` is worth running first: it renders footage along a known 45° flick
and checks the bundled OpenCV recovers it to within a tolerance, which answers
"is this download working" without needing a recording to hand.

On macOS the binary is unsigned, so Gatekeeper will block the first run —
`xattr -d com.apple.quarantine gameplay-analyzer`, or right-click → Open.
Signing needs a paid Apple Developer account.

## Building from source

```bash
pip install -r requirements-dev.txt
python tests/run_all.py                      # ~1 min: renders video, checks CV accuracy
pyinstaller gameplay-analyzer.spec --noconfirm
```

**PyInstaller cannot cross-compile.** A binary is produced by, and for, the OS
it was built on, which is the entire reason the CI matrix exists — it is the
only way to produce a Windows executable from this repo.

The bundle is ~82 MB, almost all of it OpenCV and NumPy. Two decisions keep it
from being far worse:

- **SciPy is not a runtime dependency.** The one function used from it,
  `signal.welch`, is reimplemented in [`analyzer/psd.py`](analyzer/psd.py) and
  verified against SciPy to within 1e-16 in
  [`tests/test_psd.py`](tests/test_psd.py). That is ~112 MB — more than the rest
  of the application put together — for one function.
- **`opencv-python-headless`, not `opencv-python`.** The GUI build pulls in
  GTK/Qt system libraries that are absent on a clean machine, turning a working
  binary into an import error at startup.

Running from source works too, without building anything:

```bash
python -m analyzer analyze match.mp4 --profile valorant

# AI tier — needs a key in the OS keychain
python reference/analyze_clip.py match.mp4 412300 --provider google --budget 0.10
```

## Layout

| Path | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The blueprint: pipeline, CV strategy, key security, cost control, prompt engineering, build order |
| [`analyzer/metrics.py`](analyzer/metrics.py) | SPARC smoothness, jitter, overshoot, reaction time, path efficiency |
| [`analyzer/motion.py`](analyzer/motion.py) | Camera motion from pixels: masked optical flow + RANSAC |
| [`analyzer/events.py`](analyzer/events.py) | Ammo/hitmarker/kill-feed detection, whiff derivation |
| [`analyzer/pipeline.py`](analyzer/pipeline.py) | Stages 1–6 and 9 |
| [`analyzer/cli.py`](analyzer/cli.py) | `analyze`, `overlay`, `profiles`, `doctor`, `selftest` |
| [`analyzer/psd.py`](analyzer/psd.py) | Welch PSD in NumPy, so SciPy stays out of the bundle |
| [`gameplay-analyzer.spec`](gameplay-analyzer.spec) | PyInstaller build spec |
| [`analyzer/overlay.py`](analyzer/overlay.py) | Crosshair-trace overlay and yaw-vs-time sparkline |
| [`analyzer/synthetic.py`](analyzer/synthetic.py) | Ground-truth footage generator (test infrastructure) |
| [`profiles/`](profiles/) | Per-game ROIs and FOV, as data |
| [`schemas/clip_analysis.v1.schema.json`](schemas/clip_analysis.v1.schema.json) | Canonical record for one analyzed engagement |
| [`reference/ai_engine/`](reference/ai_engine/) | Provider adapters, keychain storage, budget guard, frame grids |

## Adding a game

HUD layouts change with patches, so everything title-specific is data. Copy
[`profiles/valorant.yaml`](profiles/valorant.yaml), adjust the normalized ROIs
and FOV, and pass `--profile path/to/yours.yaml`. No code changes.

## Cost control

Five layers, all client-side, all enforced before a request is sent: event
selection (~8 clips/match), 3×3 frame grids instead of video, a cheap model per
clip with one strong-model synthesis, a pre-flight estimate plus a hard ceiling
(default $0.50/session), and content-addressed caching so re-analysis is free.
Details in §2.3 of the architecture doc.

## Anti-cheat safety

This reads saved video files and nothing else. No overlay, no process hooks, no
memory reads, no packet capture. Post-hoc analysis of a recording is
unambiguously safe; anything else risks a user's account.

## Known limitations

- **Crosshair placement is a 2D proxy.** Without scene depth, true head-level
  error is not computable. Every record carries
  `crosshair_placement_method` so a later depth-aware version stays
  distinguishable in trend charts rather than being silently averaged in.
- **Jitter and overshoot are correlated, not independent.** Corrective
  submovements inject genuine broadband energy, so a heavily-corrected flick
  reads as elevated jitter even with a steady hand. Do not present a high
  `jitter_ratio` as a grip/sensitivity diagnosis without cross-checking
  `overshoot_count`.
- **Reaction time needs 60 fps or better.** Below ~50 fps the pipeline declines
  to report it rather than publishing a number that is half quantization noise.
- **The motion model is a translation, not a rotation homography.** It inverts
  exactly near the optical axis and degrades toward frame edges. Fine for the
  MVP; a version wanting sub-0.1° accuracy at wide FOV should fit a rotation.
- **Metric thresholds are uncalibrated.** SPARC and jitter bands come from
  motor-control literature. They need validating against known-skill players
  before any number is shown to a user as a verdict.
- **Reaction time needs a stimulus detector, which only trainers have.** On
  tactical-shooter footage there is no reliable way yet to localize the moment
  an enemy became visible, so `reaction_time_ms` is `null` with a warning
  rather than a plausible-looking constant. Real support needs an enemy
  detector — a fine-tuned YOLOv8n would do it — which is the obvious next
  piece of work.
- **Whiffs need a hitmarker template, which ships per game.** Without one the
  pipeline reports `tracking_phase` rather than claiming a whiff it cannot
  observe.
- **Kill-feed detection is the least robust component.** It relies on feed rows
  having a dominant background colour (measured: 0.91 for a feed row, 0.28–0.60
  for scenery). Titles with a fully transparent feed will need a different
  discriminator, and the threshold is per-profile for that reason.
