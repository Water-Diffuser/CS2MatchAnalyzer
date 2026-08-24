# Getting started (Windows)

No installing, no account, no Python. Download two files, drag a video onto
one of them, look at the pictures that come out.

Everything happens on your own PC. Nothing is uploaded anywhere.

---

## 1. Download

1. Go to the **[Actions tab](../../actions/workflows/build.yml)** of this repo.
2. Click the newest run at the top with a green tick next to it.
3. Scroll to the bottom, to a box called **Artifacts**.
4. Click **gameplay-analyzer-windows-x86_64**. A `.zip` downloads.

## 2. Unzip it

Right-click the downloaded `.zip` → **Extract All** → **Extract**.

Inside is **one file**: `gameplay-analyzer.exe`. That is the whole program —
nothing to install, nothing that has to sit next to anything else. Put it
wherever you like.

## 3. Get past the Windows warning

The first time you run it, Windows shows a blue box:
**"Windows protected your PC"**.

Click **More info**, then **Run anyway**.

This is expected and it is not a virus warning. Windows shows it for any
program it has not seen many people run before. Removing it permanently
requires a code-signing certificate costing a few hundred dollars a year —
not something an open-source hobby project usually has.

If you would rather not take that on faith, the whole source is in this repo
and the binary is built in public by
[GitHub Actions](../../actions/workflows/build.yml).

## 4. Check it works

**Double-click `gameplay-analyzer.exe`.** With no video given it explains
itself, runs a self-check — drawing a test clip with a known camera movement
and confirming the program measures it correctly — and waits for you to press
Enter before closing.

You want to see:

```
OK - computer vision pipeline is working
```

## 5. Analyze a real clip

**Drag your gameplay video onto `gameplay-analyzer.exe`** and let go.

A black window opens and works for a few minutes — longer for a long
recording. When it finishes, a folder opens next to your video called
`yourvideo_analysis`, and the window stays open so you can read the summary.

### What's in it

**The `.png` images** are the useful part. Each shows one engagement, with the
path your crosshair actually took drawn over the frame:

- The **amber line** is where your aim travelled. Brighter means more recent.
- **Red circles** mark where your aim changed direction — overcorrections. A
  clean flick has none; three or four means you're overshooting and hunting
  back.
- The **panel top-left** lists the measurements, and the small graph under it
  plots your horizontal aim against time. Overshoot is easiest to see there: a
  clean flick rises and flattens, an overcorrected one bumps up and drops back.

**`results.json`** is the same information as raw numbers. Ignore it unless you
want it.

---

## Recording tips

- **Record at 60fps or higher.** Below that the timing measurements aren't
  trustworthy, and the tool refuses to report them rather than guessing. OBS,
  ShadowPlay and the Xbox Game Bar (`Win+G`) all do 60fps.
- **Shorter clips are faster.** A 20-second clip takes seconds; a 40-minute VOD
  takes several minutes.
- `.mp4` and `.webm` both work.

## If something goes wrong

| What you see | What to do |
|---|---|
| Nothing happens when you drag a video on | Drop it directly onto the `.exe` icon, not into an open window. |
| "could not open ..." | The video isn't in a readable format. Re-save it as `.mp4`. |
| Self-check fails | Something in the download is damaged. Delete the folder and re-download the zip. |

## What the numbers actually mean

| Name | Plain English |
|---|---|
| **overshoot** | How many times your aim shot past the target and had to come back. Lower is better; 0–1 is good. |
| **path eff.** | How direct your aim was. `1.00` is a perfect straight flick. `0.60` means you travelled nearly twice as far as needed. |
| **smoothness** | A score for how controlled the movement was. Closer to zero is smoother. |
| **jitter** | How much high-frequency shake is in your aim — often grip tension or too much sensitivity. |
| **reaction** | Time from a target appearing to your first shot. Only measurable in aim trainers right now; on game footage it says "not measurable" rather than inventing a number. |

Two honest caveats about those numbers. **Jitter and overshoot move together** —
correcting a flick creates real shake, so a high jitter reading doesn't
necessarily mean your grip is tense; check overshoot alongside it. And the
thresholds for "good" haven't been calibrated against known-skill players yet,
so treat these as *your* numbers to improve on over time, not a verdict.
