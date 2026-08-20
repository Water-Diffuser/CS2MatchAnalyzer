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

## 2. Unzip it — this step is not optional

Right-click the downloaded `.zip` → **Extract All** → **Extract**.

> **Do not double-click the launcher while it is still inside the zip.**
> Windows will happily open it from there, but it copies out only that one
> file and leaves the program behind, so the launcher starts up and then
> reports that it cannot find `gameplay-analyzer.exe`. It detects this case
> and tells you, but extracting first avoids it entirely.

After extracting you get a folder with two files:

| File | What it is |
|---|---|
| `gameplay-analyzer.exe` | the actual program |
| `Analyze Video.bat` | what you double-click |

**Keep these two together in the same folder.** The launcher looks for the
program right next to itself, and will tell you so if you separate them.

## 3. Get past the Windows warning

The first time you run it, Windows will show a blue box:
**"Windows protected your PC"**.

Click **More info**, then **Run anyway**.

This is expected and it is not a virus warning. Windows shows it for any
program it hasn't seen many people run before. Getting rid of it permanently
requires a code-signing certificate, which costs a few hundred dollars a year —
not something an open-source hobby project usually has.

If you would rather not take that on faith, the entire source is in this repo
and the binary is built in public by
[GitHub Actions](../../actions/workflows/build.yml) — you can read exactly what
went into it.

## 4. Check it works

**Double-click `Analyze Video.bat`.** With no video given, it offers to run a
self-check: it draws a test clip with a known camera movement and confirms the
program measures it correctly.

You want to see:

```
OK - computer vision pipeline is working
```

## 5. Analyze a real clip

**Drag your gameplay video onto `Analyze Video.bat`** and let go.

A black window opens and works for a few minutes — longer for a long
recording. When it finishes, a folder opens next to your video called
`yourvideo_analysis`.

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
| "Could not find gameplay-analyzer.exe" | The two files got separated. Put them back in one folder. |
| Window flashes and vanishes | You ran the `.exe` directly instead of the `.bat`. Use `Analyze Video.bat`. |
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
