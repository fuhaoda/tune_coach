# Tune Coach (MVP)

A lightweight macOS Python app that captures your vocal pitch in real time and renders a 12‑second rolling **Jianpu** (numbered notation) trace. It supports a 4‑second calibration (sing `1‑2‑3`, auto‑detect three stable notes) and an optional metronome.

## Features

- **Calibrate (4s)**: sing `1‑2‑3`; the app finds three stable note segments and treats your **Do as 1**.
- **Start/Stop**: shows the last **12 seconds** of discrete Jianpu pitch with 1‑second vertical grid lines.
- **Metronome**: toggle + BPM control (default 96 BPM).

> Note: This MVP does not judge “accuracy”; it only visualizes your pitch contour and timing.

## Install

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

No extra `aubio` is required (this project uses a pure‑numpy pitch tracker to avoid build issues).

## Run

```bash
python -m tune_coach
```

## How to use

1. Make sure your system input device is the microphone you want (the app uses the system default).
2. Click **Calibrate** and sing exactly three notes in ~4 seconds: `1 → 2 → 3` (durations can vary).
3. Click **Start** to see the rolling pitch trace; **Stop** to pause.
4. Enable **Metronome** if desired and set BPM.

## Pitch display (discrete Jianpu)

- After calibration, Do is `1`. Incoming pitch is converted to semitone distance (12‑TET) and **snapped to the nearest major‑scale Jianpu** step: `1 2 3 4 5 6 7` (octaves use dots above/below).
- This makes it easy to see if you reach the next scale step (e.g., moving from `3` toward `4`).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'numpy'`**
  - Dependencies did not install, or you are not running inside `.venv`.
  - Reinstall: `python -m pip install -e .`
  - Verify interpreter: `which python` should point to `.../tune_coach/.venv/bin/python`
- **Jittery line**: record in a quieter environment, sing louder, or get closer to the mic.
- **Calibration fails**: you likely did not hold three stable notes; try again with clearer note changes.
