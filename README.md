# Tune Coach

Tune Coach is a lightweight Python desktop app that captures your vocal pitch in real time and renders a rolling **Jianpu** (numbered notation) trace. It focuses on **fast Do calibration**, **clear visual feedback**, and **keyboard-based reference tones** while you sing.

## Features

- **Do calibration (4s)**: sing a single Do within 4 seconds. The app finds the loudest stable 0.5–1.5s window and uses its median Hz as Do.
- **Manual Do input**: default Do is **130.8 Hz**; you can edit it and press **Enter**.
- **Tuning systems**: **Just Intonation** (default) or **Equal Temperament**.
- **Keyboard instrument**: play scale tones using keys `1..7` (Piano/Guitar synth).
- **Rolling pitch trace**: shows **24 seconds** of Jianpu steps with 1‑second grid lines.
- **Metronome**: optional BPM click.

> This app visualizes pitch and timing; it does not grade accuracy.

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

Or use the CLI entrypoint:

```bash
tune-coach
```

## Quick start

1. Launch the app and click **Start** to begin listening (default Do = 130.8 Hz).
2. If you want your own Do:
   - Click **Calibrate (4s)** and sing a clear **Do** within 4 seconds, or
   - Type a Do frequency and press **Enter**.
3. Optional: enable **Metronome** and set BPM.
4. Use keyboard tones to check pitch or guide practice.

## Keyboard instrument

- `1..7`: play Do–Ti (current octave)
- `Shift + 1..7`: high octave
- `Control + 1..7`: low octave
- Hold key to sustain; release to stop.
- Works when the app window is focused.

## Pitch display (Jianpu)

- Do is `1`. Incoming pitch is mapped to the nearest major‑scale degree.
- Octaves are shown by dots above/below the digits.
- Tuning rules follow the dropdown selection:
  - **Just Intonation** (natural major scale ratios)
  - **Equal Temperament** (12‑TET)

## Troubleshooting

- **No sound / no pitch**: sing louder or closer to the mic; check system input device.
- **Calibration failed**: sing a clearer, steadier Do; avoid heavy vibrato.
- **Keyboard instrument silent**: make sure the app window has focus.
- **Missing deps**:
  - `python -m pip install -e .`
  - Verify interpreter points to `.venv`.
