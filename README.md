# Tune Coach

Tune Coach is a lightweight Python desktop app that captures your vocal pitch in real time and renders a rolling **Jianpu** (numbered notation) trace. It focuses on **fast Do calibration**, **clear visual feedback**, **key transposition**, **keyboard reference tones**, and **record + pitch‑shift playback** so you can hear “what one step higher/lower sounds like” using your own voice.

## Features

- **Do calibration (4s)**: sing a single Do within 4 seconds. The app finds the loudest stable 0.5–1.5s window and uses its median Hz as Do.
- **Do presets**: **Male (130.8 Hz)** and **Female (261.6 Hz)**.
- **Manual Do input**: type a frequency and press **Enter**.
- **Tuning systems**: **Just Intonation** (default) or **Equal Temperament**.
- **Key (1=Key) transposition**: set the song key (e.g. 1=G) and all 1..7 notes follow it.
- **Pause / Resume**: pause the session to inspect the curves, then resume.
- **Keyboard instrument**: play scale tones using keys `1..7` (default **Guitar**, switchable to Piano).
- **Record + Shift + Play**: record up to 10 seconds, then play back your voice shifted by ±5 scale steps.
- **Rolling pitch trace**: shows **24 seconds** of Jianpu steps with 1‑second grid lines.
- **Cent curve overlay**: optional continuous pitch curve aligned to the Jianpu axis.
- **Metronome**: optional BPM click.
- **Reference lines**: two guide lines at degree 7 (low/mid) for quick orientation.

> This app visualizes pitch and timing; it does not grade accuracy.

## Install

Recommended: Python 3.10+ (tested on macOS).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

This installs `librosa`, `soundfile`, and `pyworld` (for higher‑quality voice shifting).

Note: You may see a macOS console warning like:
`error messaging the mach port for IMKCFRunLoopWakeUpReliable` — it is benign.

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
   - Select **Male/Female** presets, or
   - Type a Do frequency and press **Enter**.
3. Optional: enable **Metronome** and set BPM.
4. Optional: enable **Cent Curve** to see your continuous pitch track.
5. Choose **Key (1=Key)** if the song is transposed (default 1=C).
6. Use **Pause** to freeze the display; click again to **Resume**.
7. Use keyboard tones to check pitch or guide practice (default instrument is Guitar).
8. To record and shift your voice:
   - Click **Start** (record/play only works while listening).
   - Press and hold **Recording**, sing, then release.
   - Choose **Shift -5..+5 steps**.
   - Click **Play** to hear the shifted voice once.

## Keyboard instrument

- `1..7`: play Do–Ti (current octave)
- `Shift + 1..7`: high octave
- `Control + 1..7`: low octave
- Hold key to sustain; release to stop.
- Works when the app window is focused.

## Record + Shift + Play

- **Press‑and‑hold** Recording, then release to stop (max 10 seconds).
- **Shift steps** are in major‑scale degrees (natural scale).
- **Play** plays once. **Stop** cancels recording/playback.
- The metronome is not mixed into the recording.

## Pitch display (Jianpu)

- Do is `1`. Incoming pitch is mapped to the nearest major‑scale degree.
- Octaves are shown by dots above/below the digits.
- Tuning rules follow the dropdown selection:
  - **Just Intonation** (natural major scale ratios)
  - **Equal Temperament** (12‑TET)
- **Cent Curve** (optional) shows the continuous pitch path aligned to the same Jianpu axis and tuning.
- **Key transposition**: Do is treated as the base C. Selecting 1=G, 1=F#, etc. shifts the scale so that 1 maps to that key, then 2..7 follow the selected tuning.

## Troubleshooting

- **No sound / no pitch**: sing louder or closer to the mic; check system input device.
- **Calibration failed**: sing a clearer, steadier Do; avoid heavy vibrato.
- **Keyboard instrument silent**: make sure the app window has focus.
- **Playback says “Processing…” too long**: try a shorter recording or fewer steps; check `pyworld` is installed.
- **Missing deps**:
  - `python -m pip install -e .`
  - Verify interpreter points to `.venv`.
