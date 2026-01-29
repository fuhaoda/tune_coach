from __future__ import annotations

import time

import numpy as np

from tune_coach.audio import AudioInput
from tune_coach.pitch import PitchTracker


class CalibrationError(RuntimeError):
    pass


def calibrate_do_from_stream(
    audio: AudioInput,
    pitch: PitchTracker,
    seconds: float = 4.0,
    window_min: float = 0.5,
    window_max: float = 1.5,
    target_window: float = 1.0,
) -> float:
    was_running = audio.is_running
    if not was_running:
        audio.start()
    start = time.monotonic()
    records: list[tuple[float, float, float | None]] = []

    try:
        while True:
            now = time.monotonic()
            if now - start >= seconds:
                break
            frame = audio.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            rms = float(np.sqrt(np.mean(np.square(frame))))
            hz = pitch.process(frame)
            records.append((now - start, rms, hz))
            time.sleep(0.005)
    finally:
        if not was_running:
            audio.stop()

    do_hz = _pick_do_hz(
        records,
        rms_min=max(pitch.silence_rms * 1.4, 0.02),
        window_min=float(window_min),
        window_max=float(window_max),
        target_window=float(target_window),
    )
    if do_hz is None:
        raise CalibrationError("Calibration failed")
    if not (50.0 <= do_hz <= 800.0):
        raise CalibrationError("Do out of expected range; try again closer to mic.")
    return do_hz


def _pick_do_hz(
    records: list[tuple[float, float, float | None]],
    *,
    rms_min: float,
    window_min: float,
    window_max: float,
    target_window: float,
) -> float | None:
    if len(records) < 5:
        return None
    ts = np.array([t for t, _, _ in records], dtype=np.float32)
    rms = np.array([r for _, r, _ in records], dtype=np.float32)
    hz = np.array([h if h is not None else np.nan for _, _, h in records], dtype=np.float32)

    if ts.size < 2:
        return None
    dt = np.diff(ts)
    dt = dt[dt > 0]
    median_dt = float(np.median(dt)) if dt.size else 0.01

    windows = [target_window, window_max, window_min]
    seen: set[float] = set()
    for w in windows:
        if w <= 0 or w in seen:
            continue
        seen.add(w)
        best = _best_window_hz(ts, rms, hz, rms_min, w, window_min, median_dt)
        if best is not None:
            return best
    return None


def _best_window_hz(
    ts: np.ndarray,
    rms: np.ndarray,
    hz: np.ndarray,
    rms_min: float,
    window_len: float,
    window_min: float,
    median_dt: float,
) -> float | None:
    best_score = -1.0
    best_hz: float | None = None
    for i in range(ts.size):
        end_t = ts[i] + float(window_len)
        j = int(np.searchsorted(ts, end_t, side="right"))
        if j <= i + 1:
            continue
        seg_hz = hz[i:j]
        seg_rms = rms[i:j]
        voiced_mask = np.isfinite(seg_hz) & (seg_rms >= rms_min)
        if not np.any(voiced_mask):
            continue
        voiced_hz = seg_hz[voiced_mask]
        voiced_rms = seg_rms[voiced_mask]
        voiced_duration = float(voiced_hz.size) * median_dt
        if voiced_duration < window_min:
            continue
        median_hz = float(np.median(voiced_hz))
        if median_hz <= 0:
            continue
        semitone = 12.0 * np.log2(voiced_hz / median_hz)
        mad = float(np.median(np.abs(semitone)))
        if mad > 0.45:
            continue
        score = float(np.mean(voiced_rms)) * (voiced_duration / float(window_len))
        if score > best_score:
            best_score = score
            best_hz = median_hz
    return best_hz
