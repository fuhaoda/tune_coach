from __future__ import annotations

import math
import time

import numpy as np

from tune_coach.audio import AudioInput
from tune_coach.pitch import PitchTracker


class CalibrationError(RuntimeError):
    pass


def calibrate_do_from_stream(audio: AudioInput, pitch: PitchTracker, seconds: float = 4.0) -> float:
    was_running = audio.is_running
    if not was_running:
        audio.start()
    start = time.monotonic()
    records: list[tuple[float, float]] = []

    try:
        while True:
            now = time.monotonic()
            if now - start >= seconds:
                break
            frame = audio.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            hz = pitch.process(frame)
            if hz is not None:
                records.append((now - start, hz))
            time.sleep(0.005)
    finally:
        if not was_running:
            audio.stop()

    # Split the 4s window into three stable note segments (1-2-3).
    segments = _segment_three_notes(records)
    do_hz = float(np.median(np.array([hz for _, hz in segments[0]], dtype=np.float32)))
    if not (50.0 <= do_hz <= 800.0):
        raise CalibrationError("Do out of expected range; try again closer to mic.")
    return do_hz


def _segment_three_notes(records: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    if len(records) < 20:
        raise CalibrationError("Not enough voiced frames; sing louder/clearer.")

    ts = np.array([t for t, _ in records], dtype=np.float32)
    hz = np.array([h for _, h in records], dtype=np.float32)
    midi = 69.0 + 12.0 * np.log2(hz / 440.0)

    # Robust smoothing
    midi_sm = _median_filter(midi, k=5)

    # Primary segmentation: quantize to semitone + run-length encode.
    semitone = np.round(midi_sm).astype(np.int32)
    runs = _runs(semitone)
    runs = _merge_short_runs(runs, ts, min_duration=0.15)
    runs = [r for r in runs if float(ts[r[1] - 1] - ts[r[0]]) >= 0.25]

    if len(runs) >= 3:
        first3 = runs[:3]
        med = [int(np.median(semitone[a:b])) for a, b in first3]
        if not (med[0] < med[1] < med[2]):
            raise CalibrationError("Please sing ascending 1-2-3 (three stable notes).")
        return [[(float(ts[i]), float(hz[i])) for i in range(a, b)] for a, b in first3]

    # Fallback: detect change points (approx >= 1 semitone)
    dm = np.abs(np.diff(midi_sm))
    change_idx = np.where(dm >= 1.0)[0] + 1
    if change_idx.size == 0:
        raise CalibrationError("Cannot find note changes; avoid heavy vibrato/glissando.")

    # Build candidate segments by splitting on changes, then keep stable ones
    boundaries = [0, *change_idx.tolist(), len(midi_sm)]
    raw_segments: list[tuple[int, int]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < 5:
            continue
        seg = midi_sm[a:b]
        if float(np.std(seg)) <= 0.35 and float(ts[b - 1] - ts[a]) >= 0.25:
            raw_segments.append((a, b))

    if len(raw_segments) < 3:
        # Fallback: greedily merge into 3 segments by selecting 3 longest stable regions
        raw_segments = sorted(raw_segments or [(0, len(midi_sm))], key=lambda x: (x[1] - x[0]), reverse=True)

    # Take first 3 stable segments in time order (merge overlaps/adjacent)
    raw_segments = sorted(raw_segments, key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    for a, b in raw_segments:
        if not merged:
            merged.append((a, b))
            continue
        pa, pb = merged[-1]
        if a <= pb + 3:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    if len(merged) < 3:
        # Last resort: pick 3 chunks split by 2 biggest change points
        top2 = np.argsort(dm)[-2:]
        cuts = sorted((top2 + 1).tolist())
        boundaries = [0, *cuts, len(midi_sm)]
        merged = [(boundaries[i], boundaries[i + 1]) for i in range(3)]

    merged = merged[:3]
    if len(merged) != 3:
        raise CalibrationError("Expected 3 notes; try again.")

    segments: list[list[tuple[float, float]]] = []
    for a, b in merged:
        seg = [(float(ts[i]), float(hz[i])) for i in range(a, b)]
        segments.append(seg)

    return segments


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = float(np.median(xp[i : i + k]))
    return out


def _runs(x: np.ndarray) -> list[tuple[int, int]]:
    if x.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, x.size):
        if x[i] != x[i - 1]:
            runs.append((start, i))
            start = i
    runs.append((start, x.size))
    return runs


def _merge_short_runs(
    runs: list[tuple[int, int]], ts: np.ndarray, *, min_duration: float
) -> list[tuple[int, int]]:
    if not runs:
        return runs

    merged: list[tuple[int, int]] = []
    for a, b in runs:
        dur = float(ts[b - 1] - ts[a])
        if dur < min_duration and merged:
            pa, pb = merged[-1]
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))

    # If the first run is still too short, merge it forward.
    if len(merged) >= 2:
        a, b = merged[0]
        if float(ts[b - 1] - ts[a]) < min_duration:
            na, nb = merged[1]
            merged = [(a, nb), *merged[2:]]
    return merged
