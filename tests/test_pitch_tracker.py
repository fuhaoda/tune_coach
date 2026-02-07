from __future__ import annotations

import numpy as np

from tune_coach.pitch import PitchTracker, PitchTrackerConfig


def test_pitch_tracker_detects_stable_tone() -> None:
    sample_rate = 44_100
    cfg = PitchTrackerConfig(sample_rate=sample_rate)
    tracker = PitchTracker(cfg)

    freq = 220.0
    t = np.arange(0, sample_rate // 2, dtype=np.float32) / sample_rate
    wave = 0.3 * np.sin(2 * np.pi * freq * t)

    detected = []
    for i in range(0, len(wave), cfg.hop_size):
        block = wave[i : i + cfg.hop_size]
        if block.size == 0:
            continue
        hz = tracker.process(block)
        if hz is not None:
            detected.append(hz)

    assert detected
    assert abs(float(np.median(np.array(detected))) - freq) < 5.0
