from __future__ import annotations

import numpy as np

from tune_coach.jianpu import TuningSystem
from tune_coach.web.realtime_engine import compute_degree_shift_ratio


def test_shift_ratio_is_positive() -> None:
    sample_rate = 44_100
    t = np.arange(0, sample_rate, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)

    ratio = compute_degree_shift_ratio(
        audio,
        sample_rate,
        steps=2,
        do_hz=130.8,
        tuning=TuningSystem.JUST_INTONATION,
        key_semitone=0,
    )

    assert ratio > 0.0
    assert ratio != 1.0
