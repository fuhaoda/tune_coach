from __future__ import annotations

import numpy as np

from tune_coach.calibration import _pick_do_hz


def test_pick_do_hz_selects_stable_window() -> None:
    records: list[tuple[float, float, float | None]] = []
    ts = np.arange(0.0, 4.0, 0.01)
    for t in ts:
        if 1.0 <= t <= 2.2:
            hz = 196.0 + 0.5 * np.sin(t * 8)
            rms = 0.09
        else:
            hz = None
            rms = 0.01
        records.append((float(t), float(rms), None if hz is None else float(hz)))

    do_hz = _pick_do_hz(
        records,
        rms_min=0.02,
        window_min=0.5,
        window_max=1.5,
        target_window=1.0,
    )

    assert do_hz is not None
    assert abs(do_hz - 196.0) < 2.0
