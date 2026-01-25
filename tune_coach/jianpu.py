from __future__ import annotations

import math
from dataclasses import dataclass


_MAJOR_SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # 1..7 in major scale (Do-based)


def _dot_above(label: str) -> str:
    # Combining dot above (renders as a dot above the digit in most fonts).
    return f"{label}\u0307"


def _dot_below(label: str) -> str:
    # Combining dot below (renders as a dot below the digit in most fonts).
    return f"{label}\u0323"


class JianpuAxis:
    def __init__(self, octave_gap: int = 0) -> None:
        self._octave_gap = int(max(0, octave_gap))
        # 3 octaves, each 7 degrees, plus gaps between octave groups
        self.max_y = (7 * 3 - 1) + (self._octave_gap * 2)

    def ticks(self) -> list[tuple[int, str]]:
        ticks: list[tuple[int, str]] = []
        # y: 0..20 => low, mid, high
        for octave_index in range(3):
            for degree in range(1, 8):
                y = octave_index * (7 + self._octave_gap) + (degree - 1)
                label = str(degree)
                if octave_index == 0:
                    label = _dot_below(label)
                elif octave_index == 2:
                    label = _dot_above(label)
                ticks.append((y, label))
        return ticks


@dataclass(frozen=True)
class JianpuQuantizer:
    do_hz: float
    octave_gap: int = 0

    def quantize_to_y(self, hz: float) -> int | None:
        if hz <= 0 or self.do_hz <= 0:
            return None
        semitone = int(round(12.0 * math.log2(hz / self.do_hz)))
        best = None
        best_err = 10**9
        for octave in (-1, 0, 1):
            base = octave * 12
            for degree_index, s in enumerate(_MAJOR_SCALE_SEMITONES):
                candidate = base + s
                err = abs(semitone - candidate)
                if err < best_err:
                    best_err = err
                    best = (octave, degree_index)

        if best is None:
            return None
        octave, degree_index = best
        if octave < -1 or octave > 1:
            return None
        octave_gap = int(max(0, self.octave_gap))
        y = (octave + 1) * (7 + octave_gap) + degree_index  # -1->0, 0->1, 1->2
        max_y = (7 * 3 - 1) + (octave_gap * 2)
        if y < 0 or y > max_y:
            return None
        return int(y)
