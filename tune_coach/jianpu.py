from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


_MAJOR_SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # 1..7 in major scale (Do-based)
_JUST_INTONATION_RATIOS = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]
_JUST_INTONATION_CENTS = [1200.0 * math.log2(ratio) for ratio in _JUST_INTONATION_RATIOS]
_EQUAL_TEMPERAMENT_CENTS = [float(semitone * 100) for semitone in _MAJOR_SCALE_SEMITONES]


class TuningSystem(str, Enum):
    EQUAL_TEMPERAMENT = "Equal Temperament"
    JUST_INTONATION = "Just Intonation"


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
    tuning: TuningSystem = TuningSystem.EQUAL_TEMPERAMENT

    def degree_hz(self, degree: int, octave: int = 0) -> float | None:
        if degree < 1 or degree > 7 or self.do_hz <= 0:
            return None
        degree_index = degree - 1
        if self.tuning == TuningSystem.JUST_INTONATION:
            ratio = _JUST_INTONATION_RATIOS[degree_index]
            return float(self.do_hz * ratio * (2.0**octave))
        semitone = _MAJOR_SCALE_SEMITONES[degree_index] + 12 * octave
        return float(self.do_hz * (2.0 ** (semitone / 12.0)))

    def quantize_to_y(self, hz: float) -> int | None:
        if hz <= 0 or self.do_hz <= 0:
            return None
        best = None
        best_err = 10**9
        for octave in (-1, 0, 1):
            for degree_index, semitone in enumerate(_MAJOR_SCALE_SEMITONES):
                if self.tuning == TuningSystem.JUST_INTONATION:
                    target = self.do_hz * _JUST_INTONATION_RATIOS[degree_index] * (2.0**octave)
                else:
                    target = self.do_hz * (2.0 ** ((semitone + 12 * octave) / 12.0))
                if target <= 0:
                    continue
                err = abs(math.log2(hz / target))
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

    def nearest_degree(self, hz: float) -> tuple[int, int] | None:
        if hz <= 0 or self.do_hz <= 0:
            return None
        semitone = 12.0 * math.log2(hz / self.do_hz)
        best = None
        best_err = 10**9
        for octave in range(-3, 4):
            base = octave * 12
            for degree_index, s in enumerate(_MAJOR_SCALE_SEMITONES):
                candidate = base + s
                err = abs(semitone - candidate)
                if err < best_err:
                    best_err = err
                    best = (degree_index + 1, octave)
        return best

    def continuous_y(self, hz: float) -> float | None:
        if hz <= 0 or self.do_hz <= 0:
            return None
        cents_total = 1200.0 * math.log2(hz / self.do_hz)
        octave = math.floor(cents_total / 1200.0)
        if octave < -1 or octave > 1:
            return None
        cents_in_oct = cents_total - (octave * 1200.0)
        degree_cents = (
            _JUST_INTONATION_CENTS
            if self.tuning == TuningSystem.JUST_INTONATION
            else _EQUAL_TEMPERAMENT_CENTS
        )
        idx = 0
        for i in range(len(degree_cents) - 1):
            if cents_in_oct < degree_cents[i + 1]:
                idx = i
                break
        else:
            idx = len(degree_cents) - 1
        start = degree_cents[idx]
        end = 1200.0 if idx == len(degree_cents) - 1 else degree_cents[idx + 1]
        span = max(1.0, end - start)
        frac = (cents_in_oct - start) / span
        frac = float(min(1.0, max(0.0, frac)))
        octave_gap = int(max(0, self.octave_gap))
        y = (octave + 1) * (7 + octave_gap) + idx + frac
        max_y = (7 * 3 - 1) + (octave_gap * 2)
        if y < 0:
            return None
        return float(min(max_y, y))
