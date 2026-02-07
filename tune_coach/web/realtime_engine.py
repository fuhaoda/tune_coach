from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from tune_coach.calibration import _pick_do_hz
from tune_coach.jianpu import JianpuQuantizer, TuningSystem
from tune_coach.pitch import PitchTracker, PitchTrackerConfig


@dataclass(frozen=True)
class EngineUiConfig:
    delay_seconds: float = 0.25
    smooth_window: int = 11
    cent_smooth_window: int = 3
    stable_min_count: int = 7
    switch_confirm: int = 3
    silence_timeout: float = 0.2
    calibration_window_min: float = 0.5
    calibration_window_max: float = 1.5
    calibration_target_window: float = 1.0


class CalibrationCollector:
    def __init__(self, ui: EngineUiConfig) -> None:
        self._ui = ui
        self.active = False
        self._start_t = 0.0
        self._seconds = 4.0
        self._records: list[tuple[float, float, float | None]] = []

    def start(self, now: float, seconds: float) -> None:
        self.active = True
        self._start_t = float(now)
        self._seconds = float(seconds)
        self._records.clear()

    def cancel(self) -> None:
        self.active = False
        self._records.clear()

    def push(
        self,
        now: float,
        rms: float,
        hz: float | None,
        *,
        silence_rms: float,
    ) -> tuple[bool, float | None]:
        if not self.active:
            return False, None
        self._records.append((float(now - self._start_t), float(rms), hz))
        if (now - self._start_t) < self._seconds:
            return False, None

        self.active = False
        do_hz = _pick_do_hz(
            self._records,
            rms_min=max(float(silence_rms) * 1.4, 0.02),
            window_min=self._ui.calibration_window_min,
            window_max=self._ui.calibration_window_max,
            target_window=self._ui.calibration_target_window,
        )
        if do_hz is None:
            return True, None
        if not (50.0 <= float(do_hz) <= 800.0):
            return True, None
        return True, float(do_hz)


@dataclass
class RealtimeState:
    t: float
    hz: float | None
    y: float | None
    cent_y: float | None
    voiced: bool
    rms: float

    def to_event(self) -> dict[str, object]:
        return {
            "type": "pitch_update",
            "t": float(self.t),
            "hz": float(self.hz) if self.hz is not None else None,
            "y": float(self.y) if self.y is not None else None,
            "centY": float(self.cent_y) if self.cent_y is not None else None,
            "voiced": bool(self.voiced),
            "rms": float(self.rms),
        }


class RealtimeEngine:
    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        do_hz: float = 130.8,
        tuning: TuningSystem = TuningSystem.JUST_INTONATION,
        key_semitone: int = 0,
        ui: EngineUiConfig | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.ui = ui or EngineUiConfig()
        self.pitch = PitchTracker(PitchTrackerConfig(sample_rate=self.sample_rate))
        self.quantizer = JianpuQuantizer(
            do_hz=float(do_hz),
            tuning=tuning,
            key_semitone=int(key_semitone),
        )

        self._buffer: deque[tuple[float, float]] = deque()
        self._cent_buffer: deque[tuple[float, float]] = deque()

        self._y_smooth: deque[int] = deque(maxlen=self.ui.smooth_window)
        self._y_history: deque[int] = deque(maxlen=self.ui.smooth_window)
        self._cent_smooth: deque[float] = deque(maxlen=self.ui.cent_smooth_window)

        self._current_y: int | None = None
        self._candidate_y: int | None = None
        self._candidate_count = 0

        self._last_voiced_time: float | None = None
        self._last_output_nan = False
        self._cent_last_output_nan = False

        self._last_cent_value: float | None = None
        self._last_cent_time: float | None = None

        self._transition_start: float | None = None
        self._transition_target: int | None = None

        self._last_sent_y: float | None = None
        self._last_sent_cent: float | None = None

        self._calibration = CalibrationCollector(self.ui)

    def set_config(
        self,
        *,
        do_hz: float | None = None,
        tuning: TuningSystem | None = None,
        key_semitone: int | None = None,
    ) -> None:
        self.quantizer = JianpuQuantizer(
            do_hz=float(do_hz if do_hz is not None else self.quantizer.do_hz),
            tuning=tuning if tuning is not None else self.quantizer.tuning,
            key_semitone=int(
                key_semitone if key_semitone is not None else self.quantizer.key_semitone
            ),
        )

    def start_calibration(self, now: float, seconds: float = 4.0) -> None:
        self._calibration.start(now, seconds)

    def cancel_calibration(self) -> None:
        self._calibration.cancel()

    def process_frame(self, frame: np.ndarray, now: float) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if frame.size == 0:
            return events

        mono = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        pitch_hz = self.pitch.process(mono)

        done, calibrated_hz = self._calibration.push(
            now,
            rms,
            pitch_hz,
            silence_rms=self.pitch.silence_rms,
        )
        if done:
            if calibrated_hz is None:
                events.append(
                    {
                        "type": "error",
                        "code": "calibration_failed",
                        "message": "Calibration failed. Sing a steadier Do and try again.",
                    }
                )
            else:
                self.set_config(do_hz=calibrated_hz)
                events.append(
                    {
                        "type": "calibration_done",
                        "doHz": calibrated_hz,
                    }
                )

        observed_y: int | None = None
        if pitch_hz is not None:
            y = self.quantizer.quantize_to_y(pitch_hz)
            if y is not None:
                self._y_smooth.append(y)
                if len(self._y_smooth) >= 3:
                    y_sm = int(np.median(np.array(self._y_smooth, dtype=np.float32)))
                else:
                    y_sm = y
                observed_y = y_sm
                self._y_history.append(y_sm)
                self._last_voiced_time = now
            else:
                self._y_smooth.clear()
        else:
            self._y_smooth.clear()
            self._cent_smooth.clear()

        if observed_y is not None:
            if self._current_y is None:
                if self._transition_target != observed_y:
                    self._transition_start = now
                    self._transition_target = observed_y
            elif observed_y != self._current_y:
                if self._transition_target != observed_y:
                    self._transition_start = now
                    self._transition_target = observed_y
            else:
                self._transition_start = None
                self._transition_target = None

        cent_y: float | None = None
        if pitch_hz is not None:
            cent_y = self.quantizer.continuous_y(pitch_hz)
            if cent_y is not None:
                self._cent_smooth.append(cent_y)
                if len(self._cent_smooth) >= 3:
                    cent_y = float(np.median(np.array(self._cent_smooth, dtype=np.float32)))
                self._last_cent_value = cent_y
                self._last_cent_time = now
            else:
                self._cent_smooth.clear()

        output_y: float | None
        if self._last_voiced_time is not None and (now - self._last_voiced_time) <= self.ui.silence_timeout:
            if len(self._y_history) >= self.ui.stable_min_count:
                vals, counts = np.unique(np.array(self._y_history, dtype=np.int32), return_counts=True)
                stable_y = int(vals[np.argmax(counts)])
                stable_count = int(np.max(counts))
                if stable_count >= self.ui.stable_min_count:
                    if self._current_y is None:
                        self._current_y = stable_y
                        start_t = self._transition_start if self._transition_target == stable_y else None
                        if start_t is not None:
                            self._backfill_buffer(start_t, stable_y)
                    elif stable_y != self._current_y:
                        if self._candidate_y == stable_y:
                            self._candidate_count += 1
                        else:
                            self._candidate_y = stable_y
                            self._candidate_count = 1
                        if self._candidate_count >= self.ui.switch_confirm:
                            self._current_y = stable_y
                            self._candidate_y = None
                            self._candidate_count = 0
                            start_t = self._transition_start if self._transition_target == stable_y else None
                            if start_t is not None:
                                self._backfill_buffer(start_t, stable_y)
                            self._transition_start = None
                            self._transition_target = None
                    else:
                        self._candidate_y = None
                        self._candidate_count = 0
            output_y = float(self._current_y) if self._current_y is not None else None
        else:
            self._y_history.clear()
            self._current_y = None
            self._candidate_y = None
            self._candidate_count = 0
            output_y = None

        if output_y is None:
            if not self._last_output_nan:
                self._buffer.append((now, float("nan")))
                self._last_output_nan = True
        else:
            self._buffer.append((now, output_y))
            self._last_output_nan = False

        cent_output = cent_y
        if cent_output is None and self._last_cent_time is not None:
            if (now - self._last_cent_time) <= self.ui.silence_timeout:
                cent_output = self._last_cent_value

        if cent_output is None:
            if not self._cent_last_output_nan:
                self._cent_buffer.append((now, float("nan")))
                self._cent_last_output_nan = True
        else:
            self._cent_buffer.append((now, float(cent_output)))
            self._cent_last_output_nan = False

        release_before = now - self.ui.delay_seconds
        released_t: float | None = None
        while self._buffer and self._buffer[0][0] <= release_before:
            bt, by = self._buffer.popleft()
            released_t = bt
            self._last_sent_y = None if np.isnan(by) else float(by)

        while self._cent_buffer and self._cent_buffer[0][0] <= release_before:
            ct, cy = self._cent_buffer.popleft()
            released_t = max(released_t, ct) if released_t is not None else ct
            self._last_sent_cent = None if np.isnan(cy) else float(cy)

        if released_t is not None:
            state = RealtimeState(
                t=released_t,
                hz=pitch_hz,
                y=self._last_sent_y,
                cent_y=self._last_sent_cent,
                voiced=bool(pitch_hz is not None),
                rms=rms,
            )
            events.append(state.to_event())

        return events

    def _backfill_buffer(self, start_time: float, y_value: int) -> None:
        if not self._buffer:
            return
        updated = deque()
        for bt, by in self._buffer:
            if bt >= start_time:
                updated.append((bt, float(y_value)))
            else:
                updated.append((bt, by))
        self._buffer = updated


def estimate_recording_pitch(audio: np.ndarray, sample_rate: int, hop_size: int = 1024) -> float | None:
    tracker = PitchTracker(PitchTrackerConfig(sample_rate=sample_rate))
    hz_list: list[float] = []
    for i in range(0, len(audio), hop_size):
        frame = audio[i : i + hop_size]
        if frame.size == 0:
            continue
        hz = tracker.process(frame)
        if hz is not None:
            hz_list.append(hz)
    if not hz_list:
        return None
    return float(np.median(np.array(hz_list, dtype=np.float32)))


def shift_degree(degree: int, steps: int) -> tuple[int, int]:
    idx = (degree - 1) + steps
    octave_shift = idx // 7
    new_idx = idx % 7
    return int(new_idx + 1), int(octave_shift)


def compute_degree_shift_ratio(
    audio: np.ndarray,
    sample_rate: int,
    steps: int,
    *,
    do_hz: float,
    tuning: TuningSystem,
    key_semitone: int,
) -> float:
    if steps == 0:
        return 1.0

    quantizer = JianpuQuantizer(
        do_hz=float(do_hz),
        tuning=tuning,
        key_semitone=int(key_semitone),
    )
    base_hz = estimate_recording_pitch(audio, sample_rate)
    if base_hz is None or base_hz <= 0:
        return float(2.0 ** ((2 * steps) / 12.0))

    nearest = quantizer.nearest_degree(base_hz)
    if nearest is None:
        return float(2.0 ** ((2 * steps) / 12.0))

    degree, octave = nearest
    new_degree, octave_shift = shift_degree(degree, steps)
    target = quantizer.degree_hz(new_degree, octave=octave + octave_shift)
    if target is None:
        return float(2.0 ** ((2 * steps) / 12.0))
    return float(target / base_hz)


def parse_tuning(text: str) -> TuningSystem:
    try:
        return TuningSystem(text)
    except ValueError:
        return TuningSystem.JUST_INTONATION


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(x.astype(np.float32, copy=False))))))
