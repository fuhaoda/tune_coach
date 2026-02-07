from __future__ import annotations

import threading
import uuid

import numpy as np

from tune_coach.jianpu import TuningSystem
from tune_coach.web.realtime_engine import RealtimeEngine, parse_tuning


class RealtimeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.engine = RealtimeEngine()
        self._input_sample_rate = self.engine.sample_rate
        self._processing_buffer = np.zeros(0, dtype=np.float32)
        self._clock = 0.0
        self._block_size = 1024

    def init(
        self,
        *,
        sample_rate: int,
        tuning: str,
        key_semitone: int,
        do_hz: float,
    ) -> None:
        self._input_sample_rate = int(sample_rate)
        self.engine.set_config(
            do_hz=float(do_hz),
            tuning=parse_tuning(tuning),
            key_semitone=int(key_semitone),
        )

    def set_config(
        self,
        *,
        tuning: str | None = None,
        key_semitone: int | None = None,
        do_hz: float | None = None,
    ) -> None:
        tuning_enum: TuningSystem | None = None
        if tuning is not None:
            tuning_enum = parse_tuning(tuning)
        self.engine.set_config(do_hz=do_hz, tuning=tuning_enum, key_semitone=key_semitone)

    def start_calibration(self, seconds: float = 4.0) -> None:
        self.engine.start_calibration(self._clock, seconds)

    def cancel_calibration(self) -> None:
        self.engine.cancel_calibration()

    def process_audio_bytes(self, payload: bytes) -> list[dict[str, object]]:
        if not payload:
            return []

        frame = np.frombuffer(payload, dtype=np.float32)
        if frame.size == 0:
            return []

        if self._input_sample_rate != self.engine.sample_rate:
            frame = self._resample_chunk(frame, self._input_sample_rate, self.engine.sample_rate)
            if frame.size == 0:
                return []

        self._processing_buffer = np.concatenate((self._processing_buffer, frame))
        events: list[dict[str, object]] = []

        while self._processing_buffer.size >= self._block_size:
            block = self._processing_buffer[: self._block_size]
            self._processing_buffer = self._processing_buffer[self._block_size :]
            self._clock += self._block_size / float(self.engine.sample_rate)
            events.extend(self.engine.process_frame(block, self._clock))

        return events

    def _resample_chunk(self, frame: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
        if frame.size == 0 or in_sr <= 0 or out_sr <= 0:
            return np.zeros(0, dtype=np.float32)
        if in_sr == out_sr:
            return frame

        n_out = int(round(frame.size * (float(out_sr) / float(in_sr))))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)

        x = np.linspace(0.0, 1.0, num=frame.size, endpoint=False)
        xi = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(xi, x, frame).astype(np.float32)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, RealtimeSession] = {}
        self._lock = threading.Lock()

    def create(self) -> RealtimeSession:
        session_id = uuid.uuid4().hex
        session = RealtimeSession(session_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
