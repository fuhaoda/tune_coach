from __future__ import annotations

import threading
from typing import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioInputConfig:
    sample_rate: int = 44100
    channels: int = 1
    block_size: int = 1024


class AudioInput:
    def __init__(self, config: AudioInputConfig | None = None) -> None:
        self._cfg = config or AudioInputConfig()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stream: sd.InputStream | None = None
        self._tap: Callable[[np.ndarray], None] | None = None

    @property
    def sample_rate(self) -> int:
        return self._cfg.sample_rate

    @property
    def block_size(self) -> int:
        return self._cfg.block_size

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if status:
                # Drop frames on over/underflow; keep UI responsive.
                return
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            with self._lock:
                self._latest = mono
                tap = self._tap
            if tap is not None:
                tap(mono)

        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            blocksize=self._cfg.block_size,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def read_latest(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def set_tap(self, tap: Callable[[np.ndarray], None] | None) -> None:
        with self._lock:
            self._tap = tap
