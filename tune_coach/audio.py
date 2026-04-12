from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable

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
        self._device: int | None = None
        self._sample_rate = int(self._cfg.sample_rate)
        self.refresh_default_input()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_size(self) -> int:
        return self._cfg.block_size

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def refresh_default_input(self) -> None:
        if self._stream is not None:
            return
        try:
            device = sd.query_devices(kind="input")
        except Exception:
            self._device = None
            self._sample_rate = int(self._cfg.sample_rate)
            return

        self._device = self._coerce_device_index(device)
        self._sample_rate = self._pick_sample_rate(device)

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

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                self._stream = sd.InputStream(
                    device=self._device,
                    samplerate=self._sample_rate,
                    channels=self._cfg.channels,
                    blocksize=self._cfg.block_size,
                    dtype="float32",
                    callback=callback,
                )
                self._stream.start()
                return
            except Exception as exc:
                last_exc = exc
                self._stream = None
                if attempt == 0:
                    self.refresh_default_input()
                    continue
                raise
        if last_exc is not None:
            raise last_exc

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

    def _pick_sample_rate(self, device: dict[str, object]) -> int:
        candidates: list[int] = []
        default_rate = device.get("default_samplerate")
        if default_rate is not None:
            try:
                rate = float(default_rate)
            except (TypeError, ValueError):
                rate = 0.0
            if math.isfinite(rate) and rate > 0:
                candidates.append(int(round(rate)))

        candidates.extend((int(self._cfg.sample_rate), 48_000, 44_100))

        seen: set[int] = set()
        for rate in candidates:
            if rate <= 0 or rate in seen:
                continue
            seen.add(rate)
            if self._supports_sample_rate(rate):
                return rate
        return int(self._cfg.sample_rate)

    def _supports_sample_rate(self, sample_rate: int) -> bool:
        try:
            sd.check_input_settings(
                device=self._device,
                samplerate=sample_rate,
                channels=self._cfg.channels,
                dtype="float32",
            )
        except Exception:
            return False
        return True

    def _coerce_device_index(self, device: dict[str, object]) -> int | None:
        index = device.get("index")
        if index is None:
            return None
        try:
            return int(index)
        except (TypeError, ValueError):
            return None
