from __future__ import annotations

import math
import threading
import time
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
    _PORTAUDIO_RESET_RETRY_DELAYS = (0.2, 0.8)
    _RECOVERABLE_PA_ERROR_CODES = frozenset({-9986, -9985, -9999})
    _RECOVERABLE_COREAUDIO_ERROR_CODES = frozenset({-10851})

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

    def refresh_default_input(self, *, force: bool = False) -> None:
        if self._stream is not None and not force:
            return
        device = self._resolve_default_input_device()
        if device is None:
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
        with self._lock:
            self._latest = None

        last_exc = self._open_input_stream(callback)
        if last_exc is None:
            return

        if not self._should_reset_portaudio(last_exc):
            raise last_exc

        for retry_delay in self._PORTAUDIO_RESET_RETRY_DELAYS:
            if retry_delay > 0:
                time.sleep(retry_delay)
            self._reset_portaudio()
            last_exc = self._open_input_stream(callback)
            if last_exc is None:
                return
            if not self._should_reset_portaudio(last_exc):
                break
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
            with self._lock:
                self._latest = None

    def read_latest(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def set_tap(self, tap: Callable[[np.ndarray], None] | None) -> None:
        with self._lock:
            self._tap = tap

    def _open_input_stream(self, callback: Callable[..., None]) -> Exception | None:
        last_exc: Exception | None = None
        self.refresh_default_input(force=True)
        for device, sample_rate in self._iter_open_settings():
            try:
                self._stream = sd.InputStream(
                    device=device,
                    samplerate=sample_rate,
                    channels=self._cfg.channels,
                    blocksize=self._cfg.block_size,
                    dtype="float32",
                    callback=callback,
                )
                self._stream.start()
                self._device = device
                self._sample_rate = sample_rate
                return None
            except Exception as exc:
                last_exc = exc
                self._stream = None
        return last_exc

    def _should_reset_portaudio(self, exc: Exception) -> bool:
        if not isinstance(exc, sd.PortAudioError):
            return False
        pa_error_code = exc.args[1] if len(exc.args) > 1 else None
        if pa_error_code in self._RECOVERABLE_PA_ERROR_CODES:
            return True
        if len(exc.args) > 2:
            host_error = exc.args[2]
            if (
                isinstance(host_error, tuple)
                and len(host_error) >= 2
                and host_error[1] in self._RECOVERABLE_COREAUDIO_ERROR_CODES
            ):
                return True
        return "internal portaudio error" in str(exc).lower()

    def _reset_portaudio(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass

        terminate = getattr(sd, "_terminate", None)
        initialize = getattr(sd, "_initialize", None)
        initialized = getattr(sd, "_initialized", 0)
        if callable(terminate):
            try:
                while isinstance(initialized, int) and initialized > 0:
                    terminate()
                    initialized = getattr(sd, "_initialized", 0)
            except Exception:
                pass
        if callable(initialize):
            try:
                initialize()
            except Exception:
                pass

        default_reset = getattr(getattr(sd, "default", None), "reset", None)
        if callable(default_reset):
            try:
                default_reset()
            except Exception:
                pass

    def _pick_sample_rate(self, device: dict[str, object]) -> int:
        for rate in self._sample_rate_candidates(device):
            if self._supports_sample_rate(self._coerce_device_index(device), rate):
                return rate
        return int(self._cfg.sample_rate)

    def _sample_rate_candidates(self, device: dict[str, object] | None) -> list[int]:
        candidates: list[int] = []
        if device is not None:
            default_rate = device.get("default_samplerate")
            if default_rate is not None:
                try:
                    rate = float(default_rate)
                except (TypeError, ValueError):
                    rate = 0.0
                if math.isfinite(rate) and rate > 0:
                    candidates.append(int(round(rate)))

        candidates.extend((self._sample_rate, int(self._cfg.sample_rate), 48_000, 44_100))

        seen: set[int] = set()
        supported: list[int] = []
        for rate in candidates:
            if rate <= 0 or rate in seen:
                continue
            seen.add(rate)
            supported.append(rate)
        return supported

    def _supports_sample_rate(self, device: int | None, sample_rate: int) -> bool:
        try:
            sd.check_input_settings(
                device=device,
                samplerate=sample_rate,
                channels=self._cfg.channels,
                dtype="float32",
            )
        except Exception:
            return False
        return True

    def _iter_open_settings(self) -> list[tuple[int | None, int]]:
        settings: list[tuple[int | None, int]] = []
        seen: set[tuple[int | None, int]] = set()
        for device_info in self._candidate_input_devices():
            device = self._coerce_device_index(device_info)
            for sample_rate in self._sample_rate_candidates(device_info):
                if not self._supports_sample_rate(device, sample_rate):
                    continue
                key = (device, sample_rate)
                if key in seen:
                    continue
                seen.add(key)
                settings.append(key)
        if settings:
            return settings
        return [(self._device, self._sample_rate)]

    def _candidate_input_devices(self) -> list[dict[str, object] | None]:
        candidates: list[dict[str, object] | None] = []
        if self._device is not None:
            candidates.append(
                {
                    "index": self._device,
                    "default_samplerate": float(self._sample_rate),
                }
            )
        default_device = self._resolve_default_input_device()
        if default_device is not None:
            candidates.append(default_device)
        candidates.extend(self._list_input_devices())
        if not candidates:
            return [None]

        deduped: list[dict[str, object] | None] = []
        seen_devices: set[int | None] = set()
        for device in candidates:
            device_index = self._coerce_device_index(device)
            if device_index in seen_devices:
                continue
            seen_devices.add(device_index)
            deduped.append(device)
        return deduped

    def _resolve_default_input_device(self) -> dict[str, object] | None:
        try:
            device = sd.query_devices(kind="input")
        except Exception:
            device = None
        if isinstance(device, dict):
            return device
        devices = self._list_input_devices()
        if devices:
            return devices[0]
        return None

    def _list_input_devices(self) -> list[dict[str, object]]:
        try:
            devices = sd.query_devices()
        except Exception:
            return []
        inputs: list[dict[str, object]] = []
        for device in devices:
            if not isinstance(device, dict):
                continue
            max_input_channels = device.get("max_input_channels")
            try:
                channel_count = int(max_input_channels)
            except (TypeError, ValueError):
                channel_count = 0
            if channel_count > 0:
                inputs.append(device)
        return inputs

    def _coerce_device_index(self, device: dict[str, object] | None) -> int | None:
        if device is None:
            return None
        index = device.get("index")
        if index is None:
            return None
        try:
            return int(index)
        except (TypeError, ValueError):
            return None
