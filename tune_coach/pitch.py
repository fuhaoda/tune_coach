from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchTrackerConfig:
    sample_rate: int = 44100
    window_size: int = 4096
    hop_size: int = 1024
    min_hz: float = 80.0
    max_hz: float = 1000.0
    silence_rms: float = 0.015  # ~-36 dBFS for float32 [-1,1]
    confidence_min: float = 0.35
    median_window: int = 7
    ema_alpha: float = 0.25


class PitchTracker:
    """
    Lightweight real-time pitch tracker (no aubio).

    Strategy:
    - Gate by RMS (noise threshold).
    - FFT-based autocorrelation on a Hann-windowed frame.
    - Peak picking within [sr/max_hz, sr/min_hz].
    - Confidence from normalized autocorr peak.
    - Median smoothing across last few frames.
    """

    def __init__(self, config: PitchTrackerConfig) -> None:
        self._cfg = config
        self._recent: list[float] = []
        self._ema: float | None = None
        self._buffer = np.zeros(self._cfg.window_size, dtype=np.float32)
        self._buf_fill = 0
        self._hann = np.hanning(self._cfg.window_size).astype(np.float32)

    def process(self, mono_block: np.ndarray) -> float | None:
        if mono_block.size == 0:
            return None

        mono = mono_block.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        if rms < self._cfg.silence_rms:
            self._recent.clear()
            self._buf_fill = 0
            self._ema = None
            return None

        # Push samples into a fixed-size window for FFT autocorrelation.
        self._push(mono)
        if self._buf_fill < self._cfg.window_size:
            return None

        # Estimate pitch + confidence from autocorrelation peak.
        pitch_hz, conf = _autocorr_pitch(self._buffer, self._cfg.sample_rate, self._cfg.min_hz, self._cfg.max_hz)
        if pitch_hz is None or conf < self._cfg.confidence_min:
            self._recent.clear()
            self._ema = None
            return None

        self._recent.append(pitch_hz)
        if len(self._recent) > self._cfg.median_window:
            self._recent.pop(0)
        # Median reduces spiky errors; EMA adds extra smoothing.
        median = float(np.median(np.array(self._recent, dtype=np.float32)))
        if self._ema is None:
            self._ema = median
        else:
            alpha = float(self._cfg.ema_alpha)
            self._ema = alpha * median + (1.0 - alpha) * self._ema
        return float(self._ema)

    def _push(self, x: np.ndarray) -> None:
        # Keep only the last window_size samples by shifting a hop at a time.
        n = int(x.size)
        if n >= self._cfg.window_size:
            self._buffer[:] = x[-self._cfg.window_size :].astype(np.float32, copy=False)
            self._buf_fill = self._cfg.window_size
            return

        if self._buf_fill < self._cfg.window_size:
            take = min(self._cfg.window_size - self._buf_fill, n)
            self._buffer[self._buf_fill : self._buf_fill + take] = x[-take:]
            self._buf_fill += take
            return

        hop = min(self._cfg.hop_size, n)
        self._buffer[:-hop] = self._buffer[hop:]
        self._buffer[-hop:] = x[-hop:]


def _autocorr_pitch(
    frame: np.ndarray, sample_rate: int, min_hz: float, max_hz: float
) -> tuple[float | None, float]:
    x = frame.astype(np.float32, copy=False)
    x = x - float(np.mean(x))
    x = x * np.hanning(x.size).astype(np.float32)

    # FFT autocorrelation: r = irfft(|fft(x)|^2)
    n = int(x.size)
    spec = np.fft.rfft(x, n=n)
    power = spec * np.conj(spec)
    r = np.fft.irfft(power, n=n).astype(np.float32)

    # Normalize autocorr
    r0 = float(r[0])
    if r0 <= 1e-6:
        return None, 0.0
    r /= r0

    min_lag = int(sample_rate / float(max_hz))
    max_lag = int(sample_rate / float(min_hz))
    min_lag = max(2, min(min_lag, n - 2))
    max_lag = max(min_lag + 1, min(max_lag, n - 2))

    seg = r[min_lag:max_lag]
    if seg.size < 3:
        return None, 0.0

    # Pick the maximum peak in the allowed lag range.
    i = int(np.argmax(seg)) + min_lag
    peak = float(r[i])

    # Parabolic interpolation around the peak for sub-sample accuracy.
    i0, i1, i2 = i - 1, i, i + 1
    y0, y1, y2 = float(r[i0]), float(r[i1]), float(r[i2])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) > 1e-8:
        delta = 0.5 * (y0 - y2) / denom
        lag = float(i1) + float(delta)
    else:
        lag = float(i1)

    if lag <= 0.0:
        return None, 0.0

    hz = float(sample_rate) / lag
    if not (min_hz <= hz <= max_hz):
        return None, float(max(0.0, min(1.0, peak)))

    # A simple confidence: peak height, clipped to [0,1].
    conf = float(max(0.0, min(1.0, peak)))
    # Penalize extreme octave errors by preferring stronger early peaks.
    conf = conf * _octave_sanity_bonus(r, min_lag, i)
    return hz, conf


def _octave_sanity_bonus(r: np.ndarray, min_lag: int, peak_lag: int) -> float:
    # If half-lag also has a strong peak, we may have picked an octave error.
    half = peak_lag // 2
    if half <= min_lag:
        return 1.0
    half_peak = float(r[half])
    peak = float(r[peak_lag])
    if peak <= 1e-6:
        return 0.0
    ratio = half_peak / peak
    # If half is close to peak, reduce confidence.
    return float(1.0 - min(0.5, max(0.0, ratio - 0.55)))
