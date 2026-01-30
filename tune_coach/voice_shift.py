from __future__ import annotations

import numpy as np

try:
    import pyworld
except Exception:  # pragma: no cover - optional dependency
    pyworld = None

try:
    import librosa
except Exception:  # pragma: no cover - optional dependency
    librosa = None


def pitch_shift_formant(
    y: np.ndarray,
    sample_rate: int,
    ratio: float,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    lifter: int = 28,
) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or ratio == 1.0:
        return y.copy()

    y_trim, pad_left, pad_right = _trim_silence(y, sample_rate)

    # Best quality: WORLD vocoder (if available).
    if pyworld is not None:
        shifted = _pitch_shift_world(y_trim, sample_rate, ratio)
        if (shifted is None or not shifted.size) and y_trim.size != y.size:
            shifted = _pitch_shift_world(y, sample_rate, ratio)
            pad_left = 0
            pad_right = 0
        if shifted is not None and shifted.size:
            shifted = _pad_to_length(shifted, pad_left, pad_right, len(y))
            return shifted.astype(np.float32, copy=False)

    # Pitch shift by higher-quality library if available, else phase vocoder + resample.
    shifted_base: np.ndarray
    if librosa is not None:
        try:
            n_steps = 12.0 * float(np.log2(ratio))
            n_fft_lib = max(2048, int(n_fft))
            hop_length = max(256, n_fft_lib // 4)
            shifted_base = librosa.effects.pitch_shift(
                y,
                sr=sample_rate,
                n_steps=n_steps,
                res_type="soxr_hq",
                n_fft=n_fft_lib,
                hop_length=hop_length,
                win_length=n_fft_lib,
                scale=True,
            ).astype(np.float32, copy=False)
            if shifted_base.size != y.size:
                shifted_base = _resample_linear(shifted_base, len(y))
        except Exception:
            shifted_base = _pitch_shift_vocoder(y, ratio, n_fft=n_fft, hop=hop)
    else:
        shifted_base = _pitch_shift_vocoder(y, ratio, n_fft=n_fft, hop=hop)

    # Formant correction by matching global spectral envelope to original.
    shifted = _formant_correct_global(shifted_base, y, n_fft=n_fft, hop=hop, lifter=lifter)
    rms_base = float(np.sqrt(np.mean(np.square(shifted_base))))
    rms_shift = float(np.sqrt(np.mean(np.square(shifted))))
    if rms_base > 1e-6 and rms_shift < 0.15 * rms_base:
        shifted = shifted_base.copy()

    # Avoid clipping.
    peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if peak > 0.99:
        shifted = shifted / peak * 0.99
    return shifted.astype(np.float32, copy=False)


def _pitch_shift_world(y: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray | None:
    if pyworld is None:
        return None
    if ratio <= 0:
        return None
    original_len = len(y)
    original_sr = int(sample_rate)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return None
    if np.max(np.abs(y)) < 1e-4:
        return None
    sr = original_sr
    if sr > 24000:
        target_sr = 22050
        y = _resample_linear(y.astype(np.float32), int(len(y) * target_sr / sr)).astype(np.float64)
        sr = target_sr
    # Ensure minimum length for stable F0 extraction.
    min_len = int(0.4 * sr)
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size), mode="constant")
    frame_period = 5.0
    f0, t = _extract_f0(y, sr, frame_period)
    voiced = f0 > 0.0
    frame_period_sec = frame_period / 1000.0
    if np.count_nonzero(voiced) * frame_period_sec < 0.1:
        return None
    idx = np.where(voiced)[0]
    if idx.size == 0:
        return None
    f0_work = f0.copy()
    f0_work[~voiced] = np.interp(np.where(~voiced)[0], idx, f0[idx])
    f0_work = _median_filter_1d(f0_work, 5)
    sp = pyworld.cheaptrick(y, f0, t, sr)
    ap = pyworld.d4c(y, f0, t, sr)
    f0_shift = np.zeros_like(f0_work)
    f0_shift[voiced] = f0_work[voiced] * float(ratio)
    y_shift = pyworld.synthesize(f0_shift, sp, ap, sr)
    if sr != original_sr:
        y_shift = _resample_linear(y_shift.astype(np.float32), original_len)
    if y_shift.size != original_len:
        y_shift = _resample_linear(y_shift.astype(np.float32), original_len)
    return y_shift.astype(np.float32, copy=False)


def _trim_silence(
    y: np.ndarray,
    sample_rate: int,
    *,
    frame: int = 1024,
    hop: int = 256,
    pad_ms: float = 60.0,
) -> tuple[np.ndarray, int, int]:
    y = np.asarray(y, dtype=np.float32)
    if y.size < frame:
        return y, 0, 0
    frame = int(frame)
    hop = int(hop)
    if hop <= 0 or frame <= hop:
        return y, 0, 0
    n_frames = 1 + (len(y) - frame) // hop
    if n_frames <= 0:
        return y, 0, 0
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame),
        strides=(y.strides[0] * hop, y.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    max_rms = float(np.max(rms))
    if max_rms < 1e-4:
        return y, 0, 0
    thresh = max(0.12 * max_rms, 1e-4)
    idx = np.where(rms >= thresh)[0]
    if idx.size == 0:
        return y, 0, 0
    pad = int((pad_ms / 1000.0) * sample_rate)
    start = max(0, int(idx[0] * hop) - pad)
    end = min(len(y), int(idx[-1] * hop + frame + pad))
    return y[start:end], start, len(y) - end


def _pad_to_length(y: np.ndarray, pad_left: int, pad_right: int, length: int) -> np.ndarray:
    if pad_left > 0 or pad_right > 0:
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    if y.size != length:
        y = _resample_linear(y.astype(np.float32, copy=False), length)
    return y


def _extract_f0(y: np.ndarray, sr: int, frame_period: float) -> tuple[np.ndarray, np.ndarray]:
    _f0, t = pyworld.harvest(y, sr, f0_floor=50.0, f0_ceil=800.0, frame_period=frame_period)
    f0 = pyworld.stonemask(y, _f0, t, sr)
    frame_period_sec = frame_period / 1000.0
    if np.count_nonzero(f0 > 0.0) * frame_period_sec < 0.1:
        _f0, t = pyworld.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, frame_period=frame_period)
        f0 = pyworld.stonemask(y, _f0, t, sr)
    return f0, t


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(xp[pad:-pad])
    for i in range(out.size):
        out[i] = float(np.median(xp[i : i + k]))
    return out


def _pitch_shift_vocoder(y: np.ndarray, ratio: float, *, n_fft: int, hop: int) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y.copy()
    n_fft = int(max(512, n_fft))
    hop = int(max(128, hop))
    if hop >= n_fft:
        hop = n_fft // 4
    spec = _stft(y, n_fft=n_fft, hop=hop)
    stretched = _phase_vocoder(spec, rate=1.0 / ratio, hop=hop)
    y_stretch = _istft(stretched, hop=hop, length=int(len(y) / ratio))
    return _resample_linear(y_stretch, len(y))


def _resample_linear(y: np.ndarray, n_out: int) -> np.ndarray:
    if y.size == 0 or n_out <= 0:
        return np.zeros(max(0, n_out), dtype=np.float32)
    x = np.linspace(0.0, 1.0, num=y.size, endpoint=False)
    xi = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xi, x, y).astype(np.float32)


def _stft(y: np.ndarray, *, n_fft: int, hop: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size), mode="constant")
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(y.strides[0] * hop, y.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1)
    return spec.T


def _istft(spec: np.ndarray, *, hop: int, length: int | None) -> np.ndarray:
    n_fft = (spec.shape[0] - 1) * 2
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = spec.shape[1]
    y_len = n_fft + hop * (n_frames - 1)
    y = np.zeros(y_len, dtype=np.float32)
    wsum = np.zeros(y_len, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(spec[:, i], n=n_fft).astype(np.float32)
        start = i * hop
        y[start : start + n_fft] += frame * window
        wsum[start : start + n_fft] += window**2
    nonzero = wsum > 1e-6
    y[nonzero] /= wsum[nonzero]
    if length is not None:
        if y.size >= length:
            return y[:length]
        return np.pad(y, (0, length - y.size), mode="constant")
    return y


def _phase_vocoder(spec: np.ndarray, rate: float, *, hop: int) -> np.ndarray:
    if rate <= 0:
        raise ValueError("rate must be positive")
    n_bins, n_frames = spec.shape
    time_steps = np.arange(0, n_frames, rate, dtype=np.float32)
    n_fft = (n_bins - 1) * 2
    phase_adv = (2.0 * np.pi * hop * np.arange(n_bins) / float(n_fft)).astype(np.float32)
    phase_acc = np.angle(spec[:, 0])
    out = np.empty((n_bins, len(time_steps)), dtype=np.complex64)

    for i, step in enumerate(time_steps):
        idx = int(np.floor(step))
        frac = step - idx
        if idx + 1 >= n_frames:
            spec1 = spec[:, -1]
            spec2 = spec1
        else:
            spec1 = spec[:, idx]
            spec2 = spec[:, idx + 1]
        mag = (1.0 - frac) * np.abs(spec1) + frac * np.abs(spec2)
        phase = np.angle(spec2) - np.angle(spec1)
        phase = phase - phase_adv
        phase = phase - 2.0 * np.pi * np.round(phase / (2.0 * np.pi))
        phase_acc += phase_adv + phase
        out[:, i] = mag * np.exp(1.0j * phase_acc)
    return out


def _spectral_envelope(mag: np.ndarray, lifter: int) -> np.ndarray:
    log_mag = np.log(np.maximum(mag, 1e-6))
    cep = np.fft.irfft(log_mag)
    cep[lifter:] = 0.0
    env = np.exp(np.real(np.fft.rfft(cep)))
    return env


def _formant_correct_global(
    shifted: np.ndarray, original: np.ndarray, *, n_fft: int, hop: int, lifter: int
) -> np.ndarray:
    spec_s = _stft(shifted, n_fft=n_fft, hop=hop)
    spec_o = _stft(original, n_fft=n_fft, hop=hop)
    n_frames = min(spec_s.shape[1], spec_o.shape[1])
    spec_s = spec_s[:, :n_frames]
    spec_o = spec_o[:, :n_frames]

    mag_s = np.abs(spec_s)
    mag_o = np.abs(spec_o)
    env_s = _spectral_envelope(np.median(mag_s, axis=1), lifter)
    env_o = _spectral_envelope(np.median(mag_o, axis=1), lifter)
    gain = (env_o / np.maximum(env_s, 1e-6)).astype(np.float32)
    mag_s = mag_s * gain[:, None]
    spec_s = mag_s * np.exp(1.0j * np.angle(spec_s))
    return _istft(spec_s, hop=hop, length=len(shifted))
