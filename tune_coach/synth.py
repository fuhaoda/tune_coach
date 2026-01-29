from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sounddevice as sd


class Instrument(str, Enum):
    PIANO = "Piano"
    GUITAR = "Guitar"


@dataclass
class _NoteState:
    freq: float
    phase: float
    age: int
    releasing: bool
    release_remaining: int
    attack_samples: int
    release_samples: int
    decay_samples: int
    sustain_floor: float
    harmonics: tuple[float, ...]
    inharmonicity: float
    unison_detunes: tuple[float, ...]


class BodyResonator:
    def __init__(self, sample_rate: int, delay_ms: float, feedback: float, damp: float, mix: float) -> None:
        self._delay = max(1, int(sample_rate * delay_ms / 1000.0))
        self._buf = np.zeros(self._delay, dtype=np.float32)
        self._idx = 0
        self._feedback = float(feedback)
        self._damp = float(damp)
        self._default_mix = float(mix)
        self._lp = 0.0

    @property
    def default_mix(self) -> float:
        return self._default_mix

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        buf = self._buf
        idx = self._idx
        lp = self._lp
        fb = self._feedback
        damp = self._damp
        for i, xi in enumerate(x):
            delayed = buf[idx]
            lp = lp + damp * (delayed - lp)
            out = xi + fb * lp
            y[i] = out
            buf[idx] = out
            idx += 1
            if idx >= buf.size:
                idx = 0
        self._idx = idx
        self._lp = lp
        return y


class ChorusEffect:
    def __init__(
        self, sample_rate: int, delay_ms: float, depth_ms: float, rate_hz: float, mix: float
    ) -> None:
        self._base_delay = float(sample_rate * delay_ms / 1000.0)
        self._depth = float(sample_rate * depth_ms / 1000.0)
        size = int(self._base_delay + self._depth) + 3
        self._buf = np.zeros(size, dtype=np.float32)
        self._idx = 0
        self._phase = 0.0
        self._phase_inc = 2.0 * np.pi * float(rate_hz) / float(sample_rate)
        self._default_mix = float(mix)

    @property
    def default_mix(self) -> float:
        return self._default_mix

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        buf = self._buf
        size = buf.size
        idx = self._idx
        phase = self._phase
        inc = self._phase_inc
        base = self._base_delay
        depth = self._depth
        for i, xi in enumerate(x):
            buf[idx] = xi
            delay = base + depth * np.sin(phase)
            read_pos = idx - delay
            while read_pos < 0:
                read_pos += size
            i0 = int(read_pos)
            frac = read_pos - i0
            i1 = i0 + 1
            if i1 >= size:
                i1 -= size
            delayed = (1.0 - frac) * buf[i0] + frac * buf[i1]
            y[i] = delayed
            idx += 1
            if idx >= size:
                idx = 0
            phase += inc
            if phase >= 2.0 * np.pi:
                phase -= 2.0 * np.pi
        self._idx = idx
        self._phase = phase
        return y


class NoteSynth:
    def __init__(self, sample_rate: int = 44100, block_size: int = 1024) -> None:
        self._sample_rate = int(sample_rate)
        self._block_size = int(block_size)
        self._instrument = Instrument.PIANO
        self._notes: dict[str, _NoteState] = {}
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._base_gain = 0.12
        self._soft_clip_drive = 1.6
        self._body_piano = BodyResonator(
            sample_rate=self._sample_rate,
            delay_ms=28.0,
            feedback=0.12,
            damp=0.2,
            mix=0.18,
        )
        self._chorus_piano = ChorusEffect(
            sample_rate=self._sample_rate,
            delay_ms=18.0,
            depth_ms=3.0,
            rate_hz=0.3,
            mix=0.0,
        )
        self._body_guitar = BodyResonator(
            sample_rate=self._sample_rate,
            delay_ms=13.0,
            feedback=0.28,
            damp=0.45,
            mix=0.45,
        )
        self._piano_body_mix = self._body_piano.default_mix
        self._piano_chorus_mix = self._chorus_piano.default_mix
        self._guitar_body_mix = self._body_guitar.default_mix
        self._piano_body_state = 0.0
        self._piano_chorus_state = 0.0
        self._guitar_body_state = 0.0
        self._voice_seq = 0

    def _profile(self, instrument: Instrument) -> dict[str, object]:
        if instrument == Instrument.GUITAR:
            return {
                "harmonics": (1.0, 0.0, 0.55, 0.0, 0.32, 0.0, 0.18, 0.0, 0.12),
                "attack": 0.003,
                "release": 0.05,
                "decay": 0.55,
                "sustain": 0.25,
                "inharm": 0.0,
            }
        return {
            "harmonics": (1.0, 0.75, 0.5, 0.35, 0.22, 0.16, 0.12, 0.08),
            "attack": 0.01,
            "release": 0.08,
            "decay": 1.6,
            "sustain": 0.65,
            "inharm": 0.0,
        }

    def _piano_unison(self, freq: float) -> tuple[float, ...]:
        # Lower notes tend to have fewer strings; keep detune subtle to avoid beating.
        if freq < 180.0:
            cents = np.array([-0.6, 0.6], dtype=np.float32)
        else:
            cents = np.array([-0.9, 0.0, 0.9], dtype=np.float32)
        detunes = np.power(2.0, cents / 1200.0)
        return tuple(float(x) for x in detunes)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            blocksize=self._block_size,
            dtype="float32",
            callback=self._callback,
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

    def set_instrument(self, instrument: Instrument | str) -> None:
        try:
            self._instrument = Instrument(instrument)
        except ValueError:
            self._instrument = Instrument.PIANO

    def note_on(self, key: str, freq: float) -> None:
        if freq <= 0:
            return
        profile = self._profile(self._instrument)
        attack_samples = max(8, int(float(profile["attack"]) * self._sample_rate))
        release_samples = max(32, int(float(profile["release"]) * self._sample_rate))
        decay_samples = max(1, int(float(profile["decay"]) * self._sample_rate))
        sustain_floor = float(profile["sustain"])
        harmonics = tuple(float(x) for x in profile["harmonics"])
        inharm = float(profile["inharm"])
        if self._instrument == Instrument.PIANO:
            unison_detunes = self._piano_unison(freq)
        else:
            unison_detunes = (1.0,)
        with self._lock:
            if key in self._notes:
                prev = self._notes.pop(key)
                prev.releasing = True
                release_key = f"{key}#rel{self._voice_seq}"
                self._voice_seq += 1
                self._notes[release_key] = prev
            self._notes[key] = _NoteState(
                freq=float(freq),
                phase=0.0,
                age=0,
                releasing=False,
                release_remaining=release_samples,
                attack_samples=attack_samples,
                release_samples=release_samples,
                decay_samples=decay_samples,
                sustain_floor=sustain_floor,
                harmonics=harmonics,
                inharmonicity=inharm,
                unison_detunes=unison_detunes,
            )

    def note_off(self, key: str) -> None:
        with self._lock:
            state = self._notes.get(key)
            if state is None:
                return
            state.releasing = True

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        out = np.zeros(frames, dtype=np.float32)
        t = np.arange(frames, dtype=np.float32)

        with self._lock:
            notes_items = list(self._notes.items())

        if not notes_items:
            outdata[:] = out.reshape(-1, 1)
            return

        polyphonic = len(notes_items) >= 2
        close_interval = False
        if polyphonic and self._instrument == Instrument.PIANO and len(notes_items) >= 2:
            freqs = [state.freq for _, state in notes_items]
            for i in range(len(freqs)):
                for j in range(i + 1, len(freqs)):
                    semis = abs(12.0 * math.log2(freqs[i] / freqs[j]))
                    if semis <= 2.2:
                        close_interval = True
                        break
                if close_interval:
                    break
        for key, state in notes_items:
            omega = 2.0 * np.pi * state.freq / float(self._sample_rate)
            phase = state.phase + omega * t
            wave = np.zeros(frames, dtype=np.float32)
            inharm = float(state.inharmonicity)
            nyquist = 0.48 * float(self._sample_rate)
            detunes = state.unison_detunes
            unison_gain = 1.0 / float(len(detunes))
            if polyphonic and self._instrument == Instrument.PIANO:
                poly_rolloff = 0.78 if close_interval else 0.84
            else:
                poly_rolloff = 1.0
            for detune in detunes:
                d_omega = 2.0 * np.pi * (state.freq * detune) / float(self._sample_rate)
                d_phase = state.phase + d_omega * t
                for i, amp in enumerate(state.harmonics, start=1):
                    mult = float(i * (1.0 + inharm * i))
                    if state.freq * detune * mult >= nyquist:
                        break
                    roll = poly_rolloff ** (i - 1)
                    wave += unison_gain * amp * roll * np.sin(d_phase * mult)

            env = np.ones(frames, dtype=np.float32)
            if state.age < state.attack_samples:
                n = min(frames, state.attack_samples - state.age)
                env[:n] = np.linspace(
                    float(state.age) / float(state.attack_samples),
                    1.0,
                    n,
                    endpoint=False,
                    dtype=np.float32,
                )
            age_samples = state.age + np.arange(frames, dtype=np.float32)
            decay = state.sustain_floor + (1.0 - state.sustain_floor) * np.exp(
                -age_samples / float(state.decay_samples)
            )
            env *= decay

            remove_note = False
            if state.releasing:
                start = float(state.release_remaining) / float(state.release_samples)
                end = max(0.0, float(state.release_remaining - frames) / float(state.release_samples))
                ramp = np.linspace(start, end, frames, endpoint=False, dtype=np.float32)
                env *= ramp
                state.release_remaining -= frames
                if state.release_remaining <= 0:
                    remove_note = True

            wave *= env
            state.age += frames
            state.phase = float((phase[-1] + omega) % (2.0 * np.pi))
            out += wave

            if remove_note:
                with self._lock:
                    self._notes.pop(key, None)

        if polyphonic:
            if self._instrument == Instrument.GUITAR:
                wet = self._body_guitar.process(out)
                out = (1.0 - self._guitar_body_state) * out + self._guitar_body_state * wet
            else:
                wet_body = self._body_piano.process(out)
                out = (1.0 - self._piano_body_state) * out + self._piano_body_state * wet_body
                wet_ch = self._chorus_piano.process(out)
                out = (1.0 - self._piano_chorus_state) * out + self._piano_chorus_state * wet_ch
        else:
            # When mono, still advance effect states but mix toward 0 to avoid clicks.
            if self._instrument == Instrument.GUITAR:
                wet = self._body_guitar.process(out)
                out = (1.0 - self._guitar_body_state) * out + self._guitar_body_state * wet
            else:
                wet_body = self._body_piano.process(out)
                out = (1.0 - self._piano_body_state) * out + self._piano_body_state * wet_body
                wet_ch = self._chorus_piano.process(out)
                out = (1.0 - self._piano_chorus_state) * out + self._piano_chorus_state * wet_ch

        # Smoothly ramp effect mixes to avoid pops when polyphony changes.
        block_seconds = float(frames) / float(self._sample_rate)
        alpha = 1.0 - math.exp(-block_seconds / 0.06)
        if self._instrument == Instrument.GUITAR:
            target = self._guitar_body_mix if polyphonic else 0.0
            self._guitar_body_state += (target - self._guitar_body_state) * alpha
        else:
            target_body = self._piano_body_mix if polyphonic else 0.0
            target_chorus = self._piano_chorus_mix if polyphonic else 0.0
            self._piano_body_state += (target_body - self._piano_body_state) * alpha
            self._piano_chorus_state += (target_chorus - self._piano_chorus_state) * alpha

        out *= self._base_gain
        if self._soft_clip_drive > 0:
            drive = float(self._soft_clip_drive)
            out = np.tanh(out * drive) / np.tanh(drive)
        outdata[:] = out.reshape(-1, 1)
