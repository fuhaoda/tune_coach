from __future__ import annotations

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

    def _profile(self, instrument: Instrument) -> dict[str, object]:
        if instrument == Instrument.GUITAR:
            return {
                "harmonics": (1.0, 0.0, 0.55, 0.0, 0.32, 0.0, 0.18, 0.0, 0.12),
                "attack": 0.003,
                "release": 0.05,
                "decay": 0.55,
                "sustain": 0.25,
            }
        return {
            "harmonics": (1.0, 0.75, 0.5, 0.35, 0.22, 0.16, 0.12, 0.08),
            "attack": 0.01,
            "release": 0.08,
            "decay": 1.6,
            "sustain": 0.65,
        }

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
        with self._lock:
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

        for key, state in notes_items:
            omega = 2.0 * np.pi * state.freq / float(self._sample_rate)
            phase = state.phase + omega * t
            wave = np.zeros(frames, dtype=np.float32)
            for i, amp in enumerate(state.harmonics, start=1):
                wave += amp * np.sin(phase * i)

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

        out *= self._base_gain
        if self._soft_clip_drive > 0:
            drive = float(self._soft_clip_drive)
            out = np.tanh(out * drive) / np.tanh(drive)
        outdata[:] = out.reshape(-1, 1)
