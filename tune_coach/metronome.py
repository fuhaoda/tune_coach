from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd


class Metronome:
    def __init__(self, sample_rate: int = 44100) -> None:
        self._sample_rate = int(sample_rate)
        self._bpm = 96
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def set_bpm(self, bpm: int) -> None:
        with self._lock:
            self._bpm = int(max(30, min(240, bpm)))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                interval = 60.0 / float(self._bpm)
            now = time.monotonic()
            if now >= next_t:
                self._play_click()
                next_t = now + interval
            time.sleep(0.002)

    def _play_click(self) -> None:
        # Short "click" more like common pop metronome than a sine beep.
        dur = 0.03
        n = int(self._sample_rate * dur)
        t = np.arange(n, dtype=np.float32) / float(self._sample_rate)
        freq = 1600.0
        wave = np.sign(np.sin(2.0 * np.pi * freq * t)).astype(np.float32) * 0.2
        env = np.exp(-t * 80.0).astype(np.float32)
        click = (wave * env).astype(np.float32)
        sd.play(click, samplerate=self._sample_rate, blocking=False)

