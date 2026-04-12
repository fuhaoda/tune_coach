from __future__ import annotations

import pytest

from tune_coach import audio as audio_mod


def test_audio_input_uses_default_input_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_mod.sd,
        "query_devices",
        lambda kind="input": {"index": 7, "default_samplerate": 48_000.0},
    )
    monkeypatch.setattr(audio_mod.sd, "check_input_settings", lambda **kwargs: None)

    audio = audio_mod.AudioInput()

    assert audio.sample_rate == 48_000


def test_audio_input_retries_with_refreshed_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = iter(
        (
            {"index": 7, "default_samplerate": 44_100.0},
            {"index": 9, "default_samplerate": 48_000.0},
        )
    )

    def fake_query_devices(kind="input"):
        try:
            return next(devices)
        except StopIteration:
            return {"index": 9, "default_samplerate": 48_000.0}

    created: list[dict[str, object]] = []

    class DummyStream:
        def __init__(self, params: dict[str, object]) -> None:
            self._params = params
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    def fake_input_stream(**kwargs):
        created.append(kwargs)
        if len(created) == 1:
            raise audio_mod.sd.PortAudioError("boom", -9986)
        return DummyStream(kwargs)

    monkeypatch.setattr(audio_mod.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(audio_mod.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio_mod.sd, "InputStream", fake_input_stream)

    audio = audio_mod.AudioInput()
    audio.start()

    assert audio.sample_rate == 48_000
    assert len(created) == 2
    assert created[0]["device"] == 7
    assert created[0]["samplerate"] == 44_100
    assert created[1]["device"] == 9
    assert created[1]["samplerate"] == 48_000
    assert audio.is_running

    audio.stop()
