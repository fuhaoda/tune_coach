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
    def fake_query_devices(kind="input"):
        if kind is None:
            return [
                {"index": 7, "default_samplerate": 44_100.0, "max_input_channels": 1},
            ]
        return {"index": 7, "default_samplerate": 44_100.0}

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
    monkeypatch.setattr(audio_mod.time, "sleep", lambda _: None)

    audio = audio_mod.AudioInput()
    audio.start()

    assert len(created) == 2
    assert created[0]["device"] == 7
    assert created[0]["samplerate"] == 44_100
    assert created[1]["device"] == 7
    assert created[1]["samplerate"] == 48_000
    assert audio.sample_rate == 48_000
    assert audio.is_running

    audio.stop()


def test_audio_input_falls_back_to_other_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    input_devices = [
        {
            "name": "USB Mic",
            "index": 7,
            "default_samplerate": 48_000.0,
            "max_input_channels": 2,
        },
        {
            "name": "MacBook Pro Microphone",
            "index": 9,
            "default_samplerate": 44_100.0,
            "max_input_channels": 1,
        },
    ]

    created: list[dict[str, object]] = []

    class DummyStream:
        def __init__(self, params: dict[str, object]) -> None:
            self._params = params

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_query_devices(kind=None):
        if kind == "input":
            return input_devices[0]
        return input_devices

    def fake_input_stream(**kwargs):
        created.append(kwargs)
        if kwargs["device"] == 7:
            raise audio_mod.sd.PortAudioError("invalid property", -9986)
        return DummyStream(kwargs)

    monkeypatch.setattr(audio_mod.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(audio_mod.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio_mod.sd, "InputStream", fake_input_stream)
    monkeypatch.setattr(audio_mod.time, "sleep", lambda _: None)

    audio = audio_mod.AudioInput()
    audio.start()

    assert created[0]["device"] == 7
    assert created[-1]["device"] == 9
    assert created[-1]["samplerate"] == 44_100
    assert audio.sample_rate == 44_100
    assert audio.is_running


def test_audio_input_reinitializes_portaudio_after_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    resets: list[str] = []
    recovered = {"ready": False}

    class DummyStream:
        def __init__(self, params: dict[str, object]) -> None:
            self._params = params

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    class DummyDefault:
        def reset(self) -> None:
            resets.append("default.reset")

    def fake_query_devices(kind="input"):
        if kind is None:
            return [
                {"index": 9, "default_samplerate": 44_100.0, "max_input_channels": 1},
            ]
        return {"index": 9, "default_samplerate": 44_100.0}

    def fake_input_stream(**kwargs):
        created.append(kwargs)
        if not recovered["ready"]:
            raise audio_mod.sd.PortAudioError(
                "boom",
                -9986,
                (0, -10851, "Audio Unit: Invalid Property Value"),
            )
        return DummyStream(kwargs)

    def fake_terminate() -> None:
        resets.append("_terminate")
        monkeypatch.setattr(audio_mod.sd, "_initialized", 0)

    def fake_initialize() -> None:
        resets.append("_initialize")
        monkeypatch.setattr(audio_mod.sd, "_initialized", 1)
        recovered["ready"] = True

    monkeypatch.setattr(audio_mod.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(audio_mod.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio_mod.sd, "InputStream", fake_input_stream)
    monkeypatch.setattr(audio_mod.sd, "stop", lambda: resets.append("stop"))
    monkeypatch.setattr(audio_mod.sd, "_terminate", fake_terminate)
    monkeypatch.setattr(audio_mod.sd, "_initialize", fake_initialize)
    monkeypatch.setattr(audio_mod.sd, "_initialized", 1)
    monkeypatch.setattr(audio_mod.sd, "default", DummyDefault())
    monkeypatch.setattr(audio_mod.time, "sleep", lambda _: None)

    audio = audio_mod.AudioInput()
    audio.start()

    assert "_terminate" in resets
    assert "_initialize" in resets
    assert "default.reset" in resets
    assert created[-1]["device"] == 9
    assert created[-1]["samplerate"] == 44_100
    assert audio.is_running
