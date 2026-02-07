from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tune_coach.jianpu import TuningSystem


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class InitMessage(_Model):
    type: Literal["init"]
    sample_rate: int = Field(alias="sampleRate", ge=8_000, le=192_000)
    tuning: str = Field(default=TuningSystem.JUST_INTONATION.value)
    key_semitone: int = Field(alias="keySemitone", default=0, ge=0, le=11)
    do_hz: float = Field(alias="doHz", default=130.8, gt=0.0, le=2_000.0)


class SetConfigMessage(_Model):
    type: Literal["set_config"]
    tuning: str | None = None
    key_semitone: int | None = Field(alias="keySemitone", default=None, ge=0, le=11)
    do_hz: float | None = Field(alias="doHz", default=None, gt=0.0, le=2_000.0)


class CalibrateStartMessage(_Model):
    type: Literal["calibrate_start"]
    seconds: float = Field(default=4.0, gt=0.5, le=10.0)


class CalibrateCancelMessage(_Model):
    type: Literal["calibrate_cancel"]


class TransportPingMessage(_Model):
    type: Literal["transport_ping"]
    client_ts: float = Field(alias="clientTs")


class StatusEvent(_Model):
    type: Literal["status"] = "status"
    message: str


class ErrorEvent(_Model):
    type: Literal["error"] = "error"
    code: str
    message: str


class CalibrationDoneEvent(_Model):
    type: Literal["calibration_done"] = "calibration_done"
    do_hz: float = Field(alias="doHz")


class PitchUpdateEvent(_Model):
    type: Literal["pitch_update"] = "pitch_update"
    t: float
    hz: float | None
    y: float | None
    cent_y: float | None = Field(alias="centY")
    voiced: bool
    rms: float


class TransportPongEvent(_Model):
    type: Literal["transport_pong"] = "transport_pong"
    client_ts: float = Field(alias="clientTs")
    server_ts: float = Field(alias="serverTs")
