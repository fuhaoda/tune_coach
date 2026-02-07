from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import ValidationError

from tune_coach import __version__
from tune_coach.voice_shift import pitch_shift_formant
from tune_coach.web.realtime_engine import compute_degree_shift_ratio, parse_tuning
from tune_coach.web.schemas import (
    CalibrateCancelMessage,
    CalibrateStartMessage,
    InitMessage,
    SetConfigMessage,
    TransportPingMessage,
)
from tune_coach.web.session import SessionManager

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIST_DIR = ROOT_DIR / "webapp" / "dist"

app = FastAPI(title="Tune Coach Web", version=__version__)
sessions = SessionManager()


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "activeSessions": sessions.active_count,
        "frontendBuilt": WEB_DIST_DIR.exists(),
    }


@app.post("/api/voice/shift")
async def shift_voice(
    audio: UploadFile = File(...),
    steps: int = Form(0),
    do_hz: float = Form(130.8),
    tuning: str = Form("Just Intonation"),
    key_semitone: int = Form(0),
) -> Response:
    if steps < -5 or steps > 5:
        raise HTTPException(status_code=422, detail="steps must be in range [-5, 5]")

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        waveform, sample_rate = _decode_audio(payload, filename=audio.filename or "")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {exc}") from exc

    tuning_enum = parse_tuning(tuning)
    ratio = compute_degree_shift_ratio(
        waveform,
        sample_rate,
        steps,
        do_hz=do_hz,
        tuning=tuning_enum,
        key_semitone=key_semitone,
    )

    try:
        shifted = pitch_shift_formant(waveform, sample_rate, ratio)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Shift failed: {exc}") from exc

    wav_bytes = _encode_wav(shifted, sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = sessions.create()
    await websocket.send_json({"type": "status", "message": "Connected."})

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            binary = message.get("bytes")

            if text is not None:
                for event in _handle_text_message(session, text):
                    await websocket.send_json(event)
            elif binary is not None:
                for event in session.process_audio_bytes(binary):
                    await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        sessions.remove(session.session_id)


@app.get("/", include_in_schema=False)
async def serve_root() -> Response:
    index_path = WEB_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        "<h3>Tune Coach Web backend is running.</h3>"
        "<p>Build frontend with <code>npm install && npm run build</code> in <code>webapp/</code>.</p>"
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str) -> Response:
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        raise HTTPException(status_code=404, detail="Not found")

    if not WEB_DIST_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")

    target = WEB_DIST_DIR / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)

    index_path = WEB_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not built")


def _handle_text_message(session, text: str) -> list[dict[str, object]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [{"type": "error", "code": "invalid_json", "message": "Invalid JSON payload"}]

    if not isinstance(payload, dict):
        return [{"type": "error", "code": "invalid_payload", "message": "Expected JSON object"}]

    msg_type = payload.get("type")
    try:
        if msg_type == "init":
            msg = InitMessage.model_validate(payload)
            session.init(
                sample_rate=msg.sample_rate,
                tuning=msg.tuning,
                key_semitone=msg.key_semitone,
                do_hz=msg.do_hz,
            )
            return [{"type": "status", "message": "Session initialized."}]

        if msg_type == "set_config":
            msg = SetConfigMessage.model_validate(payload)
            session.set_config(tuning=msg.tuning, key_semitone=msg.key_semitone, do_hz=msg.do_hz)
            return [{"type": "status", "message": "Config updated."}]

        if msg_type == "calibrate_start":
            msg = CalibrateStartMessage.model_validate(payload)
            session.start_calibration(seconds=msg.seconds)
            return [{"type": "status", "message": "Calibration started."}]

        if msg_type == "calibrate_cancel":
            CalibrateCancelMessage.model_validate(payload)
            session.cancel_calibration()
            return [{"type": "status", "message": "Calibration canceled."}]

        if msg_type == "transport_ping":
            msg = TransportPingMessage.model_validate(payload)
            return [
                {
                    "type": "transport_pong",
                    "clientTs": msg.client_ts,
                    "serverTs": time.time(),
                }
            ]

    except ValidationError as exc:
        return [{"type": "error", "code": "invalid_message", "message": str(exc)}]

    return [{"type": "error", "code": "unknown_message", "message": f"Unknown type: {msg_type}"}]


def _decode_audio(payload: bytes, *, filename: str) -> tuple[np.ndarray, int]:
    try:
        return _decode_with_soundfile(payload)
    except Exception:  # noqa: BLE001
        if not shutil.which("ffmpeg"):
            raise
        suffix = Path(filename).suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="tune-coach-") as tmp_dir:
            input_path = Path(tmp_dir) / f"input{suffix}"
            output_path = Path(tmp_dir) / "decoded.wav"
            input_path.write_bytes(payload)
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return _decode_with_soundfile(output_path.read_bytes())


def _decode_with_soundfile(payload: bytes) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("decoded audio is empty")
    return audio, int(sample_rate)


def _encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    data = np.asarray(waveform, dtype=np.float32)
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> None:
    uvicorn.run(
        "tune_coach.web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
