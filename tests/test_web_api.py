from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from tune_coach.web.server import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get('/api/health')
    assert resp.status_code == 200
    payload = resp.json()
    assert payload['status'] == 'ok'
    assert 'activeSessions' in payload


def test_realtime_websocket_flow() -> None:
    client = TestClient(app)

    with client.websocket_connect('/ws/realtime') as ws:
        first = ws.receive_json()
        assert first['type'] == 'status'

        ws.send_json(
            {
                'type': 'init',
                'sampleRate': 44100,
                'tuning': 'Just Intonation',
                'keySemitone': 0,
                'doHz': 130.8,
            }
        )
        init_reply = ws.receive_json()
        assert init_reply['type'] == 'status'

        # Send enough frames to trigger at least one pitch_update event.
        for _ in range(60):
            chunk = (0.2 * np.sin(2 * np.pi * 220 * np.arange(1024) / 44100)).astype(np.float32)
            ws.send_bytes(chunk.tobytes())

        got_pitch = False
        for _ in range(20):
            payload = ws.receive_json()
            if payload.get('type') == 'pitch_update':
                got_pitch = True
                break
        assert got_pitch
