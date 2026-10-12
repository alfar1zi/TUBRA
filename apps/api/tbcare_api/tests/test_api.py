import io
import wave
import pytest
import numpy as np
from tbcare_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model_version" in data


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate
    signal = np.sin(2 * np.pi * freq * t)
    audio_int16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def test_echo_test_valid_wav(client):
    wav_data = make_wav_bytes(duration_s=0.5, sample_rate=16000)
    resp = client.post(
        "/api/v1/echo-test",
        data={"audio": (io.BytesIO(wav_data), "test.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["sample_rate"] == 16000
    assert abs(data["duration_s"] - 0.5) < 0.01
    assert 0 < data["peak"] <= 1.0
    assert data["rms"] > 0
    assert data["num_samples"] == 8000
    assert data["applied_constraints"]["required_format"] == "WAV PCM 16-bit mono"


def test_echo_test_stereo_rejected(client):
    wav_data = make_wav_bytes(duration_s=0.5, sample_rate=16000)
    # Make it stereo by duplicating channel
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        audio_int16 = np.frombuffer(wav_data[44:], dtype=np.int16)
        stereo = np.column_stack([audio_int16, audio_int16]).ravel()
        wf.writeframes(stereo.tobytes())
    stereo_data = buf.getvalue()

    resp = client.post(
        "/api/v1/echo-test",
        data={"audio": (io.BytesIO(stereo_data), "stereo.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "invalid_audio"


def test_echo_test_24bit_rejected(client):
    wav_data = make_wav_bytes(duration_s=0.5, sample_rate=16000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(16000)
        audio_int16 = np.frombuffer(wav_data[44:], dtype=np.int16)
        audio_24 = np.left_shift(audio_int16.astype(np.int32), 8)
        wf.writeframes(audio_24.tobytes())
    data_24 = buf.getvalue()

    resp = client.post(
        "/api/v1/echo-test",
        data={"audio": (io.BytesIO(data_24), "24bit.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "invalid_audio"


def test_echo_test_missing_audio_field(client):
    resp = client.post(
        "/api/v1/echo-test",
        data={},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "invalid_audio"


def test_echo_test_body_too_large(client):
    # Create a large WAV (>5MB)
    large_wav = make_wav_bytes(duration_s=200, sample_rate=48000)  # ~19 MB
    resp = client.post(
        "/api/v1/echo-test",
        data={"audio": (io.BytesIO(large_wav), "large.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    data = resp.get_json()
    assert data["error"]["code"] == "too_large"