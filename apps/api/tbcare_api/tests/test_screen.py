"""
Unit tests for POST /api/v1/screen endpoint.
"""
import json
import pytest
import numpy as np
from io import BytesIO
import wave

from tbcare_api.app import create_app
from tbcare_api.schemas.payload import ScreenPayload
from tbcare_api.schemas.response import TIER_MESSAGES, ERROR_CODES


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def make_wav_bytes(duration_s=2.0, sample_rate=16000, frequency=200) -> bytes:
    """Generate a simple sine wave WAV for testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def make_valid_payload(**overrides) -> dict:
    """Create a valid payload dict."""
    base = {
        "age": 38,
        "sex": "female",
        "cough_duration": "2_4",
        "hemoptysis": False,
        "fever": False,
        "night_sweats": False,
        "weight_loss": False,
        "prior_tb": False,
        "smoking": "never",
    }
    base.update(overrides)
    return base


class TestScreenEndpoint:
    """Tests for POST /api/v1/screen."""

    def test_missing_audio(self, client):
        """Missing audio file returns 400."""
        resp = client.post(
            "/api/v1/screen",
            data={"payload": json.dumps(make_valid_payload())},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_audio"

    def test_missing_payload(self, client):
        """Missing payload returns 400."""
        audio_bytes = make_wav_bytes()
        resp = client.post(
            "/api/v1/screen",
            data={"audio": (BytesIO(audio_bytes), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_json_payload(self, client):
        """Invalid JSON in payload returns 400."""
        audio_bytes = make_wav_bytes()
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": "not valid json",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_age_too_young(self, client):
        """Age < 18 returns 400."""
        audio_bytes = make_wav_bytes()
        payload = make_valid_payload(age=17)
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": json.dumps(payload),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_age_too_old(self, client):
        """Age > 100 returns 400."""
        audio_bytes = make_wav_bytes()
        payload = make_valid_payload(age=101)
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": json.dumps(payload),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_sex(self, client):
        """Invalid sex returns 400."""
        audio_bytes = make_wav_bytes()
        payload = make_valid_payload(sex="other")
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": json.dumps(payload),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_cough_duration(self, client):
        """Invalid cough_duration returns 400."""
        audio_bytes = make_wav_bytes()
        payload = make_valid_payload(cough_duration="invalid")
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": json.dumps(payload),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_invalid_smoking(self, client):
        """Invalid smoking value returns 400."""
        audio_bytes = make_wav_bytes()
        payload = make_valid_payload(smoking="maybe")
        resp = client.post(
            "/api/v1/screen",
            data={
                "audio": (BytesIO(audio_bytes), "test.wav"),
                "payload": json.dumps(payload),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["error"] == "invalid_payload"

    def test_valid_payload_parsing(self):
        """ScreenPayload.from_json parses correctly."""
        payload = ScreenPayload.from_json(make_valid_payload())
        assert payload.age == 38
        assert payload.sex == "female"
        assert payload.cough_duration == "2_4"
        assert payload.hemoptysis is False
        assert payload.smoking == "never"

    def test_payload_optional_none(self):
        """Optional symptoms can be None (don't know)."""
        payload = ScreenPayload.from_json(make_valid_payload(hemoptysis=None, fever=None))
        assert payload.hemoptysis is None
        assert payload.fever is None

    def test_symptom_vector_conversion(self):
        """to_symptom_vector produces correct shape."""
        payload = ScreenPayload.from_json(make_valid_payload())
        symptoms, availability = payload.to_symptom_vector()
        assert len(symptoms) == 9
        assert len(availability) == 9
        assert all(v in (0, 1) for v in availability)

    def test_symptom_vector_with_none(self):
        """to_symptom_vector handles None correctly."""
        payload = ScreenPayload.from_json(make_valid_payload(hemoptysis=None, smoking=None))
        symptoms, availability = payload.to_symptom_vector()
        # hemoptysis (index 0) -> 0.5 value, 0 availability
        assert symptoms[0] == 0.5
        assert availability[0] == 0
        # smoking (indices 5,6,7) -> 1/3 each, 0 availability
        assert symptoms[5] == pytest.approx(1/3)
        assert availability[5] == 0
        assert symptoms[6] == pytest.approx(1/3)
        assert availability[6] == 0
        assert symptoms[7] == pytest.approx(1/3)
        assert availability[7] == 0

    def test_cough_duration_value(self):
        """cough_duration maps to numeric weeks."""
        payload = ScreenPayload.from_json(make_valid_payload(cough_duration="2_4"))
        assert payload.to_cough_duration_value() == 3.0
        payload = ScreenPayload.from_json(make_valid_payload(cough_duration="4_8"))
        assert payload.to_cough_duration_value() == 6.0
        payload = ScreenPayload.from_json(make_valid_payload(cough_duration="8_plus"))
        assert payload.to_cough_duration_value() == 10.0

    def test_sex_value(self):
        """sex maps to 0/1."""
        payload = ScreenPayload.from_json(make_valid_payload(sex="female"))
        assert payload.sex_value() == 0.0
        payload = ScreenPayload.from_json(make_valid_payload(sex="male"))
        assert payload.sex_value() == 1.0


class TestResponseSchemas:
    """Tests for response schemas."""

    def test_tier_messages_exist(self):
        """All tiers have messages."""
        assert "A" in TIER_MESSAGES
        assert "B" in TIER_MESSAGES
        assert "C" in TIER_MESSAGES
        for tier in ("A", "B", "C"):
            assert "title" in TIER_MESSAGES[tier]
            assert "explanation" in TIER_MESSAGES[tier]

    def test_error_codes_exist(self):
        """Expected error codes exist."""
        expected = ["invalid_audio", "invalid_payload", "quality_failed", "inference_error", "timeout"]
        for code in expected:
            assert code in ERROR_CODES

    def test_create_tier_response(self):
        """create_tier_response builds correct structure."""
        from tbcare_api.schemas.response import create_tier_response
        resp = create_tier_response("A")
        assert resp.tier == "A"
        assert "explanation" in resp.to_json()
        assert "next_steps" in resp.to_json()

    def test_create_error_response(self):
        """create_error_response builds correct structure."""
        from tbcare_api.schemas.response import create_error_response
        resp = create_error_response("invalid_audio", {"reason": "test"})
        assert resp.error == "invalid_audio"
        assert resp.message == "Berkas audio tidak valid"
        assert resp.details["reason"] == "test"


class TestScreenPayload:
    """Additional payload validation tests."""

    def test_string_boolean_conversion(self):
        """String 'true'/'false' converted to boolean."""
        payload = ScreenPayload.from_json(make_valid_payload(hemoptysis="true", fever="false"))
        assert payload.hemoptysis is True
        assert payload.fever is False

    def test_missing_optional_fields(self):
        """Missing optional fields become None."""
        minimal = {
            "age": 30,
            "sex": "male",
            "cough_duration": "2_4",
        }
        payload = ScreenPayload.from_json(minimal)
        assert payload.hemoptysis is None
        assert payload.fever is None
        assert payload.night_sweats is None
        assert payload.weight_loss is None
        assert payload.prior_tb is None
        assert payload.smoking is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])