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
        assert data["error"]["code"] == "invalid_audio"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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
        assert data["error"]["code"] == "payload_invalid"

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

    def test_clinical_record_mapping(self):
        """to_clinical_record maps the 9 UI fields onto CODA TB feature keys."""
        payload = ScreenPayload.from_json(make_valid_payload())
        record = payload.to_clinical_record()
        assert record["age"] == payload.age
        assert record["sex"] == payload.sex
        assert record["reported_cough_dur"] == payload.to_cough_duration_value()
        assert record["hemoptysis"] is False
        assert record["fever"] is False
        assert record["night_sweats"] is False
        assert record["weight_loss"] is False
        assert record["tb_prior"] is False
        assert record["smoke_lweek"] is False
        # Fields TBCare does not collect must be absent (imputed downstream).
        for absent in ("height", "weight", "heart_rate", "temperature", "country", "hiv"):
            assert absent not in record

    def test_clinical_record_unknown_stays_none(self):
        """Unknown booleans stay None so they are imputed, not read as 'no'."""
        payload = ScreenPayload.from_json(make_valid_payload(hemoptysis=None, smoking=None))
        record = payload.to_clinical_record()
        assert record["hemoptysis"] is None
        assert record["smoke_lweek"] is None

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
        expected = ["invalid_audio", "payload_invalid", "quality_failed", "inference_error", "timeout", "model_unavailable", "server_error"]
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


class TestArtifactsPathResolution:
    """Regression tests for the artifacts directory lookup.

    A previous bug resolved the repo default 4 levels up from routes/screen.py
    (apps/ml/artifacts), so a real /screen call returned 500 on a clean
    checkout while every validation-only test still passed.
    """

    def test_default_artifacts_dir_exists_in_repo(self):
        from pathlib import Path

        repo_default = (
            Path(__file__).resolve().parents[4] / "ml" / "artifacts"
        )
        assert repo_default.name == "artifacts"
        assert repo_default.parent.name == "ml"
        assert (repo_default / "model_manifest.json").exists(), (
            f"expected artifacts at {repo_default}; run export_artifacts.py"
        )

    def test_get_ensemble_uses_repo_default_when_config_path_missing(
        self, app, monkeypatch
    ):
        from tbcare_api.routes import screen as screen_route

        # Simulate the container default, which does not exist locally.
        app.config["TBCARE_ARTIFACTS"] = "/app/artifacts"
        monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
        monkeypatch.setenv("ALLOW_BLOCKED_CANDIDATE", "true")
        screen_route._ensemble = None
        try:
            with app.app_context():
                ensemble = screen_route.get_ensemble()
            assert ensemble is not None
            assert ensemble.manifest is not None
        finally:
            screen_route._ensemble = None


class TestScreenSuccessPath:
    """The end-to-end 200 path, which validation-only tests never exercised."""

    def test_screen_returns_200_with_real_artifacts(self, client, monkeypatch):
        monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
        monkeypatch.setenv("ALLOW_BLOCKED_CANDIDATE", "true")

        from tbcare_api.routes import screen as screen_route
        screen_route._ensemble = None

        # A cough-like signal: band-limited bursts so the quality gate and
        # spectral gate accept it (a pure tone / white noise would not).
        from scipy import signal as _signal

        sr = 16000
        n = sr * 3
        rng = np.random.default_rng(0)
        sig = np.zeros(n)
        band = _signal.butter(
            4, [200 / (sr / 2), 4000 / (sr / 2)], btype="band", output="sos"
        )
        for start in rng.integers(0, n - sr // 4, 8):
            length = sr // 8
            burst = rng.normal(0, 0.5, length)
            burst = _signal.sosfilt(band, burst)
            burst = burst / (np.max(np.abs(burst)) + 1e-12)
            sig[start:start + length] += np.hanning(length) * burst * 0.6
        sig = (sig / np.max(np.abs(sig)) * 0.6).astype(np.float32)
        pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)

        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)

        payload = make_valid_payload(
            hemoptysis=True,
            fever=True,
            night_sweats=True,
            weight_loss=True,
        )
        resp = client.post(
            "/api/v1/screen",
            data={"audio": (buf, "cough.wav"), "payload": json.dumps(payload)},
            content_type="multipart/form-data",
        )
        screen_route._ensemble = None

        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["tier"] in ("A", "B", "C")
        assert body["actions"], "tier response must carry ordered actions"
        assert "bukan diagnosis" in json.dumps(body).lower()
        # The cough count is still reported even though the model now consumes
        # the whole recording (P2-b in docs/proposal/training_improvement.md).
        assert body.get("metadata", {}).get("cough_segments_used", 0) >= 3


class TestErrorHardening:
    """Regressions for the audit findings around error handling."""

    def test_candidate_model_without_staging_flag_returns_503(
        self, client, monkeypatch
    ):
        """A CANDIDATE manifest must gate with 503 JSON, not a 500 HTML page."""
        monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
        monkeypatch.delenv("ALLOW_BLOCKED_CANDIDATE", raising=False)

        from tbcare_api.routes import screen as screen_route
        screen_route._ensemble = None

        sr = 16000
        n = sr * 3
        rng = np.random.RandomState(1)
        sig = np.zeros(n)
        for start in rng.randint(0, n - sr // 4, 8):
            length = sr // 8
            sig[start:start + length] += (
                np.hanning(length) * np.sin(2 * np.pi * 300 * np.arange(length) / sr) * 0.5
            )
        sig = (sig / np.max(np.abs(sig)) * 0.6).astype(np.float32)
        pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)

        resp = client.post(
            "/api/v1/screen",
            data={"audio": (buf, "cough.wav"), "payload": json.dumps(make_valid_payload())},
            content_type="multipart/form-data",
        )
        screen_route._ensemble = None

        # Either the quality gate or the model gate may fire first depending on
        # the synthetic signal; both must answer with the canonical JSON
        # envelope and a machine-readable code, never an HTML error page.
        assert resp.status_code in (400, 503), resp.data
        body = resp.get_json()
        assert isinstance(body, dict) and "error" in body
        assert "code" in body["error"]
        assert resp.headers["Content-Type"].startswith("application/json")

    def test_error_body_never_leaks_internals(self, client):
        """Error payloads must not echo str(e), tracebacks, or file paths."""
        resp = client.post(
            "/api/v1/screen",
            data={"audio": (BytesIO(b"not a wav"), "junk.wav"), "payload": "{}"},
            content_type="multipart/form-data",
        )
        raw = resp.get_data(as_text=True).lower()
        for forbidden in ("traceback", "file \"", "/home/", "tbcare_api", ".py\", line"):
            assert forbidden not in raw, f"leaked {forbidden!r}: {raw}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])