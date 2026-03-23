from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from tests.data.generate_sample_video import generate_sample_video


class DummyPipeline:
    def run(self, _video_path: Path) -> dict:
        return {
            "main_idea": "A short demonstration clip shows a moving object crossing the frame.",
            "detailed_summary": "The video begins with a simple scene and a visible object that moves from left to right. The motion remains consistent over the clip.",
            "key_insights": [
                "The scene contains one dominant moving subject.",
                "Motion is continuous throughout the sample.",
            ],
            "highlights": [
                "Object enters from the left side.",
                "Object exits near the right boundary.",
            ],
            "final_takeaway": "The clip captures a single clear movement pattern in a controlled scene.",
        }


class DummyVideoProcessing:
    async def persist_upload(self, _file) -> Path:
        return Path("tests/data/sample_test_video.mp4")

    def cleanup_path(self, _path: Path | None) -> None:
        return None


def test_analyze_video_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "get_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr(routes, "get_video_processing", lambda: DummyVideoProcessing())

    sample_video = generate_sample_video(tmp_path / "sample.mp4")
    client = TestClient(app)

    with sample_video.open("rb") as f:
        response = client.post(
            "/api/analyze-video",
            files={"file": ("sample.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "main_idea" in payload
    assert "detailed_summary" in payload
    assert isinstance(payload["key_insights"], list)
    assert isinstance(payload["highlights"], list)
    assert "final_takeaway" in payload


def test_upload_endpoint_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "get_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr(routes, "get_video_processing", lambda: DummyVideoProcessing())

    sample_video = generate_sample_video(tmp_path / "sample_upload.mp4")
    client = TestClient(app)
    with sample_video.open("rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample_upload.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "main_idea" in payload
