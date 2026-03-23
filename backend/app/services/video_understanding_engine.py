from __future__ import annotations

from pathlib import Path

from app.services.pipeline import VideoSummarizationPipeline


class VideoUnderstandingEngine:
    def __init__(self) -> None:
        self.pipeline = VideoSummarizationPipeline()

    def analyze_video(self, video_path: Path) -> dict:
        return self.pipeline.run(video_path)
