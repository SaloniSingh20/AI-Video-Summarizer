from __future__ import annotations

import logging
from pathlib import Path

from app.services.summarization import SummarizationService
from app.services.transcription import TranscriptionService
from app.services.video_processing import VideoProcessingService

logger = logging.getLogger(__name__)


class VideoSummarizationPipeline:
    def __init__(self) -> None:
        self.video_processing = VideoProcessingService()
        self.transcription = TranscriptionService()
        self.summarization = SummarizationService()

    def run(self, video_path: Path) -> dict:
        audio_path: Path | None = None
        try:
            logger.info("pipeline_start video_path=%s", video_path)
            audio_path = self.video_processing.extract_audio(video_path)
            logger.info("audio extracted path=%s", audio_path)
            logger.info("transcription started")
            transcript = self.transcription.transcribe_audio(audio_path)
            logger.info("transcription completed length=%d", len(transcript))
            no_audio = not transcript.strip()
            if no_audio:
                transcript = "No speech detected in the video"
            visual_cues = self.video_processing.sample_visual_cues(video_path, interval_seconds=3.0)
            logger.info("visual_cues_collected count=%d", len(visual_cues))
            logger.info("summary started")
            summary = self.summarization.summarize(
                transcript=transcript,
                visual_cues=visual_cues,
                no_audio=no_audio,
            )
            logger.info("summary completed")
            logger.info("pipeline_complete")
            return summary
        finally:
            self.video_processing.cleanup_path(audio_path)
