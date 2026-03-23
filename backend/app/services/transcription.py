from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import whisper

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self) -> None:
        self._ensure_ffmpeg_on_path()
        model_name = os.getenv("WHISPER_MODEL", "base").strip() or "base"
        self.transcription_timeout_seconds = float(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "240") or "240")
        try:
            self.model = whisper.load_model(model_name, device="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load whisper model '{model_name}': {exc}") from exc

    def transcribe_audio(self, audio_path: Path | None) -> str:
        if audio_path is None:
            logger.info("transcription skipped: no audio track")
            return ""

        self._ensure_ffmpeg_on_path()
        logger.info("transcription started audio_path=%s", audio_path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.model.transcribe,
                str(audio_path),
                fp16=False,
                language="en",
                task="transcribe",
            )
            try:
                result = future.result(timeout=self.transcription_timeout_seconds)
            except FutureTimeoutError as exc:
                raise RuntimeError(
                    f"Transcription timed out after {self.transcription_timeout_seconds:.0f}s"
                ) from exc

        transcript = str(result.get("text", "")).strip()
        print("TRANSCRIPT:", transcript)
        logger.info("transcript=%s", transcript)
        logger.info("transcription completed length=%d", len(transcript))

        if not transcript:
            logger.warning("empty_transcript_detected")
            return ""

        return transcript

    @staticmethod
    def _ensure_ffmpeg_on_path() -> None:
        ffmpeg_path = (os.getenv("FFMPEG_PATH", "") or "").strip()
        if not ffmpeg_path:
            return

        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        path_value = os.getenv("PATH", "")
        entries = [entry.strip() for entry in path_value.split(os.pathsep) if entry.strip()]
        if ffmpeg_dir not in entries:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + path_value

        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"ffmpeg is not available. Check FFMPEG_PATH: {ffmpeg_path}")
