from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import UploadFile

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".m4v",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".3gp",
    ".flv",
}


class VideoProcessingService:
    def __init__(self) -> None:
        self.workspace_dir = Path(tempfile.gettempdir()) / "ai_video_summarizer"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    async def persist_upload(self, upload: UploadFile) -> Path:
        if not upload.filename:
            raise ValueError("Missing filename")

        suffix = Path(upload.filename).suffix.lower()
        content_type = (upload.content_type or "").lower()
        if not suffix and content_type.startswith("video/"):
            suffix = ".mp4"

        if suffix not in ALLOWED_VIDEO_SUFFIXES and not content_type.startswith("video/"):
            raise ValueError("Unsupported video format")

        video_path = self.workspace_dir / f"{uuid4()}{suffix or '.mp4'}"
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        return video_path

    def extract_audio(self, video_path: Path) -> Path | None:
        audio_path = self.workspace_dir / f"{video_path.stem}.wav"
        ffmpeg_binary = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg") or "ffmpeg"
        ffmpeg_timeout_seconds = float(os.getenv("FFMPEG_TIMEOUT_SECONDS", "120") or "120")

        if not self._has_audio_stream(video_path):
            logger.warning("no_audio_stream_detected video_path=%s", video_path)
            return None

        cmd = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-map",
            "a:0",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=ffmpeg_timeout_seconds)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required but was not found in PATH. Set FFMPEG_PATH if needed.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Audio extraction timed out after {ffmpeg_timeout_seconds:.0f}s") from exc
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            if "does not contain any stream" in stderr or "matches no streams" in stderr:
                logger.warning("no_audio_stream_detected video_path=%s", video_path)
                return None
            raise RuntimeError(f"Audio extraction failed: {proc.stderr[-600:]}")

        logger.info("extracted_audio_path=%s", audio_path)
        return audio_path

    def _has_audio_stream(self, video_path: Path) -> bool:
        ffprobe_binary = os.getenv("FFPROBE_PATH", "").strip() or shutil.which("ffprobe") or "ffprobe"
        cmd = [
            ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                logger.warning("ffprobe_audio_check_failed returncode=%s stderr=%s", proc.returncode, (proc.stderr or "")[-300:])
                return True
            return bool((proc.stdout or "").strip())
        except FileNotFoundError:
            logger.warning("ffprobe_not_found_skipping_audio_precheck")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe_audio_check_timeout")
            return True

    def sample_visual_cues(self, video_path: Path, interval_seconds: float = 3.0) -> list[str]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            capture.release()
            return ["The available visual context is limited."]

        sample_points = sorted({0, max(0, total_frames // 2)})
        sampled_frames: list = []

        for frame_idx in sample_points:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            sampled_frames.append(frame)

        capture.release()
        if not sampled_frames:
            return ["The available visual context is limited."]

        face_hits = 0
        natural_votes = 0
        built_votes = 0
        prev_gray = None
        motion_detected = False

        for frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                face_hits += 1

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_pixels = cv2.inRange(hsv, (35, 35, 30), (90, 255, 255))
            blue_pixels = cv2.inRange(hsv, (90, 35, 30), (135, 255, 255))
            gray_pixels = cv2.inRange(hsv, (0, 0, 20), (179, 40, 230))
            total_pixels = frame.shape[0] * frame.shape[1]

            green_ratio = cv2.countNonZero(green_pixels) / max(total_pixels, 1)
            blue_ratio = cv2.countNonZero(blue_pixels) / max(total_pixels, 1)
            gray_ratio = cv2.countNonZero(gray_pixels) / max(total_pixels, 1)

            if green_ratio + blue_ratio > 0.28:
                natural_votes += 1
            elif gray_ratio > 0.45:
                built_votes += 1

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                if float(diff.mean()) > 7.5:
                    motion_detected = True
            prev_gray = gray

        cues: list[str] = []
        if face_hits > 0:
            cues.append("Sampled frames suggest one or more people are visible.")
        else:
            cues.append("No clearly visible person appears in the sampled frames.")

        if natural_votes >= built_votes and natural_votes > 0:
            cues.append("This appears to be an outdoor natural scene.")
        elif built_votes > 0:
            cues.append("This appears to be an indoor or built-environment scene.")
        else:
            cues.append("The scene type is unclear from the sampled frames.")

        if motion_detected:
            cues.append("There is visible movement in the scene.")
        else:
            cues.append("The scene appears mostly steady.")

        return cues

    @staticmethod
    def cleanup_path(path: Path | None) -> None:
        if path and path.exists():
            path.unlink(missing_ok=True)
