from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.schemas import HealthResponse, VideoSummaryResponse
from app.services.pipeline import VideoSummarizationPipeline
from app.services.video_processing import VideoProcessingService

router = APIRouter(tags=["video"])
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_pipeline() -> VideoSummarizationPipeline:
    return VideoSummarizationPipeline()


@lru_cache(maxsize=1)
def get_video_processing() -> VideoProcessingService:
    return VideoProcessingService()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/analyze-video", response_model=VideoSummaryResponse)
async def analyze_video(file: UploadFile = File(...)) -> VideoSummaryResponse:
    upload_path = None
    pipeline_timeout_seconds = float(os.getenv("PIPELINE_TIMEOUT_SECONDS", "300") or "300")

    try:
        logger.info("video received filename=%s content_type=%s", file.filename, file.content_type)
        upload_path = await get_video_processing().persist_upload(file)
        logger.info("file_received path=%s", upload_path)
        result = await asyncio.wait_for(
            run_in_threadpool(get_pipeline().run, upload_path),
            timeout=pipeline_timeout_seconds,
        )
        logger.info("returning response filename=%s", file.filename)
        logger.info("summary_generated filename=%s", file.filename)
        return VideoSummaryResponse(**result)
    except asyncio.TimeoutError:
        logger.exception("analyze_video_timeout filename=%s", file.filename)
        return JSONResponse(status_code=504, content={"error": "video analysis timed out"})
    except ValueError as exc:
        logger.warning("analyze_video_value_error filename=%s error=%s", file.filename, exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except RuntimeError as exc:
        logger.exception("analyze_video_runtime_error filename=%s", file.filename)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except Exception as exc:
        logger.exception("analyze_video_failure filename=%s", file.filename)
        return JSONResponse(status_code=500, content={"error": f"Video analysis failed: {exc}"})
    finally:
        get_video_processing().cleanup_path(upload_path)


@router.post("/upload", response_model=VideoSummaryResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoSummaryResponse:
    return await analyze_video(file=file)
