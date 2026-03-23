from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SamplingMetadata:
    fps: float
    total_frames: int
    sampled_indices: list[int]
    scene_cuts: int
    mean_motion: float


def _frame_score(curr_gray: np.ndarray, prev_gray: np.ndarray) -> float:
    diff = cv2.absdiff(curr_gray, prev_gray)
    return float(np.mean(diff))


def adaptive_sample_frames(
    video_path: Path,
    base_interval: int = 6,
    max_frames: int = 20,
    min_frames: int = 10,
    image_size: int = 224,
    scene_threshold: float = 22.0,
    motion_threshold: float = 10.0,
) -> tuple[np.ndarray, SamplingMetadata]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    sampled_candidates: list[tuple[int, np.ndarray, float]] = []
    frame_idx = 0
    prev_gray: np.ndarray | None = None
    scene_cuts = 0
    motion_scores: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = 0.0

        if prev_gray is not None:
            score = _frame_score(gray, prev_gray)
            motion_scores.append(score)

        is_uniform_pick = frame_idx % max(base_interval, 1) == 0
        is_scene_change = score >= scene_threshold
        is_motion_peak = score >= motion_threshold

        if is_scene_change:
            scene_cuts += 1

        if is_uniform_pick or is_scene_change or is_motion_peak:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            sampled_candidates.append((frame_idx, resized, score))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not sampled_candidates:
        raise ValueError("No frames were extracted from the video")

    # Deterministic ranking: prioritize high-motion candidates first.
    sampled_candidates.sort(key=lambda row: (row[2], -row[0]), reverse=True)

    # Keep within range [min_frames, max_frames], then sort back by time.
    keep = sampled_candidates[:max_frames]

    if len(keep) < min_frames:
        # Backfill uniformly from candidate pool to reach minimum count.
        pool = sorted(sampled_candidates, key=lambda row: row[0])
        needed = min_frames - len(keep)
        stride = max(len(pool) // max(needed, 1), 1)
        extras = [pool[i] for i in range(0, len(pool), stride)][:needed]
        keep.extend(extras)

    deduped: dict[int, tuple[int, np.ndarray, float]] = {row[0]: row for row in keep}
    ordered = [deduped[idx] for idx in sorted(deduped)]

    sampled_indices = [row[0] for row in ordered]
    sampled = [row[1] for row in ordered]

    frames = np.stack(sampled, axis=0)
    metadata = SamplingMetadata(
        fps=float(fps),
        total_frames=total_frames,
        sampled_indices=sampled_indices,
        scene_cuts=scene_cuts,
        mean_motion=float(np.mean(motion_scores)) if motion_scores else 0.0,
    )
    return frames, metadata
