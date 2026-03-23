from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def extract_frames(
    video_path: Path,
    sample_fps: float = 2.0,
    max_frames: int = 32,
    image_size: int = 224,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(native_fps / sample_fps)), 1)

    frames: list[np.ndarray] = []
    frame_idx = 0

    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(resized)

        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError("No frames were extracted from the video")

    return np.stack(frames, axis=0)


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)
