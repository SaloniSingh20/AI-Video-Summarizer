from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_sample_video(output_path: Path, seconds: int = 2, fps: int = 10) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")

    total_frames = seconds * fps
    for idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + idx * 4, 80), (100 + idx * 4, 160), (0, 255, 255), -1)
        cv2.putText(frame, f"F{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return output_path


if __name__ == "__main__":
    target = Path(__file__).parent / "sample_test_video.mp4"
    created = generate_sample_video(target)
    print(f"Created {created}")
