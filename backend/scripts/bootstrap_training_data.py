from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

# Random public GitHub video assets repo for quick bootstrap data.
DATASET_SOURCE = "https://github.com/mediaelement/mediaelement-files"

TRAINING_FILES = [
    {
        "label": "riding",
        "url": "https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4",
        "filename": "riding_001_big_buck_bunny.mp4",
    },
    {
        "label": "cooking",
        "url": "https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/echo-hereweare.mp4",
        "filename": "cooking_001_echo_hereweare.mp4",
    },
    {
        "label": "sports",
        "url": "https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.webm",
        "filename": "sports_001_big_buck_bunny.webm",
    },
]


def bootstrap_training_data(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "manifest.jsonl"

    records: list[dict] = []
    for item in TRAINING_FILES:
        target = output_dir / item["filename"]
        if not target.exists():
            print(f"Downloading {item['url']} -> {target.name}")
            urlretrieve(item["url"], target)

        record = {
            "video_path": str(target.resolve()),
            "label": item["label"],
            "source_repo": DATASET_SOURCE,
        }
        records.append(record)

    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return manifest_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "training" / "raw"
    manifest = bootstrap_training_data(raw_dir)
    print(f"Training data manifest ready: {manifest}")
