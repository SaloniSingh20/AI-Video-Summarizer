from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "training" / "manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run bootstrap_training_data.py first."
        )

    records = load_manifest(manifest_path)
    label_counts = Counter(item["label"] for item in records)

    print("Baseline training scaffold")
    print(f"Total samples: {len(records)}")
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count}")

    print("\nNext step: plug this manifest into your full PyTorch dataloader/training loop.")


if __name__ == "__main__":
    main()
