from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import DatasetSubsetManager


def main() -> None:
    data_root = ROOT / "data"
    manager = DatasetSubsetManager(data_root=data_root)

    records = manager.bootstrap_subsets(videos_per_class=5, max_classes=2)
    labels = manager.get_candidate_labels()
    examples = manager.get_caption_examples(action_label="sports", limit=5)

    print(f"Dataset records: {len(records)}")
    print(f"Candidate labels: {labels}")
    print("Caption examples (sports):")
    for row in examples:
        print(f" - {row}")


if __name__ == "__main__":
    main()
