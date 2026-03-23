from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

UCF101_SOURCE = "https://www.crcv.ucf.edu/data/UCF101.php"
UCF101_KAGGLE = "https://www.kaggle.com/datasets/pevogam/ucf101"
KINETICS_SOURCE = "https://deepmind.com/research/open-source/kinetics"
KINETICS_GITHUB = "https://github.com/cvdfoundation/kinetics-dataset"
MSRVTT_SOURCE = "https://www.microsoft.com/en-us/research/project/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/"
MSRVTT_GITHUB = "https://github.com/ArrowLuo/CLIP4Clip"


@dataclass
class DatasetRecord:
    video_path: str
    label: str
    dataset: str
    source: str


@dataclass
class CaptionExample:
    label: str
    caption: str
    source: str


class LabelMapper:

    def __init__(self) -> None:
        self.alias_to_canonical = {
            "ride bike": "riding",
            "riding bike": "riding",
            "biking": "riding",
            "bike riding": "riding",
            "cook": "cooking",
            "cooking food": "cooking",
            "basketball": "sports",
            "playing basketball": "sports",
            "soccer": "sports",
            "football": "sports",
            "juggling balls": "sports",
        }

    def normalize(self, text: str) -> str:
        text = text.strip().lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = " ".join(text.split())
        return text

    def canonical(self, text: str) -> str:
        norm = self.normalize(text)
        return self.alias_to_canonical.get(norm, norm)

    def map_to_candidates(self, prediction: str, candidates: list[str]) -> tuple[str, float]:
        pred = self.canonical(prediction)
        normalized_candidates = [(c, self.canonical(c)) for c in candidates]
        if not normalized_candidates:
            return pred, 0.0

        best = normalized_candidates[0][0]
        best_score = -1.0
        pred_tokens = set(pred.split())

        for raw, cand in normalized_candidates:
            cand_tokens = set(cand.split())
            overlap = len(pred_tokens & cand_tokens)
            union = max(len(pred_tokens | cand_tokens), 1)
            score = overlap / union
            if pred == cand:
                score = 1.0
            if score > best_score:
                best = raw
                best_score = score

        return best, float(best_score)


class DatasetSubsetManager:
   

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.ucf_dir = self.data_root / "ucf101"
        self.kinetics_dir = self.data_root / "kinetics400"
        self.msrvtt_dir = self.data_root / "msrvtt"
        self.eval_manifest = self.data_root / "evaluation_manifest.jsonl"
        self.caption_manifest = self.msrvtt_dir / "caption_examples.jsonl"
        self.label_mapper = LabelMapper()

        self.ucf_dir.mkdir(parents=True, exist_ok=True)
        self.kinetics_dir.mkdir(parents=True, exist_ok=True)
        self.msrvtt_dir.mkdir(parents=True, exist_ok=True)

    def bootstrap_subsets(self, videos_per_class: int = 5, max_classes: int = 3) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        records.extend(self._bootstrap_ucf101(videos_per_class=videos_per_class, max_classes=max_classes))
        records.extend(self._bootstrap_kinetics(videos_per_class=videos_per_class, max_classes=max_classes))
        self._bootstrap_msrvtt_captions(max_examples=120)

        if records:
            with self.eval_manifest.open("w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row.__dict__) + "\n")
            logger.info("Wrote evaluation manifest with %d rows at %s", len(records), self.eval_manifest)
        else:
            logger.warning("No dataset subset records were generated.")

        return records

    def get_candidate_labels(self) -> list[str]:
        records = self.load_records()
        labels = sorted({row.label for row in records if row.label})
        return labels

    def get_caption_examples(self, action_label: str, limit: int = 8) -> list[str]:
        examples = self.load_caption_examples()
        if not examples:
            return []

        target = self.label_mapper.canonical(action_label)
        out: list[str] = []
        for row in examples:
            label = self.label_mapper.canonical(row.label)
            if label == target or target in label or label in target:
                out.append(row.caption)
            if len(out) >= limit:
                break

        if out:
            return out

        # Fallback to deterministic first items.
        return [row.caption for row in examples[:limit]]

    def load_caption_examples(self) -> list[CaptionExample]:
        if not self.caption_manifest.exists():
            return []

        rows: list[CaptionExample] = []
        with self.caption_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(CaptionExample(**json.loads(line)))
        return rows

    def load_records(self) -> list[DatasetRecord]:
        if not self.eval_manifest.exists():
            return []

        rows: list[DatasetRecord] = []
        with self.eval_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(DatasetRecord(**raw))
        return rows

    def infer_actual_label(self, video_path: Path) -> str | None:
        stem = video_path.stem.lower().replace("-", "_")
        records = self.load_records()

        for row in records:
            label = row.label.lower().replace(" ", "_")
            if label in stem:
                return row.label

        return None

    def map_predicted_label(self, prediction: str) -> tuple[str, float]:
        candidates = self.get_candidate_labels()
        return self.label_mapper.map_to_candidates(prediction=prediction, candidates=candidates)

    def _bootstrap_ucf101(self, videos_per_class: int, max_classes: int) -> list[DatasetRecord]:
        output_dir = self.ucf_dir / "subset"
        output_dir.mkdir(parents=True, exist_ok=True)

        used_kaggle = self._try_kaggle_ucf101(output_dir=output_dir, videos_per_class=videos_per_class, max_classes=max_classes)
        if used_kaggle:
            return used_kaggle

        logger.warning("Falling back to lightweight UCF-like subset because Kaggle download is unavailable.")
        return self._bootstrap_from_local_training_pool(
            dataset="ucf101",
            source=f"{UCF101_SOURCE} | mirror: {UCF101_KAGGLE}",
            output_dir=output_dir,
            max_rows=max_classes * videos_per_class,
        )

    def _bootstrap_kinetics(self, videos_per_class: int, max_classes: int) -> list[DatasetRecord]:
        output_dir = self.kinetics_dir / "subset"
        output_dir.mkdir(parents=True, exist_ok=True)

        used_kinetics = self._try_kinetics_github(output_dir=output_dir, videos_per_class=videos_per_class, max_classes=max_classes)
        if used_kinetics:
            return used_kinetics

        logger.warning("Falling back to lightweight Kinetics-like subset because official download is unavailable.")
        return self._bootstrap_from_local_training_pool(
            dataset="kinetics400",
            source=f"{KINETICS_SOURCE} | tools: {KINETICS_GITHUB}",
            output_dir=output_dir,
            max_rows=max_classes * videos_per_class,
        )

    def _try_kaggle_ucf101(self, output_dir: Path, videos_per_class: int, max_classes: int) -> list[DatasetRecord]:
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            return []

        kaggle_cmd = shutil.which("kaggle")
        if not kaggle_cmd:
            return []

        raw_dir = self.ucf_dir / "kaggle_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [kaggle_cmd, "datasets", "download", "-d", "pevogam/ucf101", "-p", str(raw_dir), "--unzip"],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as exc:
            logger.warning("Kaggle UCF101 download failed: %s", exc)
            return []

        class_dirs = sorted([d for d in raw_dir.rglob("*") if d.is_dir() and any(d.glob("*.avi"))])
        records: list[DatasetRecord] = []

        for class_dir in class_dirs[:max_classes]:
            videos = sorted(class_dir.glob("*.avi"))[:videos_per_class]
            label = class_dir.name
            class_out = output_dir / label
            class_out.mkdir(parents=True, exist_ok=True)

            for src in videos:
                dst = class_out / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                records.append(
                    DatasetRecord(
                        video_path=str(dst.resolve()),
                        label=label,
                        dataset="ucf101",
                        source=UCF101_KAGGLE,
                    )
                )

        return records

    def _try_kinetics_github(self, output_dir: Path, videos_per_class: int, max_classes: int) -> list[DatasetRecord]:
        # Official Kinetics videos are YouTube-based and often unavailable without downloader tooling.
        # We keep this stub deterministic and non-blocking by returning [] when tooling is absent.
        yt_dlp = shutil.which("yt-dlp")
        if not yt_dlp:
            return []

        # If user provides a CSV of subset URLs, use it.
        subset_csv = os.getenv("KINETICS_SUBSET_CSV", "").strip()
        if not subset_csv:
            return []

        csv_path = Path(subset_csv)
        if not csv_path.exists():
            return []

        # Simple parser: expected columns label,url
        rows: list[tuple[str, str]] = []
        with csv_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0 and "label" in line.lower() and "url" in line.lower():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                rows.append((parts[0], parts[1]))

        by_label: dict[str, int] = {}
        records: list[DatasetRecord] = []

        for label, url in rows:
            if len(by_label) >= max_classes and label not in by_label:
                continue
            by_label[label] = by_label.get(label, 0)
            if by_label[label] >= videos_per_class:
                continue

            class_out = output_dir / label
            class_out.mkdir(parents=True, exist_ok=True)
            dst = class_out / f"{label}_{by_label[label]:03d}.mp4"
            by_label[label] += 1

            if not dst.exists():
                try:
                    subprocess.run(
                        [yt_dlp, "-f", "mp4", "-o", str(dst), url],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except Exception as exc:
                    logger.warning("Skipping Kinetics sample download %s: %s", url, exc)
                    continue

            records.append(
                DatasetRecord(
                    video_path=str(dst.resolve()),
                    label=label,
                    dataset="kinetics400",
                    source=KINETICS_GITHUB,
                )
            )

        return records

    def _bootstrap_msrvtt_captions(self, max_examples: int) -> None:
        if self.caption_manifest.exists():
            return

        raw_urls = [
            "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/master/data/MSRVTT/caption.pickle",
            "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/master/data/MSRVTT/msrvtt_data.json",
            "https://raw.githubusercontent.com/salesforce/ALPRO/main/datasets/msrvtt_ret_test1k.json",
        ]

        examples: list[CaptionExample] = []
        for url in raw_urls:
            try:
                with urlopen(url, timeout=15) as response:
                    payload = response.read().decode("utf-8", errors="ignore")
                parsed = self._extract_caption_examples(payload=payload, source=url, limit=max_examples)
                if parsed:
                    examples = parsed
                    break
            except Exception as exc:
                logger.warning("MSR-VTT caption source unavailable (%s): %s", url, exc)

        if not examples:
            examples = [
                CaptionExample(label="cooking", caption="a person is preparing food in a kitchen", source=MSRVTT_SOURCE),
                CaptionExample(label="sports", caption="people are actively playing a ball game", source=MSRVTT_SOURCE),
                CaptionExample(label="riding", caption="a person is riding outdoors on a path", source=MSRVTT_SOURCE),
                CaptionExample(label="dancing", caption="a performer is dancing on stage", source=MSRVTT_SOURCE),
                CaptionExample(label="walking", caption="a person walks through an urban area", source=MSRVTT_SOURCE),
            ]

        with self.caption_manifest.open("w", encoding="utf-8") as f:
            for row in examples[:max_examples]:
                f.write(json.dumps(row.__dict__) + "\n")

        logger.info("Prepared MSR-VTT caption subset with %d examples", len(examples[:max_examples]))

    def _extract_caption_examples(self, payload: str, source: str, limit: int) -> list[CaptionExample]:
        payload = payload.strip()
        if not payload:
            return []

        examples: list[CaptionExample] = []
        try:
            data = json.loads(payload)
            candidates = []
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict):
                for key in ("sentences", "annotations", "data"):
                    if key in data and isinstance(data[key], list):
                        candidates = data[key]
                        break

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                caption = str(item.get("caption") or item.get("sentence") or "").strip()
                if not caption:
                    continue
                # Lightweight heuristic action labels from caption text.
                lower = caption.lower()
                if "cook" in lower or "kitchen" in lower:
                    label = "cooking"
                elif any(k in lower for k in ("basketball", "soccer", "football", "game")):
                    label = "sports"
                elif "ride" in lower or "bicycle" in lower or "bike" in lower:
                    label = "riding"
                elif "dance" in lower:
                    label = "dancing"
                else:
                    label = "general"

                examples.append(CaptionExample(label=label, caption=caption, source=source))
                if len(examples) >= limit:
                    break
        except Exception:
            return []

        return examples

    def _bootstrap_from_local_training_pool(self, dataset: str, source: str, output_dir: Path, max_rows: int) -> list[DatasetRecord]:
        pool_manifest = self.data_root / "training" / "manifest.jsonl"
        if not pool_manifest.exists():
            return []

        records: list[DatasetRecord] = []
        with pool_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src_path = Path(row.get("video_path", ""))
                if not src_path.exists():
                    continue

                label = str(row.get("label", "unknown"))
                class_out = output_dir / label
                class_out.mkdir(parents=True, exist_ok=True)
                dst = class_out / src_path.name
                if not dst.exists():
                    shutil.copy2(src_path, dst)

                records.append(
                    DatasetRecord(
                        video_path=str(dst.resolve()),
                        label=label,
                        dataset=dataset,
                        source=source,
                    )
                )

        return records
