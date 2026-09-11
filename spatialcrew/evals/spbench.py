"""SPBench benchmark loader and evaluator."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from spatialcrew.evals.base import (
    BaseBenchmark, Sample, compute_accuracy, compute_mra,
)
from spatialcrew.types.frames import FrameImage, InputImages
from spatialcrew.types.metadata import Metadata

logger = logging.getLogger(__name__)


class SPBench(BaseBenchmark):
    """SPBench: Spatial Reasoning Benchmark with single-image and multi-view subsets."""

    def __init__(self, data_root: str, num_key_frames: int = 16):
        self._data_root = Path(data_root)
        self._num_key_frames = num_key_frames
        self._samples: List[Dict[str, Any]] = []
        self._subset: str = ""

    def load(self, subset: str = "si") -> None:
        """Load a subset of SPBench (si=single-image, mv=multi-view)."""
        self._subset = subset
        anno_path = self._data_root / f"spbench_{subset}.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"SPBench annotation file not found: {anno_path}")

        with open(anno_path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            self._samples = data.get("samples", data.get("data", []))
        else:
            self._samples = data

        logger.info(f"Loaded SPBench/{subset}: {len(self._samples)} samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        for item in self._samples:
            yield self._parse_sample(item)

    def _parse_sample(self, item: Dict[str, Any]) -> Sample:
        """Convert a raw annotation dict to a Sample."""
        sample_id = str(item.get("id", item.get("sample_id", "")))
        question = item.get("question", "")
        gt = str(item.get("answer", item.get("ground_truth", "")))
        choices = item.get("choices", item.get("options", None))

        # Load images
        images = InputImages()
        image_paths = item.get("images", item.get("image_paths", []))
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for i, img_path in enumerate(image_paths):
            full_path = self._data_root / img_path
            if full_path.exists():
                images.append(FrameImage.from_path(str(full_path), frame_index=i))

        if not images and "image" in item:
            single_path = self._data_root / item["image"]
            if single_path.exists():
                images.append(FrameImage.from_path(str(single_path), frame_index=0))

        metadata = Metadata(
            num_videos=0,
            image_width=images[0].width if images else 0,
            image_height=images[0].height if images else 0,
            question_type=item.get("question_type", item.get("category", "")),
            choices=choices,
        )

        return Sample(
            sample_id=sample_id,
            question=question,
            images=images,
            metadata=metadata,
            ground_truth=gt,
            choices=choices,
        )

    def evaluate(self, predictions: Dict[str, str]) -> Dict[str, float]:
        """Compute SPBench evaluation metrics."""
        gt_map = {}
        choices_map = {}
        category_map: Dict[str, Dict[str, str]] = {}

        for item in self._samples:
            sid = str(item.get("id", item.get("sample_id", "")))
            gt_map[sid] = str(item.get("answer", item.get("ground_truth", "")))
            choices_map[sid] = item.get("choices", item.get("options", ["A", "B", "C", "D"]))
            cat = item.get("question_type", item.get("category", "other"))
            if cat not in category_map:
                category_map[cat] = {}
            category_map[cat][sid] = gt_map[sid]

        metrics = {
            "accuracy": compute_accuracy(predictions, gt_map),
            "mra": compute_mra(predictions, gt_map, choices_map),
            "num_samples": float(len(gt_map)),
            "num_predicted": float(len(predictions)),
        }

        for cat, cat_gts in category_map.items():
            cat_preds = {sid: predictions[sid] for sid in cat_gts if sid in predictions}
            cat_choices = {sid: choices_map.get(sid, ["A", "B", "C", "D"]) for sid in cat_gts}
            metrics[f"accuracy_{cat}"] = compute_accuracy(cat_preds, cat_gts)
            metrics[f"mra_{cat}"] = compute_mra(cat_preds, cat_gts, cat_choices)

        return metrics
