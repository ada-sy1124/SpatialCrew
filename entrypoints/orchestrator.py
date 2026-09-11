"""High-level orchestrator for batch evaluation with multiple models/datasets."""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialcrew.config import SpatialCrewConfig

logger = logging.getLogger(__name__)


def discover_configs(config_dir: str, pattern: str = "*.json"):
    """Discover all JSON config files in a directory."""
    return sorted(Path(config_dir).glob(pattern))


async def run_single(model_json: str, dataset_json: str, work_dir: str):
    """Run a single evaluation configuration."""
    from entrypoints.run import run_evaluation
    import shortuuid

    session_id = shortuuid.uuid()[:8]
    config = SpatialCrewConfig.from_files(
        model_json=model_json,
        dataset_json=dataset_json,
        work_dir=work_dir,
    )
    return await run_evaluation(config, session_id)


def main():
    parser = argparse.ArgumentParser(description="SpatialCrew Batch Orchestrator")
    parser.add_argument("--model-configs", type=str, default="configs/models",
                        help="Directory containing model JSON configs")
    parser.add_argument("--dataset-configs", type=str, default="configs/datasets",
                        help="Directory containing dataset JSON configs")
    parser.add_argument("--work-dir", type=str, default="./outputs")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Specific model config filenames to run (without directory)")
    parser.add_argument("--datasets", type=str, nargs="*",
                        help="Specific dataset config filenames to run (without directory)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_configs = discover_configs(args.model_configs)
    dataset_configs = discover_configs(args.dataset_configs)

    if args.models:
        model_configs = [Path(args.model_configs) / m for m in args.models]
    if args.datasets:
        dataset_configs = [Path(args.dataset_configs) / d for d in args.datasets]

    logger.info(f"Found {len(model_configs)} model configs, {len(dataset_configs)} dataset configs")

    all_results = []
    for model_cfg, dataset_cfg in product(model_configs, dataset_configs):
        logger.info(f"Running: {model_cfg.stem} x {dataset_cfg.stem}")
        try:
            result = asyncio.run(run_single(str(model_cfg), str(dataset_cfg), args.work_dir))
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed: {model_cfg.stem} x {dataset_cfg.stem}: {e}")

    summary_path = Path(args.work_dir) / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Batch complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
