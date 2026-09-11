"""Main entry point for running SpatialCrew evaluations."""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialcrew.config import SpatialCrewConfig
from spatialcrew.coordinator.agent import CoordinatorAgent
from spatialcrew.evals.spbench import SPBench
from spatialcrew.kernel.pool import KernelPool
from spatialcrew.llm.client import LLMClient
from spatialcrew.tools.gpu_client import GPURPCClient
from spatialcrew.tools.registry import ToolRegistry
from spatialcrew.utils.logging import SessionLogger, setup_logging

logger = logging.getLogger(__name__)


async def run_evaluation(config: SpatialCrewConfig, session_id: str) -> dict:
    """Run a full benchmark evaluation."""
    setup_logging(config.log_dir, session_id=session_id)
    logger.info(f"Starting evaluation: model={config.llm_model}, benchmark={config.benchmark}")

    work_dir = Path(config.work_dir) / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    session_logger = SessionLogger(work_dir / "trace.jsonl", session_id)

    # Initialize components
    kernel_pool = KernelPool(pool_size=config.concurrency)
    await kernel_pool.initialize()

    llm_client = LLMClient(config)
    gpu_client = GPURPCClient(config)
    tool_registry = ToolRegistry(config, gpu_client=gpu_client)
    coordinator = CoordinatorAgent(config, llm_client, kernel_pool, tool_registry)

    # Load benchmark
    benchmark = SPBench(config.data_root, num_key_frames=config.num_key_frames)
    predictions = {}
    total_time = 0.0

    for subset in config.subsets:
        benchmark.load(subset)
        logger.info(f"Running subset {subset}: {len(benchmark)} samples")

        for sample in benchmark:
            t0 = time.monotonic()
            try:
                answer = await coordinator.solve(
                    question=sample.question,
                    images=sample.images,
                    metadata=sample.metadata,
                )
                predictions[sample.sample_id] = answer.answer
                elapsed = time.monotonic() - t0
                total_time += elapsed

                session_logger.log_event("sample_complete", {
                    "sample_id": sample.sample_id,
                    "prediction": answer.answer,
                    "ground_truth": sample.ground_truth,
                    "elapsed_sec": elapsed,
                })
                logger.info(
                    f"[{sample.sample_id}] pred={answer.answer} "
                    f"gt={sample.ground_truth} ({elapsed:.1f}s)"
                )
            except Exception as e:
                logger.error(f"[{sample.sample_id}] Error: {e}")
                session_logger.log_event("sample_error", {
                    "sample_id": sample.sample_id,
                    "error": str(e),
                })

    # Evaluate
    benchmark.load(config.subsets[0] if config.subsets else "si")
    metrics = benchmark.evaluate(predictions)
    metrics["total_time_sec"] = total_time
    metrics["avg_time_per_sample"] = total_time / max(len(predictions), 1)

    # Save results
    results = {
        "session_id": session_id,
        "config": config.to_dict(),
        "metrics": metrics,
        "predictions": predictions,
    }
    results_path = work_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Evaluation complete. Results saved to {results_path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    await kernel_pool.shutdown_all()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="SpatialCrew Evaluation Runner")
    parser.add_argument("--model-config", type=str, help="Path to model JSON config")
    parser.add_argument("--dataset-config", type=str, help="Path to dataset JSON config")
    parser.add_argument("--data-root", type=str, help="Override data root directory")
    parser.add_argument("--model", type=str, help="Override LLM model name")
    parser.add_argument("--subsets", type=str, nargs="+", help="Subsets to evaluate")
    parser.add_argument("--concurrency", type=int, help="Number of concurrent workers")
    parser.add_argument("--max-steps", type=int, help="Max WEVA steps per specialist")
    parser.add_argument("--work-dir", type=str, help="Output directory")
    parser.add_argument("--session-id", type=str, help="Session identifier")
    return parser.parse_args()


def main():
    args = parse_args()
    import shortuuid
    session_id = args.session_id or shortuuid.uuid()[:8]

    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.model:
        overrides["llm_model"] = args.model
    if args.subsets:
        overrides["subsets"] = args.subsets
    if args.concurrency:
        overrides["concurrency"] = args.concurrency
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.work_dir:
        overrides["work_dir"] = args.work_dir

    config = SpatialCrewConfig.from_files(
        model_json=args.model_config,
        dataset_json=args.dataset_config,
        **overrides,
    )

    asyncio.run(run_evaluation(config, session_id))


if __name__ == "__main__":
    main()
