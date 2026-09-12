# SpatialCrew

**Multi-Agent Framework for Reliable Tool-Augmented Spatial Reasoning**

SpatialCrew is a multi-agent framework that decomposes spatial reasoning tasks into domain-specific sub-tasks, each handled by a specialist agent operating within a structured Write-Execute-Verify-Adjust (WEVA) loop. A Coordinator agent orchestrates task decomposition, cross-agent verification, and answer synthesis.

## Architecture

```
                        ┌─────────────┐
                        │ Coordinator │
                        │   Agent     │
                        └──────┬──────┘
                   ┌───────────┼───────────┐
                   ▼           ▼           ▼
           ┌──────────┐ ┌──────────┐ ┌──────────┐
           │Perception│ │ Geometry │ │ Temporal │
           │Specialist│ │Specialist│ │Specialist│
           └────┬─────┘ └────┬─────┘ └──── ┬────┘
                │            │             │
        ┌───────┴───┐   ┌────┴────┐   ┌────┴────┐
        │SAM3│Mask  │   │Reconstr.│   │Time     │
        │Draw│      │   │Geometry │   │Graph    │
        └───────────┘   └─────────┘   └─────────┘
               GPU Tools (RPC)       CPU Tools (local)
```

<img width="900" height="523" alt="image" src="https://github.com/user-attachments/assets/09a670e2-48d7-4561-aa58-e6300a826327" />


### Key Features

- **Domain-Isolated Specialists**: Each agent only accesses tools within its domain, preventing error cascade across domains.
- **WEVA Loop**: Structured Write-Execute-Verify-Adjust cycle with automatic verification and self-correction.
- **Cross-Agent Verification**: Coordinator checks consistency across specialist results before synthesizing the final answer.
- **Structured Returns**: Every specialist returns `AgentResult` with confidence scores, verification logs, and warnings.
- **Flexible LLM Backend**: Supports both open-source models (via vLLM) and closed-source APIs (OpenAI, etc.).
- **GPU RPC Architecture**: Heavy models (depth estimation, segmentation) run on a separate GPU server with msgpack serialization.

## Project Structure

```
SpatialCrew/
├── configs/
│   ├── models/              # LLM model configurations
│   │   ├── gemma-4-31b.json
│   │   ├── qwen3.6-27b.json
│   │   └── gpt-4o.json
│   └── datasets/            # Benchmark configurations
│       └── spbench.json
├── entrypoints/
│   ├── run.py               # Main evaluation entry point
│   ├── launch_gpu_server.py # GPU model server launcher
│   └── orchestrator.py      # Batch evaluation orchestrator
├── spatialcrew/
│   ├── config.py            # Configuration management
│   ├── coordinator/         # Coordinator agent (planning, dispatch, synthesis)
│   ├── agents/              # Specialist agents (perception, geometry, temporal)
│   ├── tools/               # Domain-isolated tool implementations
│   ├── kernel/              # Jupyter kernel management + security sandbox
│   ├── llm/                 # LLM client + per-agent prompt templates
│   │   └── prompts/         # System prompts for each agent role
│   ├── types/               # Data types (frames, results, spatial)
│   ├── evals/               # Benchmark loaders and metrics
│   ├── gpu_server/          # FastAPI GPU model server
│   └── utils/               # Logging, image utilities
├── scripts/
│   └── setup.sh             # Quick setup script
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### 1. Environment Setup

```bash
# Create and activate a conda environment
conda create -n spatialcrew python=3.11 -y
conda activate spatialcrew

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings
```

### 2. GPU Server Setup (Optional)

If you want to use GPU-accelerated tools (depth estimation, segmentation), start the GPU server on a machine with NVIDIA GPUs:

```bash
# Install GPU dependencies (on the GPU machine)
pip install torch torchvision transformers

# Launch the GPU server
python entrypoints/launch_gpu_server.py --host 0.0.0.0 --port 8190
```

### 3. LLM Backend

**Option A: Open-source models via vLLM**

```bash
# Start vLLM server (example with Gemma)
vllm serve gemma-4-31b --tensor-parallel-size 4

# The framework auto-discovers vLLM endpoints from logs/serve.json
```

**Option B: Closed-source API**

Set your API key in `.env`:
```
SPATIALCREW_LLM_BASE_URL=https://api.openai.com/v1
SPATIALCREW_LLM_API_KEY=sk-...
SPATIALCREW_LLM_MODEL=gpt-4o
```

## Quick Start

### Run SPBench Evaluation

```bash
# Single model evaluation
python entrypoints/run.py \
    --model-config configs/models/gemma-4-31b.json \
    --dataset-config configs/datasets/spbench.json \
    --data-root /path/to/spbench/data \
    --subsets si mv \
    --work-dir ./outputs

# Batch evaluation (all models x all datasets)
python entrypoints/orchestrator.py \
    --model-configs configs/models \
    --dataset-configs configs/datasets \
    --work-dir ./outputs
```

### Expected Data Layout

SPBench data should be organized as:
```
/path/to/spbench/data/
├── spbench_si.json      # Single-image subset annotations
├── spbench_mv.json      # Multi-view subset annotations
└── images/              # Image files referenced in annotations
```

Each annotation JSON should contain a list of samples:
```json
[
    {
        "id": "si_001",
        "question": "Which object is closer to the camera?",
        "image": "images/scene_001.jpg",
        "choices": ["A. The chair", "B. The table", "C. The lamp", "D. The sofa"],
        "answer": "B"
    }
]
```

## Configuration

### Model Config

Model configs control LLM parameters per role:

```json
{
    "llm_model": "gemma-4-31b",
    "llm_base_url": "vllm",
    "roles": {
        "coordinator": { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "specialist": { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "vlm": { "max_tokens": 32768, "temperature": 0.6, "enable_thinking": true }
    }
}
```

### Environment Variables

All settings can be overridden via environment variables with the `SPATIALCREW_` prefix:

| Variable | Description | Default |
|---|---|---|
| `SPATIALCREW_LLM_BASE_URL` | LLM endpoint (`vllm` for auto-discovery) | `vllm` |
| `SPATIALCREW_LLM_API_KEY` | API key for the LLM service | `bearer` |
| `SPATIALCREW_LLM_MODEL` | Model name | `gemma-4-31b` |
| `SPATIALCREW_GPU_SERVER_PORT` | GPU server port | `8190` |
| `SPATIALCREW_CONCURRENCY` | Parallel evaluation workers | `4` |
| `SPATIALCREW_MAX_STEPS` | Max WEVA steps per specialist | `30` |

## Evaluation Metrics

- **MCA (Accuracy)**: Exact-match accuracy after answer normalization.
- **NA (MRA)**: Normalized Accuracy that adjusts for random guessing baseline, computed as `(acc - 1/N) / (1 - 1/N)` per sample.
- **Per-category metrics**: Accuracy and MRA broken down by question type.

## Adding New Benchmarks

1. Create a new loader in `spatialcrew/evals/` that extends `BaseBenchmark`.
2. Implement `load()`, `__len__()`, `__iter__()`, and `evaluate()` methods.
3. Add a dataset config JSON in `configs/datasets/`.

## Adding New Tools

1. Create a tool class in `spatialcrew/tools/` extending `GPUTool` or `CPUTool`.
2. Define `TOOL_NAME`, `DOMAIN`, and `TOOL_PROMPT_DESCRIPTION`.
3. Register the tool in `spatialcrew/tools/registry.py` under the appropriate domain.

## Citation

```bibtex
@inproceedings{spatialcrew2027,
    title={SpatialCrew: Multi-Agent Framework for Reliable Tool-Augmented Spatial Reasoning},
    booktitle={ICLR},
    year={2027}
}
```

