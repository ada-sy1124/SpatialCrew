"""Configuration management for SpatialCrew."""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns in a string."""
    def _replacer(m):
        var = m.group(1)
        default = m.group(3)
        return os.environ.get(var, default if default is not None else m.group(0))
    return re.sub(r"\$\{(\w+)(:-(.*?))?\}", _replacer, value)


@dataclass
class LLMRoleParams:
    """Parameters for a specific LLM role (coordinator, specialist, vlm)."""
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = True


@dataclass
class SpatialCrewConfig:
    """Main configuration for the SpatialCrew framework."""

    # Benchmark
    benchmark: str = "spbench"
    data_root: str = ""
    subsets: list = field(default_factory=lambda: ["si", "mv"])
    num_key_frames: int = 16

    # LLM
    llm_model: str = "gemma-4-31b"
    llm_base_url: str = "vllm"
    llm_api_key: str = "bearer"

    # Role-specific LLM params
    coordinator_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    specialist_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    vlm_params: LLMRoleParams = field(default_factory=LLMRoleParams)

    # Execution
    max_steps: int = 30
    max_failures: int = 5
    max_dispatch_rounds: int = 3
    concurrency: int = 4

    # Paths
    work_dir: str = "./outputs"
    log_dir: str = "./logs"

    # GPU server
    gpu_server_host: str = "0.0.0.0"
    gpu_server_port: int = 8190
    reconstruct_backend: str = "da3"

    # Features
    enable_cross_verification: bool = True
    enable_planning: bool = True
    video_max_fps: float = 2.0

    @classmethod
    def from_files(
        cls,
        model_json: Optional[str] = None,
        dataset_json: Optional[str] = None,
        **overrides,
    ) -> "SpatialCrewConfig":
        """Build config from optional JSON files + CLI overrides."""
        merged: Dict[str, Any] = {}

        # Load dataset config
        if dataset_json and Path(dataset_json).exists():
            with open(dataset_json) as f:
                merged.update(json.load(f))

        # Load model config
        roles = {}
        if model_json and Path(model_json).exists():
            with open(model_json) as f:
                model_cfg = json.load(f)
                roles = model_cfg.pop("roles", {})
                merged.update(model_cfg)

        # Apply env vars (prefix: SPATIALCREW_)
        env_prefix = "SPATIALCREW_"
        for f in fields(cls):
            env_key = env_prefix + f.name.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                merged.setdefault(f.name, env_val)

        # Apply CLI overrides (highest priority)
        merged.update({k: v for k, v in overrides.items() if v is not None})

        # Expand env vars in string values
        for k, v in merged.items():
            if isinstance(v, str):
                merged[k] = _expand_env(v)

        # Build role params
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in merged.items() if k in config_fields}

        cfg = cls(**kwargs)

        # Apply role-specific LLM params
        if "coordinator" in roles:
            cfg.coordinator_params = LLMRoleParams(**roles["coordinator"])
        if "specialist" in roles:
            cfg.specialist_params = LLMRoleParams(**roles["specialist"])
        if "vlm" in roles:
            cfg.vlm_params = LLMRoleParams(**roles["vlm"])

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary."""
        from dataclasses import asdict
        return asdict(self)
