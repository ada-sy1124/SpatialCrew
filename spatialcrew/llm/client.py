"""OpenAI-compatible LLM client with vLLM auto-discovery support."""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client supporting vLLM local deployment and cloud APIs."""

    def __init__(self, config):
        self._config = config
        self._model = config.llm_model
        self._base_url = config.llm_base_url
        self._api_key = config.llm_api_key

        self._clients: List[AsyncOpenAI] = []
        self._last_discovery = 0.0
        self._discovery_interval = 60.0
        self._is_vllm = (self._base_url == "vllm")

        if not self._is_vllm:
            self._clients = [
                AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
            ]

    async def _discover_vllm_endpoints(self) -> None:
        """Discover vLLM endpoints from service registry."""
        now = time.monotonic()
        if self._clients and (now - self._last_discovery) < self._discovery_interval:
            return

        registry_path = Path(self._config.log_dir) / "serve.json"
        if not registry_path.exists():
            if not self._clients:
                raise RuntimeError(
                    f"vLLM service registry not found at {registry_path}. "
                    "Start vLLM first or configure a direct LLM base URL."
                )
            return

        try:
            with open(registry_path) as f:
                endpoints = json.load(f)

            new_clients = []
            for ep in endpoints:
                url = ep.get("url", "")
                if url:
                    new_clients.append(
                        AsyncOpenAI(base_url=url, api_key=self._api_key)
                    )

            if new_clients:
                self._clients = new_clients
                self._last_discovery = now
                logger.info(f"Discovered {len(new_clients)} vLLM endpoint(s).")
        except Exception as e:
            logger.warning(f"Failed to discover vLLM endpoints: {e}")

    def _select_client(self, session_id: Optional[str] = None) -> AsyncOpenAI:
        """Select a client, using sticky routing for session-based cache hits."""
        if not self._clients:
            raise RuntimeError("No LLM clients available.")
        if len(self._clients) == 1:
            return self._clients[0]
        if session_id:
            idx = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % len(self._clients)
            return self._clients[idx]
        return self._clients[0]

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        role: str = "specialist",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a completion from the LLM."""
        if self._is_vllm:
            await self._discover_vllm_endpoints()

        role_params = self._get_role_params(role)
        client = self._select_client(session_id)

        request_kwargs = {
            "model": self._model,
            "messages": messages,
            "max_tokens": role_params.max_tokens,
            "temperature": role_params.temperature,
            "top_p": role_params.top_p,
        }

        if self._is_vllm:
            extra_body = {}
            if role_params.top_k > 0:
                extra_body["top_k"] = role_params.top_k
            if role_params.min_p > 0:
                extra_body["min_p"] = role_params.min_p
            if role_params.repetition_penalty != 1.0:
                extra_body["repetition_penalty"] = role_params.repetition_penalty
            if role_params.enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            if extra_body:
                request_kwargs["extra_body"] = extra_body

        request_kwargs.update(kwargs)

        try:
            response = await client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_vision(
        self,
        text: str,
        images: List[Any],
        system_prompt: str = "",
        role: str = "vlm",
        **kwargs,
    ) -> str:
        """Generate a completion with image inputs."""
        import base64
        import io

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content_parts = [{"type": "text", "text": text}]
        for img in images:
            if hasattr(img, "image"):
                pil_img = img.image
            else:
                pil_img = img
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content_parts})
        return await self.generate(messages, role=role, **kwargs)

    def _get_role_params(self, role: str):
        """Get LLM parameters for the given role."""
        mapping = {
            "coordinator": self._config.coordinator_params,
            "specialist": self._config.specialist_params,
            "vlm": self._config.vlm_params,
        }
        return mapping.get(role, self._config.specialist_params)
