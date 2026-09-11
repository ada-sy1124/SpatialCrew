"""GPU RPC client for communicating with the GPU model server."""

from __future__ import annotations
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _encode_value(obj: Any) -> Any:
    """Encode Python objects for msgpack transport."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj)
        return {"__type__": "ndarray", "data": buf.getvalue()}
    elif isinstance(obj, Image.Image):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return {"__type__": "pil_image", "data": buf.getvalue()}
    elif isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _encode_value(v) for k, v in obj.items()}
    return obj


def _decode_value(obj: Any) -> Any:
    """Decode msgpack-transported objects back to Python."""
    if isinstance(obj, dict):
        if obj.get("__type__") == "ndarray":
            buf = io.BytesIO(obj["data"])
            return np.load(buf, allow_pickle=False)
        elif obj.get("__type__") == "pil_image":
            return Image.open(io.BytesIO(obj["data"])).convert("RGB")
        return {k: _decode_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_decode_value(v) for v in obj]
    return obj


class GPURPCClient:
    """Client for the SpatialCrew GPU model server."""

    def __init__(self, config):
        self._config = config
        self._server_url: Optional[str] = None
        self._token: str = "spatialcrew-gpu"
        self._last_discovery = 0.0

    def _discover_server(self) -> str:
        """Find the GPU server URL from the service registry."""
        registry = Path(self._config.log_dir) / "gpu_server.json"
        if registry.exists():
            with open(registry) as f:
                info = json.load(f)
            if isinstance(info, list) and info:
                info = info[0]
            url = info.get("url", "")
            if url:
                self._server_url = url
                return url

        host = self._config.gpu_server_host
        port = self._config.gpu_server_port
        self._server_url = f"http://{host}:{port}"
        return self._server_url

    async def call(self, service: str, method: str, **kwargs) -> Any:
        """Make an RPC call to the GPU server."""
        url = self._server_url or self._discover_server()

        payload = _encode_value({"service": service, "method": method, "kwargs": kwargs})
        packed = msgpack.packb(payload, use_bin_type=True)

        try:
            resp = requests.post(
                f"{url}/call",
                data=packed,
                headers={
                    "Content-Type": "application/msgpack",
                    "Authorization": f"Bearer {self._token}",
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = msgpack.unpackb(resp.content, raw=False)
            return _decode_value(result)
        except Exception as e:
            logger.error(f"GPU RPC call failed ({service}.{method}): {e}")
            raise
