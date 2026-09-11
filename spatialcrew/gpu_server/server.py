"""FastAPI GPU model server for SpatialCrew."""

from __future__ import annotations
import io
import logging
import time
from typing import Any, Dict

import msgpack
import numpy as np
from fastapi import FastAPI, Request, Response
from PIL import Image

logger = logging.getLogger(__name__)

app = FastAPI(title="SpatialCrew GPU Server")

_services: Dict[str, Any] = {}


def register_service(name: str, service: Any) -> None:
    """Register a GPU model service."""
    _services[name] = service
    logger.info(f"Registered GPU service: {name}")


def _decode_value(obj: Any) -> Any:
    """Decode msgpack-transported objects."""
    if isinstance(obj, dict):
        if obj.get(b"__type__") == b"ndarray" or obj.get("__type__") == "ndarray":
            data = obj.get(b"data", obj.get("data"))
            buf = io.BytesIO(data)
            return np.load(buf, allow_pickle=False)
        elif obj.get(b"__type__") == b"pil_image" or obj.get("__type__") == "pil_image":
            data = obj.get(b"data", obj.get("data"))
            return Image.open(io.BytesIO(data)).convert("RGB")
        decoded = {}
        for k, v in obj.items():
            key = k.decode() if isinstance(k, bytes) else k
            decoded[key] = _decode_value(v)
        return decoded
    elif isinstance(obj, (list, tuple)):
        return [_decode_value(v) for v in obj]
    elif isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj
    return obj


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
    elif isinstance(obj, dict):
        return {k: _encode_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_encode_value(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


@app.post("/call")
async def handle_call(request: Request) -> Response:
    """Handle an RPC call from the agent."""
    body = await request.body()
    payload = msgpack.unpackb(body, raw=False)
    payload = _decode_value(payload)

    service_name = payload.get("service", "")
    method_name = payload.get("method", "")
    kwargs = payload.get("kwargs", {})

    if service_name not in _services:
        return Response(
            content=msgpack.packb({"error": f"Unknown service: {service_name}"}, use_bin_type=True),
            media_type="application/msgpack",
            status_code=404,
        )

    service = _services[service_name]
    method = getattr(service, method_name, None)
    if method is None:
        return Response(
            content=msgpack.packb({"error": f"Unknown method: {method_name}"}, use_bin_type=True),
            media_type="application/msgpack",
            status_code=404,
        )

    t0 = time.monotonic()
    try:
        result = method(**kwargs)
        elapsed = time.monotonic() - t0
        logger.info(f"{service_name}.{method_name}: {elapsed:.2f}s")
        encoded = _encode_value(result)
        return Response(
            content=msgpack.packb(encoded, use_bin_type=True),
            media_type="application/msgpack",
        )
    except Exception as e:
        logger.error(f"{service_name}.{method_name} failed: {e}")
        return Response(
            content=msgpack.packb({"error": str(e)}, use_bin_type=True),
            media_type="application/msgpack",
            status_code=500,
        )


@app.get("/health")
async def health():
    return {"status": "ok", "services": list(_services.keys())}
