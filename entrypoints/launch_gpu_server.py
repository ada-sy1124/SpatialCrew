"""Launch the SpatialCrew GPU model server."""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialcrew.gpu_server.server import app, register_service
from spatialcrew.gpu_server.models import DepthEstimationService, SegmentationService

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SpatialCrew GPU Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--depth-model", type=str, default="Depth-Anything-V2")
    parser.add_argument("--seg-model", type=str, default="sam3-large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    depth_service = DepthEstimationService(model_path=args.depth_model, device=args.device)
    seg_service = SegmentationService(model_path=args.seg_model, device=args.device)

    register_service("spatial_Reconstruct", depth_service)
    register_service("spatial_SAM3", seg_service)

    registry_path = Path(args.log_dir) / "gpu_server.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump([{"url": f"http://{args.host}:{args.port}"}], f)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
