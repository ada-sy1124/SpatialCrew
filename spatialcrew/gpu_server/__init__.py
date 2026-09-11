"""GPU model server for SpatialCrew."""

from spatialcrew.gpu_server.server import app, register_service
from spatialcrew.gpu_server.models import DepthEstimationService, SegmentationService

__all__ = ["app", "register_service", "DepthEstimationService", "SegmentationService"]
