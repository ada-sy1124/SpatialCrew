"""Frame and image container types."""

from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
import numpy as np


@dataclass
class FrameImage:
    """A single image frame with optional index metadata."""
    image: Image.Image
    frame_index: int = 0
    _path: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_path(cls, path: str, frame_index: int = 0) -> "FrameImage":
        img = Image.open(path).convert("RGB")
        return cls(image=img, frame_index=frame_index, _path=path)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, frame_index: int = 0) -> "FrameImage":
        img = Image.fromarray(arr)
        return cls(image=img, frame_index=frame_index)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def to_numpy(self) -> np.ndarray:
        return np.array(self.image)

    def to_bytes(self, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format=fmt)
        return buf.getvalue()

    def resize(self, width: int, height: int) -> "FrameImage":
        resized = self.image.resize((width, height), Image.LANCZOS)
        return FrameImage(image=resized, frame_index=self.frame_index)


class InputImages(list):
    """Ordered list of FrameImage objects representing visual input to an agent."""

    @property
    def frame_indices(self) -> List[int]:
        return [f.frame_index for f in self]

    @property
    def pil_images(self) -> List[Image.Image]:
        return [f.image for f in self]

    def __repr__(self) -> str:
        indices = self.frame_indices
        return f"InputImages(n={len(self)}, frames={indices[:5]}{'...' if len(indices) > 5 else ''})"
