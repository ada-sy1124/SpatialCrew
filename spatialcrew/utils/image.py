"""Image utility functions."""

from __future__ import annotations
import io
import base64
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    """Convert base64 string to PIL image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_image(
    img: Image.Image,
    max_size: int = 1024,
    keep_aspect: bool = True,
) -> Image.Image:
    """Resize image to fit within max_size while preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if keep_aspect:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = max_size, max_size
    return img.resize((new_w, new_h), Image.LANCZOS)


def extract_video_frames(
    video_path: str,
    num_frames: int = 16,
    max_fps: float = 2.0,
) -> list:
    """Extract key frames from a video file using ffmpeg."""
    import subprocess
    import tempfile
    from pathlib import Path

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={max_fps}",
            "-q:v", "2",
            f"{tmpdir}/frame_%06d.jpg",
            "-hide_banner", "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True)

        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if len(frame_files) > num_frames:
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        for i, fp in enumerate(frame_files):
            img = Image.open(fp).convert("RGB")
            from spatialcrew.types.frames import FrameImage
            frames.append(FrameImage(image=img, frame_index=i))

    return frames
