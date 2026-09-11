"""Drawing and visualization utilities."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatialcrew.tools.base import CPUTool


class DrawUtils(CPUTool):
    """CPU-based drawing and annotation utilities."""
    TOOL_NAME = "Draw"
    DOMAIN = "perception"
    TOOL_PROMPT_DESCRIPTION = """### tools.Draw - Drawing Utilities

**tools.Draw.draw_point(image, point, color="red", radius=5) -> Image**
Draw a point on the image. `point` is (x, y) in pixel coords.

**tools.Draw.draw_box(image, box, color="green", width=2) -> Image**
Draw a bounding box [x1, y1, x2, y2] on the image.

**tools.Draw.draw_mask_overlay(image, mask, color="blue", alpha=0.4) -> Image**
Overlay a boolean mask on the image with transparency.

**tools.Draw.draw_text(image, text, position, color="white", font_size=16) -> Image**
Draw text at the given position on the image.
"""

    @staticmethod
    def draw_point(image, point, color="red", radius=5):
        img = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        x, y = point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        return img

    @staticmethod
    def draw_box(image, box, color="green", width=2):
        img = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline=color, width=width)
        return img

    @staticmethod
    def draw_mask_overlay(image, mask, color="blue", alpha=0.4):
        img = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        color_map = {"blue": (0, 0, 255), "red": (255, 0, 0), "green": (0, 255, 0)}
        rgb = color_map.get(color, (0, 0, 255))
        mask_arr = np.array(mask, dtype=bool)
        overlay_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
        overlay_arr[mask_arr] = (*rgb, int(255 * alpha))
        overlay = Image.fromarray(overlay_arr, "RGBA")
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    @staticmethod
    def draw_text(image, text, position, color="white", font_size=16):
        img = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text(position, text, fill=color)
        return img
