"""Graph and visualization utilities for temporal analysis."""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from spatialcrew.tools.base import CPUTool


class GraphDrawer(CPUTool):
    """CPU-based plotting utility for temporal data visualization."""
    TOOL_NAME = "Graph"
    DOMAIN = "temporal"
    TOOL_PROMPT_DESCRIPTION = """### tools.Graph - Plotting Utilities

**tools.Graph.plot(values, xlabel="Frame", ylabel="Value", title="") -> Image**
Plot a 1D time series as a line chart. Returns a PIL Image.

**tools.Graph.plot_multiple(series_dict, xlabel="Frame", ylabel="Value", title="") -> Image**
Plot multiple named series on the same chart.
"""

    @staticmethod
    def plot(values, xlabel="Frame", ylabel="Value", title=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(len(values)), values, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def plot_multiple(series_dict, xlabel="Frame", ylabel="Value", title=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        fig, ax = plt.subplots(figsize=(8, 4))
        for name, values in series_dict.items():
            ax.plot(np.arange(len(values)), values, label=name, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
