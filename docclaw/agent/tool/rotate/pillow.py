"""Pillow-backed rotate renderer."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.tool.rotate.rotate import RotateTool
from docclaw.agent.tool.zoom.zoom import PixelBBox, clamp_bbox


class PillowRotateTool(RotateTool):
    """Create rotated view artifacts from raster page images with Pillow."""

    def render_rotate_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
        angle_degree: float,
    ) -> tuple[int, int]:
        from PIL import Image

        with Image.open(source_path) as image:
            width, height = image.size
            x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox is outside the page image")

            rotate_view = image.crop((x0, y0, x1, y1))
            if angle_degree:
                rotate_view = rotate_view.rotate(-angle_degree, expand=True)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            rotate_view.save(output_path)
            return rotate_view.size
