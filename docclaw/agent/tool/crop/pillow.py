"""Pillow-backed crop renderer."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.tool.crop.crop import CropTool, PixelBBox
from docclaw.agent.tool.zoom.zoom import clamp_bbox


class PillowCropTool(CropTool):
    """Create crop-expanded view artifacts from raster page images with Pillow."""

    def render_crop_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
    ) -> tuple[int, int]:
        from PIL import Image

        with Image.open(source_path) as image:
            width, height = image.size
            x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox is outside the page image")

            crop_view = image.crop((x0, y0, x1, y1))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop_view.save(output_path)
            return crop_view.size
