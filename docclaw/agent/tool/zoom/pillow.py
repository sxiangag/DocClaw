"""Pillow-backed zoom renderer."""

from __future__ import annotations

from pathlib import Path

from docclaw.agent.tool.zoom.zoom import PixelBBox, ZoomTool, clamp_bbox


class PillowZoomTool(ZoomTool):
    """Create zoom-view artifacts from raster page images with Pillow."""

    def render_zoom_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
        target_long_side_px: int,
    ) -> tuple[int, int]:
        from PIL import Image, ImageFilter

        with Image.open(source_path) as image:
            width, height = image.size
            x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox is outside the page image")

            zoom_view = image.crop((x0, y0, x1, y1))
            current_long_side = max(zoom_view.width, zoom_view.height)
            if current_long_side <= 0:
                raise ValueError("zoom crop has invalid size")
            scale = float(target_long_side_px) / float(current_long_side)
            if scale != 1.0:
                output_width = max(1, int(round(zoom_view.width * scale)))
                output_height = max(1, int(round(zoom_view.height * scale)))
                zoom_view = zoom_view.resize(
                    (output_width, output_height),
                    resample=Image.Resampling.LANCZOS,
                )

            zoom_view = zoom_view.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=50, threshold=3)
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            zoom_view.save(output_path)
            return zoom_view.size
