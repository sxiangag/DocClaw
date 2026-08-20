"""Zoom tool abstractions and implementations."""

from docclaw.agent.tool.zoom.pillow import PillowZoomTool
from docclaw.agent.tool.zoom.zoom import PixelBBox, ZoomTool

__all__ = [
    "PillowZoomTool",
    "PixelBBox",
    "ZoomTool",
]
