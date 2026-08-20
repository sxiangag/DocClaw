"""OCR tool abstractions and implementations."""

from docclaw.agent.tool.ocr.assemble import TranscribeTool
from docclaw.agent.tool.ocr.ocr import OcrTool
from docclaw.agent.tool.ocr.paddleocr import PaddleOCRTool
from docclaw.agent.tool.ocr.paddleocrvl import PaddleOCRVLTool
from docclaw.agent.tool.ocr.vlm import VLMOcrTool

__all__ = [
    "TranscribeTool",
    "OcrTool",
    "PaddleOCRTool",
    "PaddleOCRVLTool",
    "VLMOcrTool",
]
