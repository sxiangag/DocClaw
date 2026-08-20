"""Built-in document action tools."""

from docclaw.agent.tool.tool import (
    StopTool,
    Tool,
    ToolRegistry,
    build_default_tool_registry,
)
from docclaw.agent.tool.answer import AnswerTool, LLMAnswerTool, LLMJsonAnswerTool
from docclaw.agent.tool.chart import ChartTool, PaddleOCRVLChartTool
from docclaw.agent.tool.enhancement import EnhancementTool
from docclaw.agent.tool.evidence import EvidenceTool, LLMEvidenceTool
from docclaw.agent.tool.figure import FigureTool, VLMFigureTool
from docclaw.agent.tool.formula import FormulaTool
from docclaw.agent.tool.formula import MinerUFormulaTool
from docclaw.agent.tool.formula import PaddleOCRVLFormulaTool
from docclaw.agent.tool.internal_search import (
    ColBERTTextEmbedder,
    ColPaliImageEmbedder,
    ImageSemanticRetriever,
    InternalSearchTool,
    InternalSearchToolBase,
    KeywordRetriever,
    TextSemanticRetriever,
)
from docclaw.agent.tool.layout import LayoutTool, PPDocLayoutTool
from docclaw.agent.tool.crop import CropTool, PillowCropTool
from docclaw.agent.tool.ocr import (
    OcrTool,
    PaddleOCRTool,
    PaddleOCRVLTool,
    TranscribeTool,
)
from docclaw.agent.tool.inspect_ocr import LLMInspectOcrTool
from docclaw.agent.tool.select_ocr import LLMSelectOcrTool
from docclaw.agent.tool.select_pages import SelectPagesTool, VLMSelectPagesTool
from docclaw.agent.tool.rotate import PillowRotateTool, RotateTool
from docclaw.agent.tool.table import PPStructureTableTool, PaddleOCRVLTableTool, TableTool
from docclaw.agent.tool.zoom import PillowZoomTool, ZoomTool

__all__ = [
    "AnswerTool",
    "ChartTool",
    "EnhancementTool",
    "EvidenceTool",
    "FigureTool",
    "FormulaTool",
    "MinerUFormulaTool",
    "PaddleOCRVLFormulaTool",
    "PaddleOCRVLChartTool",
    "InternalSearchTool",
    "InternalSearchToolBase",
    "ColPaliImageEmbedder",
    "ImageSemanticRetriever",
    "KeywordRetriever",
    "ColBERTTextEmbedder",
    "LayoutTool",
    "CropTool",
    "LLMAnswerTool",
    "LLMJsonAnswerTool",
    "LLMEvidenceTool",
    "LLMSelectOcrTool",
    "OcrTool",
    "PaddleOCRTool",
    "PaddleOCRVLTool",
    "SelectPagesTool",
    "LLMInspectOcrTool",
    "VLMSelectPagesTool",
    "TranscribeTool",
    "PillowCropTool",
    "PillowZoomTool",
    "PPDocLayoutTool",
    "PaddleOCRVLTableTool",
    "PillowRotateTool",
    "StopTool",
    "PPStructureTableTool",
    "RotateTool",
    "TextSemanticRetriever",
    "TableTool",
    "Tool",
    "ToolRegistry",
    "VLMFigureTool",
    "ZoomTool",
    "build_default_tool_registry",
]
