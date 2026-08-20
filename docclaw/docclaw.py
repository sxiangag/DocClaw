"""High-level application facade for DocClaw."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docclaw.agent.executor import Executor
from docclaw.agent.loop import AgentLoop, RunEventCallback
from docclaw.agent.planner import LLMPlanner, Planner
from docclaw.agent.runner import DocClawRunner
from docclaw.agent.tool.answer import LLMAnswerTool, LLMJsonAnswerTool
from docclaw.agent.tool.answer.answer import ExplicitAnswerOnlyTool
from docclaw.agent.tool.chart import PaddleOCRVLChartTool, VLMChartTool
from docclaw.agent.tool.enhancement import EnhancementTool
from docclaw.agent.tool.evidence import LLMEvidenceTool
from docclaw.agent.tool.figure import VLMFigureTool
from docclaw.agent.tool.formula import PaddleOCRVLFormulaTool, VLMFormulaTool
from docclaw.agent.tool.internal_search import InternalSearchTool
from docclaw.agent.tool.layout import PPDocLayoutTool
from docclaw.agent.tool.crop import PillowCropTool
from docclaw.agent.tool.ocr import (
    PaddleOCRTool,
    PaddleOCRVLTool,
    TranscribeTool,
    VLMOcrTool,
)
from docclaw.agent.tool.inspect_ocr import LLMInspectOcrTool
from docclaw.agent.tool.select_ocr import LLMSelectOcrTool
from docclaw.agent.tool.select_pages import VLMSelectPagesTool
from docclaw.agent.tool.rotate import PillowRotateTool
from docclaw.agent.tool.table import PaddleOCRVLTableTool, VLMTableTool
from docclaw.agent.tool.tool import StopTool, ToolRegistry
from docclaw.agent.tool.zoom import PillowZoomTool
from docclaw.agent.utils import DocumentState, RunResult, Task
from docclaw.config.loader import get_config_path, load_config
from docclaw.config.paths import (
    resolve_document_artifact_dir,
    resolve_sessions_dir,
    resolve_skills_dir,
)
from docclaw.config.schema import DocClawConfig
from docclaw.document import load_document as load_document_state
from docclaw.provider import LLMProvider, make_named_provider
from docclaw.session.manager import Session, SessionManager
from docclaw.skills import TaskSkillsLoader


@dataclass(slots=True)
class DocClaw:
    """Application facade assembled from runtime config."""

    config: DocClawConfig
    providers: dict[str, LLMProvider]
    planner: Planner
    runner: DocClawRunner
    config_path: Path

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> DocClaw:
        config_path = get_config_path(path)
        config = load_config(config_path)
        providers = _build_providers(config)
        planner_provider = providers[config.planner.provider]
        tools = _build_tools(providers, config)
        planner = LLMPlanner(
            planner_provider,
            tools=tools,
            model=config.planner.model or config.providers.require(config.planner.provider).model,
            temperature=config.planner.temperature,
            task_skills_loader=(
                TaskSkillsLoader(
                    workspace_skills_dir=resolve_skills_dir(config, config_path=config_path),
                )
                if config.skills.enabled
                else None
            ),
        )
        runner = DocClawRunner(
            loop=AgentLoop(executor=Executor(tools=tools)),
            session_manager=SessionManager(resolve_sessions_dir(config, config_path=config_path)),
            session_max_messages=config.runtime.session_max_messages,
        )
        return cls(
            config=config,
            providers=providers,
            planner=planner,
            runner=runner,
            config_path=config_path,
        )

    def load_document(
        self,
        path: str | Path,
        *,
        document_id: str | None = None,
    ) -> DocumentState:
        resolved_document_id = document_id or _document_id_hint(path)
        artifact_dir = resolve_document_artifact_dir(
            self.config,
            resolved_document_id,
            config_path=self.config_path,
        )
        document = load_document_state(
            path,
            artifact_dir=artifact_dir,
            document_id=resolved_document_id,
        )
        document.metadata.setdefault("artifact_dir", str(artifact_dir))
        return document

    async def run(
        self,
        document: DocumentState,
        task: Task | str,
        *,
        max_steps: int | None = None,
        session: Session | None = None,
        session_id: str | None = None,
        planner: Planner | None = None,
        on_event: RunEventCallback | None = None,
    ) -> RunResult:
        artifact_dir = resolve_document_artifact_dir(
            self.config,
            document.document_id,
            config_path=self.config_path,
        )
        document.metadata.setdefault("artifact_dir", str(artifact_dir))
        return await self.runner.run(
            document,
            task,
            planner or self.planner,
            max_steps=max_steps or self.config.runtime.max_steps,
            session=session,
            session_id=session_id,
            on_event=on_event,
        )


def _build_providers(config: DocClawConfig) -> dict[str, LLMProvider]:
    return {
        name: make_named_provider(config.providers, name)
        for name in config.providers.entries
    }


def _build_tools(providers: dict[str, LLMProvider], config: DocClawConfig) -> ToolRegistry:
    registry = ToolRegistry()
    layout_tool: PPDocLayoutTool | None = None
    crop_tool: PillowCropTool | None = None
    rotate_tool: PillowRotateTool | None = None
    zoom_tool: PillowZoomTool | None = None
    ocr_tool: PaddleOCRTool | PaddleOCRVLTool | VLMOcrTool | None = None

    if config.tools.layout.enabled:
        layout_tool = PPDocLayoutTool(
            pipeline_kwargs=_pipeline_kwargs(device=config.tools.layout.device),
        )
        registry.register(layout_tool)
    if config.tools.formula.enabled:
        if config.tools.formula.backend == "vlm":
            provider = providers[config.tools.formula.provider]
            registry.register(
                VLMFormulaTool(
                    provider,
                    model=config.tools.formula.model,
                    max_tokens=config.tools.formula.max_tokens,
                    temperature=config.tools.formula.temperature,
                )
            )
        else:
            registry.register(
                PaddleOCRVLFormulaTool(
                    pipeline_kwargs=_pipeline_kwargs(device=config.tools.formula.device),
                )
            )
    if config.tools.chart.enabled:
        if config.tools.chart.backend == "vlm":
            provider = providers[config.tools.chart.provider]
            registry.register(
                VLMChartTool(
                    provider,
                    model=config.tools.chart.model,
                    max_tokens=config.tools.chart.max_tokens,
                    temperature=config.tools.chart.temperature,
                )
            )
        else:
            registry.register(
                PaddleOCRVLChartTool(
                    pipeline_kwargs=_pipeline_kwargs(device=config.tools.chart.device),
                )
            )
    if config.tools.table.enabled:
        if config.tools.table.backend == "vlm":
            provider = providers[config.tools.table.provider]
            registry.register(
                VLMTableTool(
                    provider,
                    model=config.tools.table.model,
                    max_tokens=config.tools.table.max_tokens,
                    temperature=config.tools.table.temperature,
                )
            )
        else:
            registry.register(
                PaddleOCRVLTableTool(
                    pipeline_kwargs=_pipeline_kwargs(device=config.tools.table.device),
                )
            )
    if config.tools.crop.enabled:
        crop_tool = PillowCropTool()
        registry.register(crop_tool)
    if config.tools.zoom.enabled:
        zoom_tool = PillowZoomTool()
        registry.register(zoom_tool)
    if config.tools.ocr_enhancement.enabled:
        rotate_tool = PillowRotateTool()
        registry.register(rotate_tool)
    if config.tools.ocr.enabled:
        ocr_pipeline_kwargs = _pipeline_kwargs(device=config.tools.ocr.device)
        if config.tools.ocr.backend == "vlm":
            provider = providers[config.tools.ocr.provider]
            ocr_tool = VLMOcrTool(
                provider,
                model=config.tools.ocr.model,
                max_tokens=config.tools.ocr.max_tokens,
                temperature=config.tools.ocr.temperature,
                allow_chart_region_ocr=config.tools.chart.enabled,
            )
            registry.register(ocr_tool)
        elif config.tools.ocr.backend == "paddleocr_vl":
            ocr_tool = PaddleOCRVLTool(
                pipeline_kwargs=ocr_pipeline_kwargs,
                allow_chart_region_ocr=config.tools.chart.enabled,
            )
            registry.register(ocr_tool)
        else:
            ocr_tool = PaddleOCRTool(
                pipeline_kwargs=ocr_pipeline_kwargs,
            )
            registry.register(ocr_tool)
        if config.tools.ocr_enhancement.enabled and (
            zoom_tool is not None or crop_tool is not None or rotate_tool is not None
        ):
            registry.register(
                EnhancementTool(
                    ocr_tool,
                    zoom_tool=zoom_tool,
                    crop_tool=crop_tool,
                    rotate_tool=rotate_tool,
                )
            )
        registry.register(TranscribeTool())
    if config.tools.internal_search.enabled:
        registry.register(
            InternalSearchTool(
                layout_tool=layout_tool,
                ocr_tool=ocr_tool,
            )
        )
    if config.tools.figure.enabled:
        provider = providers[config.tools.figure.provider]
        registry.register(
            VLMFigureTool(
                provider,
                model=config.tools.figure.model,
                temperature=config.tools.figure.temperature,
                layout_tool=layout_tool,
            )
        )
    if config.tools.select_pages.enabled:
        provider = providers[config.tools.select_pages.provider]
        registry.register(
            VLMSelectPagesTool(
                provider,
                model=config.tools.select_pages.model,
                temperature=config.tools.select_pages.temperature,
                layout_tool=layout_tool,
            )
        )
    if config.tools.evidence.enabled:
        provider = providers[config.tools.evidence.provider]
        registry.register(
            LLMEvidenceTool(
                provider,
                model=config.tools.evidence.model,
                temperature=config.tools.evidence.temperature,
            )
        )
    if config.tools.inspect_ocr.enabled:
        provider = providers[config.tools.inspect_ocr.provider]
        registry.register(
            LLMInspectOcrTool(
                provider,
                model=config.tools.inspect_ocr.model,
                temperature=config.tools.inspect_ocr.temperature,
                default_max_actions=config.tools.inspect_ocr.max_refine_regions,
            )
        )
    if config.tools.select_ocr.enabled:
        provider = providers[config.tools.select_ocr.provider]
        registry.register(
            LLMSelectOcrTool(
                provider,
                model=config.tools.select_ocr.model,
                temperature=config.tools.select_ocr.temperature,
            )
        )
    if config.tools.answer_from_evidence.enabled:
        provider = providers[config.tools.answer_from_evidence.provider]
        registry.register(
            LLMAnswerTool(
                provider,
                model=config.tools.answer_from_evidence.model,
                temperature=config.tools.answer_from_evidence.temperature,
            )
        )
    else:
        registry.register(ExplicitAnswerOnlyTool())
    if config.tools.answer_json.enabled:
        provider = providers[config.tools.answer_json.provider]
        registry.register(
            LLMJsonAnswerTool(
                provider,
                model=config.tools.answer_json.model,
                temperature=config.tools.answer_json.temperature,
            )
        )
    registry.register(StopTool())
    return registry


def _document_id_hint(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in candidate.stem).strip("._")
    return stem or "document"


def _pipeline_kwargs(*, device: str) -> dict[str, str]:
    return {"device": device}
