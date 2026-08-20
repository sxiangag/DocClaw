"""Typed runtime configuration for DocClaw."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    return int(str(value))


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _coerce_optional_str(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _coerce_str(value, field_name=field_name)


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _coerce_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a table")
    return dict(value)


@dataclass(slots=True)
class RuntimeConfig:
    max_steps: int = 100
    session_max_messages: int = 10

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("runtime.max_steps must be positive")
        if self.session_max_messages < 0:
            raise ValueError("runtime.session_max_messages must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuntimeConfig:
        values = _coerce_dict(data, field_name="runtime")
        return cls(
            max_steps=_coerce_int(values.get("max_steps", 100), field_name="runtime.max_steps"),
            session_max_messages=_coerce_int(
                values.get("session_max_messages", 10),
                field_name="runtime.session_max_messages",
            ),
        )


@dataclass(slots=True)
class ProviderConfig:
    name: str = "openai_codex"
    model: str | None = None
    url: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    originator: str | None = None
    verify_ssl: bool = True
    allow_insecure_ssl_fallback: bool = True
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        self.name = self.name.replace("-", "_").strip().lower()
        if self.name not in {"openai_codex", "openai", "anthropic", "gemini"}:
            raise ValueError(f"unsupported provider.name: {self.name}")
        if self.model is not None and not self.model.strip():
            raise ValueError("provider.model must not be empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("provider.url must not be empty")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("provider.api_key must not be empty")
        if self.api_version is not None and not self.api_version.strip():
            raise ValueError("provider.api_version must not be empty")
        if self.originator is not None and not self.originator.strip():
            raise ValueError("provider.originator must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("provider.request_timeout must be positive")
        if self.name in {"openai", "anthropic", "gemini"} and self.api_key is None:
            raise ValueError(f"provider.api_key is required for {self.name}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProviderConfig:
        values = _coerce_dict(data, field_name="provider")
        return cls(
            name=_coerce_str(values.get("name", "openai_codex"), field_name="provider.name"),
            model=_coerce_optional_str(values.get("model"), field_name="provider.model"),
            url=_coerce_optional_str(values.get("url"), field_name="provider.url"),
            api_key=_coerce_optional_str(values.get("api_key"), field_name="provider.api_key"),
            api_version=_coerce_optional_str(values.get("api_version"), field_name="provider.api_version"),
            originator=_coerce_optional_str(values.get("originator"), field_name="provider.originator"),
            verify_ssl=_coerce_bool(values.get("verify_ssl", True), field_name="provider.verify_ssl"),
            allow_insecure_ssl_fallback=_coerce_bool(
                values.get("allow_insecure_ssl_fallback", True),
                field_name="provider.allow_insecure_ssl_fallback",
            ),
            request_timeout=_coerce_float(
                values.get("request_timeout", 60.0),
                field_name="provider.request_timeout",
            ),
        )


@dataclass(slots=True)
class ProvidersConfig:
    entries: dict[str, ProviderConfig] = field(
        default_factory=lambda: {"planner": ProviderConfig()}
    )

    def __post_init__(self) -> None:
        normalized: dict[str, ProviderConfig] = {}
        for raw_name, config in self.entries.items():
            name = _coerce_str(raw_name, field_name="providers.<name>").strip()
            if not name:
                raise ValueError("providers keys must not be empty")
            normalized[name] = config
        if not normalized:
            raise ValueError("providers must define at least one provider")
        self.entries = normalized

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProvidersConfig:
        values = _coerce_dict(data, field_name="providers")
        if not values:
            return cls()
        entries: dict[str, ProviderConfig] = {}
        for name, raw in values.items():
            if not isinstance(name, str):
                raise ValueError("providers keys must be strings")
            entries[name] = ProviderConfig.from_dict(raw)
        return cls(entries=entries)

    def require(self, name: str) -> ProviderConfig:
        try:
            return self.entries[name]
        except KeyError as exc:
            raise ValueError(f"unknown provider reference: {name}") from exc


@dataclass(slots=True)
class PlannerConfig:
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PlannerConfig:
        values = _coerce_dict(data, field_name="planner")
        return cls(
            provider=_coerce_str(values.get("provider", "planner"), field_name="planner.provider"),
            model=_coerce_optional_str(values.get("model"), field_name="planner.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="planner.temperature",
            ),
        )


@dataclass(slots=True)
class LayoutToolConfig:
    enabled: bool = True
    device: str = "gpu:0"

    def __post_init__(self) -> None:
        if not self.device.startswith("gpu:"):
            raise ValueError("tools.layout.device must be a GPU device like gpu:0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LayoutToolConfig:
        values = _coerce_dict(data, field_name="tools.layout")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.layout.enabled"),
            device=_coerce_str(values.get("device", "gpu:0"), field_name="tools.layout.device"),
        )


@dataclass(slots=True)
class ZoomToolConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ZoomToolConfig:
        values = _coerce_dict(data, field_name="tools.zoom")
        return cls(enabled=_coerce_bool(values.get("enabled", True), field_name="tools.zoom.enabled"))


@dataclass(slots=True)
class CropToolConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CropToolConfig:
        values = _coerce_dict(data, field_name="tools.crop")
        return cls(enabled=_coerce_bool(values.get("enabled", True), field_name="tools.crop.enabled"))


@dataclass(slots=True)
class OcrEnhancementToolConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OcrEnhancementToolConfig:
        values = _coerce_dict(data, field_name="tools.ocr_enhancement")
        return cls(
            enabled=_coerce_bool(
                values.get("enabled", True),
                field_name="tools.ocr_enhancement.enabled",
            )
        )


@dataclass(slots=True)
class OcrToolConfig:
    enabled: bool = True
    backend: str = "paddleocr_vl"
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    device: str = "gpu:0"

    def __post_init__(self) -> None:
        if self.backend not in {"paddleocr", "paddleocr_vl", "vlm"}:
            raise ValueError(f"unsupported tools.ocr.backend: {self.backend}")
        if self.backend != "vlm" and not self.device.startswith("gpu:"):
            raise ValueError("tools.ocr.device must be a GPU device like gpu:0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OcrToolConfig:
        values = _coerce_dict(data, field_name="tools.ocr")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.ocr.enabled"),
            backend=_coerce_str(
                values.get("backend", "paddleocr_vl"),
                field_name="tools.ocr.backend",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.ocr.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.ocr.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.ocr.temperature",
            ),
            max_tokens=_coerce_optional_int(
                values.get("max_tokens"),
                field_name="tools.ocr.max_tokens",
            ),
            device=_coerce_str(values.get("device", "gpu:0"), field_name="tools.ocr.device"),
        )


@dataclass(slots=True)
class TableToolConfig:
    enabled: bool = True
    backend: str = "paddleocr_vl"
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    device: str = "gpu:0"

    def __post_init__(self) -> None:
        if self.backend not in {"paddleocr_vl", "vlm"}:
            raise ValueError(f"unsupported tools.table.backend: {self.backend}")
        if self.backend != "vlm" and not self.device.startswith("gpu:"):
            raise ValueError("tools.table.device must be a GPU device like gpu:0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TableToolConfig:
        values = _coerce_dict(data, field_name="tools.table")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.table.enabled"),
            backend=_coerce_str(
                values.get("backend", "paddleocr_vl"),
                field_name="tools.table.backend",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.table.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.table.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.table.temperature",
            ),
            max_tokens=_coerce_optional_int(
                values.get("max_tokens"),
                field_name="tools.table.max_tokens",
            ),
            device=_coerce_str(values.get("device", "gpu:0"), field_name="tools.table.device"),
        )


@dataclass(slots=True)
class FormulaToolConfig:
    enabled: bool = False
    backend: str = "paddleocr_vl"
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    device: str = "gpu:0"

    def __post_init__(self) -> None:
        if self.backend not in {"paddleocr_vl", "vlm"}:
            raise ValueError(f"unsupported tools.formula.backend: {self.backend}")
        if self.backend != "vlm" and not self.device.startswith("gpu:"):
            raise ValueError("tools.formula.device must be a GPU device like gpu:0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FormulaToolConfig:
        values = _coerce_dict(data, field_name="tools.formula")
        return cls(
            enabled=_coerce_bool(
                values.get("enabled", False),
                field_name="tools.formula.enabled",
            ),
            backend=_coerce_str(
                values.get("backend", "paddleocr_vl"),
                field_name="tools.formula.backend",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.formula.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.formula.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.formula.temperature",
            ),
            max_tokens=_coerce_optional_int(
                values.get("max_tokens"),
                field_name="tools.formula.max_tokens",
            ),
            device=_coerce_str(values.get("device", "gpu:0"), field_name="tools.formula.device"),
        )


@dataclass(slots=True)
class ChartToolConfig:
    enabled: bool = False
    backend: str = "paddleocr_vl"
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    device: str = "gpu:0"

    def __post_init__(self) -> None:
        if self.backend not in {"paddleocr_vl", "vlm"}:
            raise ValueError(f"unsupported tools.chart.backend: {self.backend}")
        if self.backend != "vlm" and not self.device.startswith("gpu:"):
            raise ValueError("tools.chart.device must be a GPU device like gpu:0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ChartToolConfig:
        values = _coerce_dict(data, field_name="tools.chart")
        return cls(
            enabled=_coerce_bool(values.get("enabled", False), field_name="tools.chart.enabled"),
            backend=_coerce_str(
                values.get("backend", "paddleocr_vl"),
                field_name="tools.chart.backend",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.chart.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.chart.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.chart.temperature",
            ),
            max_tokens=_coerce_optional_int(
                values.get("max_tokens"),
                field_name="tools.chart.max_tokens",
            ),
            device=_coerce_str(values.get("device", "gpu:0"), field_name="tools.chart.device"),
        )


@dataclass(slots=True)
class InternalSearchToolConfig:
    enabled: bool = True
    provider: str = "keyword"

    def __post_init__(self) -> None:
        self.provider = self.provider.strip().lower()
        if self.provider == "lexical":
            self.provider = "keyword"
        elif self.provider == "semantic":
            self.provider = "text"
        if self.provider not in {"keyword", "text", "image"}:
            raise ValueError(f"unsupported tools.internal_search.provider: {self.provider}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InternalSearchToolConfig:
        values = _coerce_dict(data, field_name="tools.internal_search")
        return cls(
            enabled=_coerce_bool(
                values.get("enabled", True),
                field_name="tools.internal_search.enabled",
            ),
            provider=_coerce_str(
                values.get("provider", "keyword"),
                field_name="tools.internal_search.provider",
            ),
        )


@dataclass(slots=True)
class FigureToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("tools.figure.model must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigureToolConfig:
        values = _coerce_dict(data, field_name="tools.figure")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.figure.enabled"),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.figure.provider",
            ),
            model=_coerce_optional_str(
                values.get("model"),
                field_name="tools.figure.model",
            ),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.figure.temperature",
            ),
        )


@dataclass(slots=True)
class SelectPagesToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("tools.select_pages.model must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SelectPagesToolConfig:
        values = _coerce_dict(data, field_name="tools.select_pages")
        return cls(
            enabled=_coerce_bool(
                values.get("enabled", True),
                field_name="tools.select_pages.enabled",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.select_pages.provider",
            ),
            model=_coerce_optional_str(
                values.get("model"),
                field_name="tools.select_pages.model",
            ),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.select_pages.temperature",
            ),
        )


@dataclass(slots=True)
class EvidenceToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvidenceToolConfig:
        values = _coerce_dict(data, field_name="tools.evidence")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.evidence.enabled"),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.evidence.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.evidence.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.evidence.temperature",
            ),
        )


@dataclass(slots=True)
class InspectOcrToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0
    max_refine_regions: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InspectOcrToolConfig:
        values = _coerce_dict(data, field_name="tools.inspect_ocr")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.inspect_ocr.enabled"),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.inspect_ocr.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.inspect_ocr.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.inspect_ocr.temperature",
            ),
            max_refine_regions=_coerce_int(
                values.get("max_refine_regions", 5),
                field_name="tools.inspect_ocr.max_refine_regions",
            ),
        )


@dataclass(slots=True)
class SelectOcrToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SelectOcrToolConfig:
        values = _coerce_dict(data, field_name="tools.select_ocr")
        return cls(
            enabled=_coerce_bool(
                values.get("enabled", True),
                field_name="tools.select_ocr.enabled",
            ),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.select_ocr.provider",
            ),
            model=_coerce_optional_str(
                values.get("model"),
                field_name="tools.select_ocr.model",
            ),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.select_ocr.temperature",
            ),
        )


@dataclass(slots=True)
class AnswerFromEvidenceToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnswerFromEvidenceToolConfig:
        values = _coerce_dict(data, field_name="tools.answer_from_evidence")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.answer_from_evidence.enabled"),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.answer_from_evidence.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.answer_from_evidence.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.answer_from_evidence.temperature",
            ),
        )


@dataclass(slots=True)
class AnswerJsonToolConfig:
    enabled: bool = True
    provider: str = "planner"
    model: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnswerJsonToolConfig:
        values = _coerce_dict(data, field_name="tools.answer_json")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="tools.answer_json.enabled"),
            provider=_coerce_str(
                values.get("provider", "planner"),
                field_name="tools.answer_json.provider",
            ),
            model=_coerce_optional_str(values.get("model"), field_name="tools.answer_json.model"),
            temperature=_coerce_float(
                values.get("temperature", 0.0),
                field_name="tools.answer_json.temperature",
            ),
        )


@dataclass(slots=True)
class ToolsConfig:
    layout: LayoutToolConfig = field(default_factory=LayoutToolConfig)
    formula: FormulaToolConfig = field(default_factory=FormulaToolConfig)
    chart: ChartToolConfig = field(default_factory=ChartToolConfig)
    table: TableToolConfig = field(default_factory=TableToolConfig)
    internal_search: InternalSearchToolConfig = field(default_factory=InternalSearchToolConfig)
    zoom: ZoomToolConfig = field(default_factory=ZoomToolConfig)
    crop: CropToolConfig = field(default_factory=CropToolConfig)
    ocr_enhancement: OcrEnhancementToolConfig = field(default_factory=OcrEnhancementToolConfig)
    ocr: OcrToolConfig = field(default_factory=OcrToolConfig)
    figure: FigureToolConfig = field(default_factory=FigureToolConfig)
    select_pages: SelectPagesToolConfig = field(default_factory=SelectPagesToolConfig)
    evidence: EvidenceToolConfig = field(default_factory=EvidenceToolConfig)
    inspect_ocr: InspectOcrToolConfig = field(default_factory=InspectOcrToolConfig)
    select_ocr: SelectOcrToolConfig = field(default_factory=SelectOcrToolConfig)
    answer_from_evidence: AnswerFromEvidenceToolConfig = field(default_factory=AnswerFromEvidenceToolConfig)
    answer_json: AnswerJsonToolConfig = field(default_factory=AnswerJsonToolConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ToolsConfig:
        values = _coerce_dict(data, field_name="tools")
        return cls(
            layout=LayoutToolConfig.from_dict(values.get("layout")),
            formula=FormulaToolConfig.from_dict(values.get("formula")),
            chart=ChartToolConfig.from_dict(values.get("chart")),
            table=TableToolConfig.from_dict(values.get("table")),
            internal_search=InternalSearchToolConfig.from_dict(values.get("internal_search")),
            zoom=ZoomToolConfig.from_dict(values.get("zoom")),
            crop=CropToolConfig.from_dict(values.get("crop")),
            ocr_enhancement=OcrEnhancementToolConfig.from_dict(values.get("ocr_enhancement")),
            ocr=OcrToolConfig.from_dict(values.get("ocr")),
            figure=FigureToolConfig.from_dict(values.get("figure")),
            select_pages=SelectPagesToolConfig.from_dict(values.get("select_pages")),
            evidence=EvidenceToolConfig.from_dict(values.get("evidence")),
            inspect_ocr=InspectOcrToolConfig.from_dict(values.get("inspect_ocr")),
            select_ocr=SelectOcrToolConfig.from_dict(values.get("select_ocr")),
            answer_from_evidence=AnswerFromEvidenceToolConfig.from_dict(values.get("answer_from_evidence")),
            answer_json=AnswerJsonToolConfig.from_dict(values.get("answer_json")),
        )


@dataclass(slots=True)
class StorageConfig:
    root_dir: str = ".docclaw"
    sessions_dir: str | None = None
    artifacts_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.root_dir.strip():
            raise ValueError("storage.root_dir must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StorageConfig:
        values = _coerce_dict(data, field_name="storage")
        return cls(
            root_dir=_coerce_str(values.get("root_dir", ".docclaw"), field_name="storage.root_dir"),
            sessions_dir=_coerce_optional_str(values.get("sessions_dir"), field_name="storage.sessions_dir"),
            artifacts_dir=_coerce_optional_str(values.get("artifacts_dir"), field_name="storage.artifacts_dir"),
        )


@dataclass(slots=True)
class SkillsConfig:
    enabled: bool = True
    workspace_dir: str | None = "skills"

    def __post_init__(self) -> None:
        if self.workspace_dir is not None and not self.workspace_dir.strip():
            raise ValueError("skills.workspace_dir must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SkillsConfig:
        values = _coerce_dict(data, field_name="skills")
        return cls(
            enabled=_coerce_bool(values.get("enabled", True), field_name="skills.enabled"),
            workspace_dir=_coerce_optional_str(
                values.get("workspace_dir", "skills"),
                field_name="skills.workspace_dir",
            ),
        )


@dataclass(slots=True)
class DocClawConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)

    def __post_init__(self) -> None:
        self.providers.require(self.planner.provider)
        if self.tools.ocr.enabled and self.tools.ocr.backend == "vlm":
            self.providers.require(self.tools.ocr.provider)
        if self.tools.formula.enabled and self.tools.formula.backend == "vlm":
            self.providers.require(self.tools.formula.provider)
        if self.tools.table.enabled and self.tools.table.backend == "vlm":
            self.providers.require(self.tools.table.provider)
        if self.tools.chart.enabled and self.tools.chart.backend == "vlm":
            self.providers.require(self.tools.chart.provider)
        if self.tools.figure.enabled:
            self.providers.require(self.tools.figure.provider)
        if self.tools.select_pages.enabled:
            self.providers.require(self.tools.select_pages.provider)
        if self.tools.evidence.enabled:
            self.providers.require(self.tools.evidence.provider)
        if self.tools.inspect_ocr.enabled:
            self.providers.require(self.tools.inspect_ocr.provider)
        if self.tools.select_ocr.enabled:
            self.providers.require(self.tools.select_ocr.provider)
        if self.tools.answer_from_evidence.enabled:
            self.providers.require(self.tools.answer_from_evidence.provider)
        if self.tools.answer_json.enabled:
            self.providers.require(self.tools.answer_json.provider)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DocClawConfig:
        values = _coerce_dict(data, field_name="config")
        return cls(
            runtime=RuntimeConfig.from_dict(values.get("runtime")),
            providers=ProvidersConfig.from_dict(values.get("providers")),
            planner=PlannerConfig.from_dict(values.get("planner")),
            tools=ToolsConfig.from_dict(values.get("tools")),
            storage=StorageConfig.from_dict(values.get("storage")),
            skills=SkillsConfig.from_dict(values.get("skills")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": {
                "max_steps": self.runtime.max_steps,
                "session_max_messages": self.runtime.session_max_messages,
            },
            "providers": {
                name: {
                    "name": provider.name,
                    "model": provider.model,
                    "url": provider.url,
                    "api_key": provider.api_key,
                    "api_version": provider.api_version,
                    "originator": provider.originator,
                    "verify_ssl": provider.verify_ssl,
                    "allow_insecure_ssl_fallback": provider.allow_insecure_ssl_fallback,
                    "request_timeout": provider.request_timeout,
                }
                for name, provider in self.providers.entries.items()
            },
            "planner": {
                "provider": self.planner.provider,
                "model": self.planner.model,
                "temperature": self.planner.temperature,
            },
            "tools": {
                "layout": {
                    "enabled": self.tools.layout.enabled,
                    "device": self.tools.layout.device,
                },
                "formula": {
                    "enabled": self.tools.formula.enabled,
                    "backend": self.tools.formula.backend,
                    "provider": self.tools.formula.provider,
                    "model": self.tools.formula.model,
                    "temperature": self.tools.formula.temperature,
                    "max_tokens": self.tools.formula.max_tokens,
                    "device": self.tools.formula.device,
                },
                "chart": {
                    "enabled": self.tools.chart.enabled,
                    "backend": self.tools.chart.backend,
                    "provider": self.tools.chart.provider,
                    "model": self.tools.chart.model,
                    "temperature": self.tools.chart.temperature,
                    "max_tokens": self.tools.chart.max_tokens,
                    "device": self.tools.chart.device,
                },
                "table": {
                    "enabled": self.tools.table.enabled,
                    "backend": self.tools.table.backend,
                    "provider": self.tools.table.provider,
                    "model": self.tools.table.model,
                    "temperature": self.tools.table.temperature,
                    "max_tokens": self.tools.table.max_tokens,
                    "device": self.tools.table.device,
                },
                "internal_search": {
                    "enabled": self.tools.internal_search.enabled,
                    "provider": self.tools.internal_search.provider,
                },
                "zoom": {
                    "enabled": self.tools.zoom.enabled,
                },
                "crop": {
                    "enabled": self.tools.crop.enabled,
                },
                "ocr_enhancement": {
                    "enabled": self.tools.ocr_enhancement.enabled,
                },
                "ocr": {
                    "enabled": self.tools.ocr.enabled,
                    "backend": self.tools.ocr.backend,
                    "provider": self.tools.ocr.provider,
                    "model": self.tools.ocr.model,
                    "temperature": self.tools.ocr.temperature,
                    "max_tokens": self.tools.ocr.max_tokens,
                    "device": self.tools.ocr.device,
                },
                "figure": {
                    "enabled": self.tools.figure.enabled,
                    "provider": self.tools.figure.provider,
                    "model": self.tools.figure.model,
                    "temperature": self.tools.figure.temperature,
                },
                "select_pages": {
                    "enabled": self.tools.select_pages.enabled,
                    "provider": self.tools.select_pages.provider,
                    "model": self.tools.select_pages.model,
                    "temperature": self.tools.select_pages.temperature,
                },
                "evidence": {
                    "enabled": self.tools.evidence.enabled,
                    "provider": self.tools.evidence.provider,
                    "model": self.tools.evidence.model,
                    "temperature": self.tools.evidence.temperature,
                },
                "inspect_ocr": {
                    "enabled": self.tools.inspect_ocr.enabled,
                    "provider": self.tools.inspect_ocr.provider,
                    "model": self.tools.inspect_ocr.model,
                    "temperature": self.tools.inspect_ocr.temperature,
                    "max_refine_regions": self.tools.inspect_ocr.max_refine_regions,
                },
                "select_ocr": {
                    "enabled": self.tools.select_ocr.enabled,
                    "provider": self.tools.select_ocr.provider,
                    "model": self.tools.select_ocr.model,
                    "temperature": self.tools.select_ocr.temperature,
                },
                "answer_from_evidence": {
                    "enabled": self.tools.answer_from_evidence.enabled,
                    "provider": self.tools.answer_from_evidence.provider,
                    "model": self.tools.answer_from_evidence.model,
                    "temperature": self.tools.answer_from_evidence.temperature,
                },
                "answer_json": {
                    "enabled": self.tools.answer_json.enabled,
                    "provider": self.tools.answer_json.provider,
                    "model": self.tools.answer_json.model,
                    "temperature": self.tools.answer_json.temperature,
                },
            },
            "storage": {
                "root_dir": self.storage.root_dir,
                "sessions_dir": self.storage.sessions_dir,
                "artifacts_dir": self.storage.artifacts_dir,
            },
            "skills": {
                "enabled": self.skills.enabled,
                "workspace_dir": self.skills.workspace_dir,
            },
        }
