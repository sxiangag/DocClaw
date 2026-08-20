"""Base evidence extraction tool."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Evidence,
    EvidenceAssessment,
    Observation,
    RunState,
    is_page_level_synthetic_region_id,
)


class EvidenceTool(Tool):
    """Base class for tools that add evidence to run state."""

    @property
    def action_type(self) -> ActionType:
        return "extract_evidence"

    @property
    def description(self) -> str:
        return (
            "Assess and extract question-relevant evidence from selected pages "
            "or regions. Use page_ids to review page-level evidence candidates, "
            "use region_ids for specific known regions, and use both only when you "
            "want to focus on specific regions within a specific page subset. Do not "
            "use placeholder values such as '__all__', 'string', or empty strings."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": (
                "Optional focused page set or region set for evidence assessment. "
                "Use page_ids for page-level focus, region_ids for specific known "
                "regions, or both when you need the intersection."
            ),
            "properties": {
                "page_ids": {
                    "type": "array",
                    "description": "Focused page set for evidence assessment. Leave region_ids empty or omit it when only page-level focus is needed.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "region_ids": {
                    "type": "array",
                    "description": "Focused region set for evidence assessment. Leave this empty or omit it when only page-level focus is needed.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": (
                "Optional execution parameters. Omit mode, or use mode='default', for "
                "ordinary evidence extraction. Use mode='not_answerable_recheck' only "
                "as a final page-level reassessment immediately before concluding "
                "'Not answerable'."
            ),
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["default", "not_answerable_recheck"],
                    "description": (
                        "Use 'default' for ordinary evidence extraction. Use "
                        "'not_answerable_recheck' only as a final reassessment "
                        "immediately before concluding 'Not answerable'. Do not "
                        "use it during ordinary evidence extraction."
                    ),
                },
                "with_page_images": {
                    "type": "boolean",
                    "description": (
                        "Whether to include page images together with page-level "
                        "text context when page_ids are provided during ordinary "
                        "evidence extraction. Defaults to true. This parameter is "
                        "ignored for not_answerable_recheck."
                    ),
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        try:
            evidence = self.extract_evidence(state, action)
        except Exception as exc:
            return self.error(action, str(exc))

        if not evidence:
            return Observation(
                action_id=action.action_id,
                success=True,
                data={"evidence": []},
                message="No evidence extracted from current document state.",
            )

        pages = sorted({
            item.page_index
            for item in evidence
            if item.page_index is not None
        })
        regions = sorted({
            item.region_id
            for item in evidence
            if item.region_id
        })
        scope = []
        if pages:
            scope.append("pages=" + ",".join(str(page) for page in pages))
        if regions:
            scope.append("regions=" + ",".join(regions[:3]))
        return Observation(
            action_id=action.action_id,
            success=True,
            data={"evidence": [item.to_dict() for item in evidence]},
            message=(
                f"Prepared {len(evidence)} evidence item(s)"
                + (f" ({'; '.join(scope)})" if scope else "")
                + "."
            ),
        )

    @abstractmethod
    def extract_evidence(self, state: RunState, action: Action) -> list[Evidence]:
        """Return evidence items for the current action."""

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        evidence_data = observation.data.get("evidence")
        if isinstance(evidence_data, dict):
            state.add_evidence(Evidence.from_dict(evidence_data))
        if isinstance(evidence_data, list):
            for item in evidence_data:
                if isinstance(item, dict):
                    state.add_evidence(Evidence.from_dict(item))
        if any(
            key in observation.data
            for key in ("assessment_id", "answerability_status", "missing_information")
        ):
            state.add_evidence_assessment(
                EvidenceAssessment(
                    assessment_id=str(observation.data.get("assessment_id") or action.action_id),
                    action_id=action.action_id,
                    page_indices=_normalize_int_list(action.target.get("page_indices")),
                    region_ids=_known_region_ids(
                        state,
                        action.target.get("region_ids"),
                    ),
                    answerability_status=str(observation.data.get("answerability_status") or "inconclusive"),
                    missing_information=(
                        str(observation.data["missing_information"])
                        if observation.data.get("missing_information") is not None
                        else None
                    ),
                    evidence_ids=_evidence_ids_from_data(observation.data.get("evidence")),
                )
            )


def build_evidence(
    state: RunState,
    action: Action,
    *,
    content: str,
    trust_level: str | None = None,
    reference: str | None = None,
    page_index: int | None = None,
    region_id: str | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    """Build and validate an Evidence object against run state."""
    metadata_dict = dict(metadata or {})
    resolved_trust_level = (trust_level or "").strip().lower() or "trusted"
    if trust_level is None and isinstance(metadata_dict.get("source_kind"), str):
        resolved_trust_level = "untrusted"
    if reference is None:
        for key in ("resource_id", "search_id"):
            value = metadata_dict.get(key)
            if isinstance(value, str) and value.strip():
                reference = value.strip()
                break

    if region_id and state.get_region(region_id) is None and not is_page_level_synthetic_region_id(region_id):
        metadata_dict.setdefault("unresolved_region_id", region_id)
        region_id = None
    if page_index is not None and state.get_page(page_index) is None:
        raise ValueError(f"unknown page_index: {page_index}")
    return Evidence(
        content=content,
        trust_level=resolved_trust_level,  # type: ignore[arg-type]
        reference=reference,
        page_index=page_index,
        region_id=region_id,
        action_id=action.action_id,
        confidence=confidence,
        metadata=metadata_dict,
    )


def _normalize_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, int)]


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _known_region_ids(state: RunState, value: Any) -> list[str]:
    return [
        region_id
        for region_id in _normalize_str_list(value)
        if state.get_region(region_id) is not None
    ]


def _evidence_ids_from_data(value: Any) -> list[str]:
    if isinstance(value, dict):
        evidence_id = value.get("evidence_id")
        return [str(evidence_id)] if isinstance(evidence_id, str) and evidence_id.strip() else []
    if not isinstance(value, list):
        return []
    evidence_ids: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id.strip():
            evidence_ids.append(evidence_id)
    return evidence_ids
