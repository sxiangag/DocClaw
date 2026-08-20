"""Abstract rotate tool shared by concrete rotate renderers."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.zoom.zoom import (
    PixelBBox,
    _resolve_artifact_dir,
    _resolve_zoom_targets,
    _safe_cache_fragment,
    _to_pixel_bbox,
)
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RotateRegionView,
    RunState,
)


class RotateTool(Tool):
    """Base class for creating rotated visual artifacts for known regions."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @property
    def action_type(self) -> ActionType:
        return "rotate"

    @property
    def description(self) -> str:
        return (
            "Create rotated visual artifacts for known regions. The artifacts can "
            "later be used for region-level OCR refinement when orientation may be hurting quality."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Select one or more known region_ids to rotate.",
            "properties": {
                "region_ids": {
                    "type": "array",
                    "description": "Known region identifiers to rotate as an explicit region set.",
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
            "description": "Rotate controls.",
            "properties": {
                "angle_degree": {
                    "type": "number",
                    "description": (
                        "Rotation angle in degrees. Positive values rotate clockwise; "
                        "negative values rotate counterclockwise."
                    ),
                },
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory where the rotated artifact should be written.",
                },
            },
            "required": ["angle_degree"],
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        try:
            angle_degree = float(action.parameters.get("angle_degree"))
        except (TypeError, ValueError):
            return self.error(action, "angle_degree must be a number")

        targets, error = _resolve_zoom_targets(state, action)
        if error is not None:
            return self.error(action, error.replace("zoom", "rotate", 1))
        assert targets is not None

        artifact_dir = _resolve_artifact_dir(state, action, self.artifact_dir)
        if artifact_dir is None:
            return self.error(action, "rotate requires artifact_dir")

        results: list[dict[str, Any]] = []
        artifacts: list[str] = []
        for target in targets:
            page = state.require_page(target["page_index"])
            if not page.image_path:
                return self.error(action, f"page {page.page_index} has no image_path")
            pixel_bbox, error = _to_pixel_bbox(
                target["bbox"],
                target["coordinate_space"],
                page.width,
                page.height,
            )
            if error is not None:
                return self.error(action, error)
            source_path = Path(page.image_path).expanduser()
            if not source_path.exists():
                return self.error(action, f"page image_path does not exist: {source_path}")
            output_path = artifact_dir / (
                f"{action.action_id}_{_safe_cache_fragment(target['region_id'])}.png"
            )
            try:
                artifact_width, artifact_height = self.render_rotate_view(
                    source_path,
                    output_path,
                    pixel_bbox,
                    angle_degree,
                )
            except Exception as exc:
                return self.error(action, f"rotate failed: {exc}")
            results.append(
                {
                    "page_index": page.page_index,
                    "region_id": target["region_id"],
                    "bbox": list(target["bbox"]),
                    "coordinate_space": target["coordinate_space"],
                    "pixel_bbox": list(pixel_bbox),
                    "angle_degree": angle_degree,
                    "artifact_kind": "rotate_view",
                    "source_image_path": str(source_path),
                    "artifact_path": str(output_path),
                    "artifact_width": artifact_width,
                    "artifact_height": artifact_height,
                }
            )
            artifacts.append(str(output_path))

        if len(results) == 1:
            result = results[0]
            message = (
                f"Created rotated view for region {result['region_id']} on page "
                f"{result['page_index']} ({result['artifact_width']}x"
                f"{result['artifact_height']}, angle={angle_degree:g})."
            )
        else:
            message = (
                f"Created {len(results)} rotated view artifact(s) "
                f"(angle={angle_degree:g})."
            )

        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results},
            message=message,
            artifacts=artifacts,
        )

    @abstractmethod
    def render_rotate_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
        angle_degree: float,
    ) -> tuple[int, int]:
        """Write a rotated-view artifact and return its width and height."""

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        results = observation.data.get("results")
        if not isinstance(results, list):
            return
        for result in results:
            if not isinstance(result, dict):
                continue
            region_id = result.get("region_id")
            if not isinstance(region_id, str) or not region_id.strip():
                continue
            state.add_rotate_region_view(
                RotateRegionView(
                    region_id=region_id,
                    page_index=int(result["page_index"]),
                    artifact_path=str(result["artifact_path"]),
                    angle_degree=float(result["angle_degree"]),
                    bbox=tuple(float(value) for value in result["bbox"]),  # type: ignore[arg-type]
                    pixel_bbox=tuple(int(value) for value in result["pixel_bbox"]),  # type: ignore[arg-type]
                    coordinate_space=result.get("coordinate_space", "pixel"),
                    artifact_width=(
                        int(result["artifact_width"])
                        if result.get("artifact_width") is not None
                        else None
                    ),
                    artifact_height=(
                        int(result["artifact_height"])
                        if result.get("artifact_height") is not None
                        else None
                    ),
                    source_image_path=(
                        str(result["source_image_path"])
                        if result.get("source_image_path") is not None
                        else None
                    ),
                )
            )

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        return {
            "rotate": {
                "rotated_region_ids": list(state.rotate_regions.ordered_region_ids),
            }
        }
