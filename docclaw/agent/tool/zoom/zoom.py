"""Abstract zoom tool shared by concrete zoom renderers."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    Region,
    RunState,
    ZoomRegionView,
)


PixelBBox = tuple[int, int, int, int]


class ZoomTool(Tool):
    """Base class for creating zoomed visual artifacts for known regions."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @property
    def action_type(self) -> ActionType:
        return "zoom"

    @property
    def description(self) -> str:
        return (
            "Create zoomed visual artifacts for known regions. The artifacts can "
            "later be used for region-level OCR refinement. Repeating zoom on the "
            "same region and target resolution usually has no value."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Select one or more known region_ids to zoom into.",
            "properties": {
                "region_ids": {
                    "type": "array",
                    "description": "Known region identifiers to zoom as an explicit region set.",
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
            "description": "Zoom controls.",
            "properties": {
                "target_long_side_px": {
                    "type": "integer",
                    "description": "Requested output long-side size in pixels.",
                },
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory where the zoom-view artifact should be written.",
                },
            },
            "required": ["target_long_side_px"],
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        try:
            target_long_side_px = int(action.parameters.get("target_long_side_px"))
        except (TypeError, ValueError):
            return self.error(action, "target_long_side_px must be a positive integer")
        if target_long_side_px <= 0:
            return self.error(action, "target_long_side_px must be a positive integer")

        targets, error = _resolve_zoom_targets(state, action)
        if error is not None:
            return self.error(action, error)
        assert targets is not None

        artifact_dir = _resolve_artifact_dir(state, action, self.artifact_dir)
        if artifact_dir is None:
            return self.error(action, "zoom requires artifact_dir")

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
                artifact_width, artifact_height = self.render_zoom_view(
                    source_path,
                    output_path,
                    pixel_bbox,
                    target_long_side_px,
                )
            except Exception as exc:
                return self.error(action, f"zoom failed: {exc}")
            results.append(
                {
                    "page_index": page.page_index,
                    "region_id": target["region_id"],
                    "bbox": list(target["bbox"]),
                    "coordinate_space": target["coordinate_space"],
                    "pixel_bbox": list(pixel_bbox),
                    "target_long_side_px": target_long_side_px,
                    "artifact_kind": "zoom_view",
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
                f"Created zoom view for region {result['region_id']} on page "
                f"{result['page_index']} ({result['artifact_width']}x"
                f"{result['artifact_height']}, long_side={target_long_side_px})."
            )
        else:
            message = (
                f"Created {len(results)} zoom view artifact(s) "
                f"(long_side={target_long_side_px})."
            )

        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results},
            message=message,
            artifacts=artifacts,
        )

    @abstractmethod
    def render_zoom_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
        target_long_side_px: int,
    ) -> tuple[int, int]:
        """Write a zoom-view artifact and return its width and height."""

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
            state.add_zoom_region_view(
                ZoomRegionView(
                    region_id=region_id,
                    page_index=int(result["page_index"]),
                    artifact_path=str(result["artifact_path"]),
                    target_long_side_px=int(result["target_long_side_px"]),
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
            "zoom": {
                "zoomed_region_ids": list(state.zoom_regions.ordered_region_ids),
            }
        }


def _resolve_zoom_targets(
    state: RunState,
    action: Action,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if "region_id" in action.target or "page_index" in action.target or "bbox" in action.target:
        return None, "zoom requires target.region_ids"
    region_ids = action.target.get("region_ids")
    if not isinstance(region_ids, list) or not region_ids:
        return None, "zoom target.region_ids must be a non-empty list"
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_region_id in region_ids:
        region_id = str(raw_region_id).strip()
        if not region_id:
            continue
        if region_id in seen:
            continue
        region = state.get_region(region_id)
        if region is None:
            return None, f"unknown region_id: {region_id}"
        targets.append(
            {
                "page_index": region.page_index,
                "region_id": region.region_id,
                "bbox": region.bbox,
                "coordinate_space": region.coordinate_space,
            }
        )
        seen.add(region_id)
    if not targets:
        return None, "zoom target.region_ids must contain non-empty strings"
    return targets, None


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    coordinate_space: str,
    page_width: int | None,
    page_height: int | None,
) -> tuple[PixelBBox, str | None]:
    if len(bbox) != 4:
        return (0, 0, 0, 0), "bbox must contain four coordinates"

    if coordinate_space == "relative":
        if page_width is None or page_height is None:
            return (0, 0, 0, 0), "relative bbox requires page width and height"
        values = (
            float(bbox[0]) * page_width,
            float(bbox[1]) * page_height,
            float(bbox[2]) * page_width,
            float(bbox[3]) * page_height,
        )
    elif coordinate_space == "pixel":
        values = tuple(float(value) for value in bbox)
    else:
        return (0, 0, 0, 0), f"unsupported coordinate_space: {coordinate_space}"

    x0, y0, x1, y1 = (int(round(value)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), "bbox must have positive width and height"

    if page_width is not None and page_height is not None:
        x0, y0, x1, y1 = clamp_bbox((x0, y0, x1, y1), page_width, page_height)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0), "bbox is outside the page bounds"

    return (x0, y0, x1, y1), None


def clamp_bbox(bbox: PixelBBox, width: int, height: int) -> PixelBBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _resolve_artifact_dir(
    state: RunState,
    action: Action,
    default_artifact_dir: Path | None,
) -> Path | None:
    value = action.parameters.get("artifact_dir")
    if value is None:
        value = state.metadata.get("artifact_dir")
    if value is None:
        value = state.document.metadata.get("artifact_dir")
    if value is None:
        return default_artifact_dir
    return Path(str(value)).expanduser()


def _safe_cache_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() else "_" for ch in value)
    fragment = fragment.strip("_")
    return fragment[:80] or "target"
