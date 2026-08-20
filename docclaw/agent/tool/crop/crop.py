"""Abstract crop tool shared by concrete crop renderers."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.zoom.zoom import (
    _resolve_artifact_dir,
    _resolve_zoom_targets,
    _safe_cache_fragment,
    _to_pixel_bbox,
)
from docclaw.agent.utils import (
    Action,
    ActionType,
    CropRegionView,
    Observation,
    RunState,
)


PixelBBox = tuple[int, int, int, int]


class CropTool(Tool):
    """Base class for creating crop-expanded visual artifacts for known regions."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @property
    def action_type(self) -> ActionType:
        return "crop"

    @property
    def description(self) -> str:
        return (
            "Create crop-expanded visual artifacts for known regions. The artifacts "
            "can later be used for region-level OCR refinement when the original crop "
            "looks too tight or clipped."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Select one or more known region_ids to expand with crop.",
            "properties": {
                "region_ids": {
                    "type": "array",
                    "description": "Known region identifiers to expand as an explicit region set.",
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
            "description": "Crop artifact controls.",
            "properties": {
                "left_px": {
                    "type": "integer",
                    "description": "Signed pixel offset applied to the left side.",
                },
                "right_px": {
                    "type": "integer",
                    "description": "Signed pixel offset applied to the right side.",
                },
                "top_px": {
                    "type": "integer",
                    "description": "Signed pixel offset applied to the top side.",
                },
                "bottom_px": {
                    "type": "integer",
                    "description": "Signed pixel offset applied to the bottom side.",
                },
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory where the crop-view artifact should be written.",
                },
            },
            "required": ["left_px", "right_px", "top_px", "bottom_px"],
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        offsets, error = _resolve_crop_offsets(action.parameters)
        if error is not None:
            return self.error(action, error)
        assert offsets is not None

        targets, error = _resolve_zoom_targets(state, action)
        if error is not None:
            return self.error(action, error.replace("zoom", "crop", 1))
        assert targets is not None

        artifact_dir = _resolve_artifact_dir(state, action, self.artifact_dir)
        if artifact_dir is None:
            return self.error(action, "crop requires artifact_dir")

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
            expanded_pixel_bbox, error = _offset_pixel_bbox(
                pixel_bbox,
                left_px=offsets["left_px"],
                right_px=offsets["right_px"],
                top_px=offsets["top_px"],
                bottom_px=offsets["bottom_px"],
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
                artifact_width, artifact_height = self.render_crop_view(
                    source_path,
                    output_path,
                    expanded_pixel_bbox,
                )
            except Exception as exc:
                return self.error(action, f"crop failed: {exc}")
            results.append(
                {
                    "page_index": page.page_index,
                    "region_id": target["region_id"],
                    "bbox": list(target["bbox"]),
                    "coordinate_space": target["coordinate_space"],
                    "pixel_bbox": list(expanded_pixel_bbox),
                    **offsets,
                    "artifact_kind": "crop_view",
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
                f"Created crop view for region {result['region_id']} on page "
                f"{result['page_index']} ({result['artifact_width']}x"
                f"{result['artifact_height']})."
            )
        else:
            message = f"Created {len(results)} crop view artifact(s)."

        return Observation(
            action_id=action.action_id,
            success=True,
            data={"results": results},
            message=message,
            artifacts=artifacts,
        )

    @abstractmethod
    def render_crop_view(
        self,
        source_path: Path,
        output_path: Path,
        bbox: PixelBBox,
    ) -> tuple[int, int]:
        """Write a crop-view artifact and return its width and height."""

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
            state.add_crop_region_view(
                CropRegionView(
                    region_id=region_id,
                    page_index=int(result["page_index"]),
                    artifact_path=str(result["artifact_path"]),
                    left_px=int(result["left_px"]),
                    right_px=int(result["right_px"]),
                    top_px=int(result["top_px"]),
                    bottom_px=int(result["bottom_px"]),
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
            "crop": {
                "crop_region_ids": list(state.crop_regions.ordered_region_ids),
            }
        }


def _resolve_crop_offsets(
    parameters: dict[str, Any],
) -> tuple[dict[str, int] | None, str | None]:
    values: dict[str, int] = {}
    for key in ("left_px", "right_px", "top_px", "bottom_px"):
        try:
            values[key] = int(parameters.get(key))
        except (TypeError, ValueError):
            return None, f"{key} must be an integer"
    return values, None


def _offset_pixel_bbox(
    bbox: PixelBBox,
    *,
    left_px: int,
    right_px: int,
    top_px: int,
    bottom_px: int,
) -> tuple[PixelBBox, str | None]:
    x0, y0, x1, y1 = bbox
    adjusted = (
        x0 - left_px,
        y0 - top_px,
        x1 + right_px,
        y1 + bottom_px,
    )
    if adjusted[2] <= adjusted[0] or adjusted[3] <= adjusted[1]:
        return (0, 0, 0, 0), "crop offsets collapse the bbox"
    return adjusted, None
