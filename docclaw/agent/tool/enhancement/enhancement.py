"""Enhancement tool that orchestrates crop refinement and OCR candidate generation."""

from __future__ import annotations

import inspect
from typing import Any

from docclaw.agent.tool.crop import CropTool
from docclaw.agent.tool.ocr import OcrTool
from docclaw.agent.tool.rotate import RotateTool
from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.zoom import ZoomTool
from docclaw.agent.utils import Action, ActionType, Observation, RunState


class EnhancementTool(Tool):
    """Generate OCR refinement candidates from structured refinement actions."""

    def __init__(
        self,
        ocr_tool: OcrTool,
        *,
        zoom_tool: ZoomTool | None = None,
        crop_tool: CropTool | None = None,
        rotate_tool: RotateTool | None = None,
    ) -> None:
        self.ocr_tool = ocr_tool
        self.zoom_tool = zoom_tool
        self.crop_tool = crop_tool
        self.rotate_tool = rotate_tool

    @property
    def action_type(self) -> ActionType:
        return "ocr_enhancement"

    @property
    def description(self) -> str:
        return (
            "Generate OCR refinement candidates for known regions by executing "
            "structured zoom, crop, or rotate refinement actions and then rerunning OCR."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Provide one or more structured refinement actions for OCR refinement.",
            "properties": {
                "refinement_actions": {
                    "type": "array",
                    "description": "Structured refinement actions emitted by inspect_ocr.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "region_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "action": {"type": "string", "enum": ["zoom", "crop", "rotate"]},
                            "target_long_side_px": {"type": "integer"},
                            "left_px": {"type": "integer"},
                            "right_px": {"type": "integer"},
                            "top_px": {"type": "integer"},
                            "bottom_px": {"type": "integer"},
                            "angle_degree": {"type": "number"},
                        },
                        "required": ["region_id", "action"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
            },
            "required": ["refinement_actions"],
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Enhancement controls.",
            "properties": {
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory for intermediate enhancement artifacts.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        refinement_actions, error = _resolve_refinement_actions(
            action.target.get("refinement_actions"),
            available_modes=self._available_modes(),
        )
        if error is not None:
            return self.error(action, error)
        assert refinement_actions is not None

        ocr_action = Action(
            action_type="ocr",
            target={},
            parameters=_ocr_parameters(action.parameters),
        )

        view_results: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        artifacts: list[str] = []

        for refinement_action in refinement_actions:
            mode = refinement_action["action"]
            render_tool = self._tool_for_mode(mode)
            if render_tool is None:
                return self.error(action, f"ocr_enhancement mode is unavailable: {mode}")

            render_action = Action(
                action_type=mode,  # type: ignore[arg-type]
                target={"region_ids": [refinement_action["region_id"]]},
                parameters=_render_parameters(
                    action.parameters,
                    refinement_action=refinement_action,
                ),
            )
            render_observation = await render_tool.execute(state, render_action)
            if not render_observation.success:
                return self.error(
                    action,
                    render_observation.error or f"ocr_enhancement {mode} failed",
                )

            raw_view_results = render_observation.data.get("results")
            if not isinstance(raw_view_results, list) or not raw_view_results:
                return self.error(action, "ocr_enhancement produced no render results")

            view_result = dict(raw_view_results[0])
            view_result["mode"] = mode
            view_result["reason"] = refinement_action.get("reason")
            view_results.append(view_result)
            artifacts.extend(render_observation.artifacts)

            target = _build_enhanced_ocr_target(mode=mode, view_result=view_result)
            result, target_artifacts, error = await self._execute_ocr_target(
                state,
                ocr_action,
                target,
            )
            if error is not None:
                return self.error(action, error)
            if result is None:
                continue
            result["candidate_kind"] = mode
            results.append(result)
            artifacts.extend(target_artifacts)
            source = str(result.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        return Observation(
            action_id=action.action_id,
            success=True,
            data={
                "view_results": view_results,
                "results": results,
                "sources": source_counts,
            },
            message=f"Generated {len(results)} OCR refinement candidate(s).",
            artifacts=artifacts,
        )

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        view_results = observation.data.get("view_results")
        if isinstance(view_results, list):
            for mode in self._available_modes():
                render_tool = self._tool_for_mode(mode)
                if render_tool is None:
                    continue
                mode_results = [
                    item
                    for item in view_results
                    if isinstance(item, dict) and item.get("mode") == mode
                ]
                if not mode_results:
                    continue
                render_tool.update_state(
                    state,
                    Action(action_type=mode, target={}, parameters={}),  # type: ignore[arg-type]
                    Observation(
                        action_id=observation.action_id,
                        success=True,
                        data={"results": mode_results},
                        artifacts=observation.artifacts,
                    ),
                )

        results = observation.data.get("results")
        if isinstance(results, list):
            self.ocr_tool.update_state(
                state,
                Action(action_type="ocr", target={}, parameters={}),
                Observation(
                    action_id=observation.action_id,
                    success=True,
                    data={"results": results},
                    artifacts=observation.artifacts,
                ),
            )

    def _available_modes(self) -> list[str]:
        modes: list[str] = []
        if self.zoom_tool is not None:
            modes.append("zoom")
        if self.crop_tool is not None:
            modes.append("crop")
        if self.rotate_tool is not None:
            modes.append("rotate")
        return modes

    def _tool_for_mode(self, mode: str) -> Tool | None:
        if mode == "zoom":
            return self.zoom_tool
        if mode == "crop":
            return self.crop_tool
        if mode == "rotate":
            return self.rotate_tool
        return None

    async def _execute_ocr_target(
        self,
        state: RunState,
        action: Action,
        target: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str], str | None]:
        execute_single_target = getattr(self.ocr_tool, "_execute_single_target", None)
        if not callable(execute_single_target):
            return None, [], "configured ocr tool does not support ocr_enhancement execution"
        result = execute_single_target(
            state,
            action,
            target,
            force=bool(action.parameters.get("force", False)),
        )
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, tuple) or len(result) != 3:
            return None, [], "configured ocr tool returned an invalid ocr_enhancement result"
        result, artifacts, error = result
        return result, artifacts, error


def _resolve_refinement_actions(
    value: Any,
    *,
    available_modes: list[str],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not available_modes:
        return None, "ocr_enhancement requires at least one available mode"
    if not isinstance(value, list) or not value:
        return None, "ocr_enhancement target.refinement_actions must be a non-empty list"

    allowed = set(available_modes)
    results: list[dict[str, Any]] = []
    seen_region_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            return None, "ocr_enhancement target.refinement_actions must contain objects"
        region_id = str(item.get("region_id", "")).strip()
        mode = str(item.get("action", "")).strip()
        if not region_id:
            return None, "ocr_enhancement refinement action requires region_id"
        if mode not in allowed:
            return None, f"ocr_enhancement action must be one of: {', '.join(available_modes)}"
        if region_id in seen_region_ids:
            continue

        normalized = {"region_id": region_id, "action": mode}
        if isinstance(item.get("reason"), str):
            normalized["reason"] = item["reason"]

        if mode == "zoom":
            try:
                normalized["target_long_side_px"] = int(item.get("target_long_side_px"))
            except (TypeError, ValueError):
                return None, "ocr_enhancement zoom action requires target_long_side_px"
        elif mode == "crop":
            for key in ("left_px", "right_px", "top_px", "bottom_px"):
                try:
                    normalized[key] = int(item.get(key))
                except (TypeError, ValueError):
                    return None, f"ocr_enhancement crop action requires integer {key}"
        elif mode == "rotate":
            try:
                normalized["angle_degree"] = float(item.get("angle_degree"))
            except (TypeError, ValueError):
                return None, "ocr_enhancement rotate action requires angle_degree"

        results.append(normalized)
        seen_region_ids.add(region_id)
    return results, None


def _render_parameters(
    parameters: dict[str, Any],
    *,
    refinement_action: dict[str, Any],
) -> dict[str, Any]:
    render_parameters: dict[str, Any] = {}
    artifact_dir = parameters.get("artifact_dir")
    if artifact_dir is not None:
        render_parameters["artifact_dir"] = artifact_dir

    mode = refinement_action["action"]
    if mode == "zoom":
        render_parameters["target_long_side_px"] = refinement_action["target_long_side_px"]
    elif mode == "crop":
        for key in ("left_px", "right_px", "top_px", "bottom_px"):
            render_parameters[key] = refinement_action[key]
    elif mode == "rotate":
        render_parameters["angle_degree"] = refinement_action["angle_degree"]
    return render_parameters


def _ocr_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    ocr_parameters: dict[str, Any] = {}
    if parameters.get("artifact_dir") is not None:
        ocr_parameters["artifact_dir"] = parameters.get("artifact_dir")
    if parameters.get("force") is not None:
        ocr_parameters["force"] = parameters.get("force")
    return ocr_parameters


def _build_enhanced_ocr_target(
    *,
    mode: str,
    view_result: dict[str, Any],
) -> dict[str, Any]:
    artifact_path = str(view_result["artifact_path"])
    page_index = int(view_result["page_index"])
    region_id = str(view_result["region_id"])
    data: dict[str, Any] = {
        "page_index": page_index,
        "region_id": region_id,
    }
    if mode == "zoom":
        data["zoom_artifact_path"] = artifact_path
        data["target_long_side_px"] = view_result.get("target_long_side_px")
    elif mode == "crop":
        data["crop_artifact_path"] = artifact_path
        data["left_px"] = view_result.get("left_px")
        data["right_px"] = view_result.get("right_px")
        data["top_px"] = view_result.get("top_px")
        data["bottom_px"] = view_result.get("bottom_px")
    else:
        data["rotate_artifact_path"] = artifact_path
        data["angle_degree"] = view_result.get("angle_degree")
    return {
        "page_index": page_index,
        "artifact_name": f"{mode}_region_{region_id}",
        "artifact_path": artifact_path,
        "data": data,
    }
