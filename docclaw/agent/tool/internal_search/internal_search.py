"""Abstract internal search tool."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from docclaw.agent.tool.tool import Tool
from docclaw.agent.tool.internal_search.common import (
    ordered_hit_page_ids,
    ordered_hit_region_ids,
)
from docclaw.agent.utils import (
    Action,
    ActionType,
    Observation,
    RunState,
    SearchHint,
    SearchHistoryEntry,
    new_id,
    page_index_from_id,
    text_preview,
)


SEARCH_MODES = ("pages", "regions")
MIN_PAGE_SEARCH_TOP_K = 8


class InternalSearchToolBase(Tool):
    """Base class for tools that search inside the current document state."""

    @property
    def action_type(self) -> ActionType:
        return "internal_search"

    @property
    def description(self) -> str:
        return (
            "Search region-level units inside a narrowed page set. This is a "
            "text-only search tool: it narrows by region text, not visual elements. "
            "Use target.mode='regions' and target.page_ids to define the page "
            "scope, and use parameters.retriever='keyword' or 'text' to search "
            "regions. Provide a list of keyword or phrase variants in "
            "one action, and the tool searches each one then merges the ranked "
            "hits into one result set with search coverage details. Use this "
            "tool only after candidate pages are known. Results are top-k limited, so "
            "parameters.top_k must be set explicitly for each search. top_k is an "
            "upper bound on the final merged result set, not a guarantee that that "
            "many hits will be returned. When multiple query variants are "
            "provided, their hits are merged and the combined result is then "
            "truncated to top_k. Smaller top_k values are better for local "
            "narrowing. The immediate observation returns hit_page_ids and "
            "hit_region_ids for planner targeting, while richer per-hit hints "
            "are preserved in task-level search_history for later planner use. If "
            "the narrowed page scope is not yet ready for region-level text search, "
            "the tool can auto-prepare that scope before retrying. Repeating "
            "near-duplicate searches on the same narrowed scope usually has no value."
        )

    @property
    def target_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Required narrowed region-search target.",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Search mode over current searchable units.",
                    "enum": ["regions"],
                },
                "page_ids": {
                    "type": "array",
                    "description": "Explicit page set restriction for the region search.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["mode", "page_ids"],
            "additionalProperties": False,
        }

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "Internal search controls.",
            "properties": {
                "retriever": {
                    "type": "string",
                    "description": "Required retrieval backend for this search. Use 'keyword' or 'text' for region-level narrowing.",
                },
                "queries": {
                    "type": "array",
                    "description": "Keyword or phrase variants to search for inside the current document state.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of region hits to return.",
                    "minimum": 1,
                },
                "include_snippets": {
                    "type": "boolean",
                    "description": "Whether to return snippets in the ranked hits.",
                },
            },
            "required": ["retriever", "queries", "top_k"],
            "additionalProperties": False,
        }

    def document_overview_fragment(self, state: RunState) -> dict[str, Any]:
        from docclaw.retrieval.corpus import SearchCorpus

        corpus = SearchCorpus.from_document_state(state.document)
        page_nodes = [node for node in corpus.nodes if node.node_type == "page"]
        region_nodes = [node for node in corpus.nodes if node.node_type == "region"]
        return {
            "search": {
                "page_nodes": {
                    "page_indexes": sorted({node.page_index for node in page_nodes}),
                },
                "region_nodes": {
                    "page_indexes": sorted({node.page_index for node in region_nodes}),
                },
            }
        }

    async def execute(self, state: RunState, action: Action) -> Observation:
        raw_retriever = action.parameters.get("retriever")
        if not isinstance(raw_retriever, str) or not raw_retriever.strip():
            return self.error(
                action,
                "internal_search requires an explicit retriever. Use 'keyword' or 'text' for region-level narrowing.",
            )
        raw_queries = action.parameters.get("queries")
        if not isinstance(raw_queries, list) or not raw_queries:
            return self.error(action, "internal_search requires a non-empty queries parameter")
        queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        if not queries:
            return self.error(action, "internal_search requires at least one non-empty query")
        raw_top_k = action.parameters.get("top_k")
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            return self.error(action, "internal_search requires a positive integer top_k")
        if top_k <= 0:
            return self.error(action, "internal_search requires a positive integer top_k")

        try:
            mode, explicit_page_indices = normalize_search_target(
                action.target,
                document=state.document,
            )
        except Exception as exc:
            return self.error(action, str(exc))
        top_k = normalize_internal_search_top_k(top_k, mode=mode)

        retriever_name = {
            "lexical": "keyword",
            "semantic": "text",
        }.get(str(raw_retriever).strip().lower(), str(raw_retriever).strip().lower())
        auto_recovery_steps: list[dict[str, Any]] = []
        try:
            payload = self.search_internal(state, action)
        except Exception as exc:
            error_message = str(exc)
            if not self._should_auto_recover(
                retriever_name=retriever_name,
                mode=mode,
                error_message=error_message,
            ):
                return self.error(action, error_message)
            recovery_error = await self._auto_prepare_search_scope(
                state,
                mode=mode,
                explicit_page_indices=explicit_page_indices,
                recovery_steps=auto_recovery_steps,
            )
            if recovery_error is not None:
                return self.error(action, recovery_error)
            try:
                payload = self.search_internal(
                    state,
                    action,
                    enforce_page_text_coverage=False if mode == "pages" else True,
                )
            except Exception as retry_exc:
                return self.error(action, str(retry_exc))

        if not isinstance(payload.get("search_id"), str) or not str(payload.get("search_id")).strip():
            payload["search_id"] = new_id("search")

        hits = payload.get("hits")
        searched_pages = payload.get("searched_pages")
        searched_node_count = payload.get("searched_node_count")
        searchable_count = (
            searched_node_count if isinstance(searched_node_count, int) else 0
        )
        page_count = len(searched_pages) if isinstance(searched_pages, list) else 0
        hit_count = len(hits) if isinstance(hits, list) else 0

        if searchable_count == 0 and not auto_recovery_steps and self._should_auto_recover(
            retriever_name=retriever_name,
            mode=mode,
            error_message="No searchable document units available",
        ):
            recovery_error = await self._auto_prepare_search_scope(
                state,
                mode=mode,
                explicit_page_indices=explicit_page_indices,
                recovery_steps=auto_recovery_steps,
            )
            if recovery_error is not None:
                return self.error(action, recovery_error)
            try:
                payload = self.search_internal(
                    state,
                    action,
                    enforce_page_text_coverage=False if mode == "pages" else True,
                )
            except Exception as retry_exc:
                return self.error(action, str(retry_exc))
            if not isinstance(payload.get("search_id"), str) or not str(payload.get("search_id")).strip():
                payload["search_id"] = new_id("search")
            hits = payload.get("hits")
            searched_pages = payload.get("searched_pages")
            searched_node_count = payload.get("searched_node_count")
            searchable_count = (
                searched_node_count if isinstance(searched_node_count, int) else 0
            )
            page_count = len(searched_pages) if isinstance(searched_pages, list) else 0
            hit_count = len(hits) if isinstance(hits, list) else 0

        if searchable_count == 0:
            message = "No searchable document units available; run OCR, layout analysis, or parsing first."
        elif hit_count == 0:
            message = (
                f"Searched {searchable_count} unit(s) across {page_count} page(s) and found no matching hits."
            )
        else:
            message = (
                f"Found {hit_count} hit(s) across {page_count} page(s) from {searchable_count} searchable unit(s)."
            )
        if auto_recovery_steps:
            payload["auto_recovery"] = auto_recovery_steps
            recovery_kinds = ", ".join(step["action_type"] for step in auto_recovery_steps)
            message = f"{message} Auto-recovery applied: {recovery_kinds}."

        return Observation(
            action_id=action.action_id,
            success=True,
            data=payload,
            message=message,
        )

    def _should_auto_recover(
        self,
        *,
        retriever_name: str,
        mode: str,
        error_message: str,
    ) -> bool:
        if retriever_name not in {"keyword", "text"}:
            return False
        normalized = str(error_message).strip().lower()
        if not normalized:
            return False
        if "no searchable document units available" in normalized:
            return True
        if mode == "pages":
            return "requires searchable page text" in normalized
        if mode == "regions":
            return "requires searchable region text" in normalized
        return False

    async def _auto_prepare_search_scope(
        self,
        state: RunState,
        *,
        mode: str,
        explicit_page_indices: list[int] | None,
        recovery_steps: list[dict[str, Any]],
    ) -> str | None:
        page_indices = explicit_page_indices or [page.page_index for page in state.document.pages]
        if not page_indices:
            return "internal_search auto-recovery requires at least one page in scope"

        if mode == "pages":
            ocr_tool = getattr(self, "ocr_tool", None)
            if ocr_tool is None:
                return "internal_search page-mode auto-recovery requires an OCR tool"
            return await self._run_recovery_action(
                state,
                tool=ocr_tool,
                action=Action(
                    action_type="ocr",
                    target={"mode": "page", "page_indices": page_indices},
                    parameters={},
                    rationale="Auto-recovery for internal_search: add page OCR to the current page search scope.",
                ),
                recovery_steps=recovery_steps,
            )

        if mode == "regions":
            layout_tool = getattr(self, "layout_tool", None)
            ocr_tool = getattr(self, "ocr_tool", None)
            if layout_tool is None or ocr_tool is None:
                return "internal_search region-mode auto-recovery requires layout and OCR tools"
            layout_error = await self._run_recovery_action(
                state,
                tool=layout_tool,
                action=Action(
                    action_type="parse_layout",
                    target={"page_indices": page_indices},
                    parameters={},
                    rationale="Auto-recovery for internal_search: add layout before region search.",
                ),
                recovery_steps=recovery_steps,
            )
            if layout_error is not None:
                return layout_error
            return await self._run_recovery_action(
                state,
                tool=ocr_tool,
                action=Action(
                    action_type="ocr",
                    target={"mode": "region", "page_indices": page_indices},
                    parameters={},
                    rationale="Auto-recovery for internal_search: add region OCR to the current region search scope.",
                ),
                recovery_steps=recovery_steps,
            )

        return f"internal_search auto-recovery does not support mode={mode}"

    async def _run_recovery_action(
        self,
        state: RunState,
        *,
        tool: Tool,
        action: Action,
        recovery_steps: list[dict[str, Any]],
    ) -> str | None:
        observation = await tool.execute(state, action)
        recovery_steps.append(
            {
                "action_type": action.action_type,
                "target": dict(action.target),
                "message": observation.message,
                "success": observation.success,
            }
        )
        if not observation.success:
            return observation.error or f"{action.action_type} failed during internal_search auto-recovery"
        tool.update_state(state, action, observation)
        return None

    def update_state(
        self,
        state: RunState,
        action: Action,
        observation: Observation,
    ) -> None:
        data = observation.data
        if not isinstance(data, dict):
            return
        search_id = data.get("search_id")
        if not isinstance(search_id, str) or not search_id.strip():
            return

        hits = data.get("hits")
        hit_items = [item for item in hits if isinstance(item, dict)] if isinstance(hits, list) else []
        hit_page_ids = ordered_hit_page_ids(hit_items)
        hit_region_ids = ordered_hit_region_ids(hit_items)
        hints: list[SearchHint] = []
        for item in hit_items:
            raw_snippets = item.get("snippets")
            snippets = [
                str(snippet).strip()
                for snippet in raw_snippets
                if isinstance(snippet, str) and str(snippet).strip()
            ] if isinstance(raw_snippets, list) else []
            snippet_preview = text_preview(" | ".join(snippets), limit=60) if snippets else None
            hints.append(
                SearchHint(
                    page_index=item.get("page_index") if isinstance(item.get("page_index"), int) else None,
                    region_id=(
                        str(item["region_id"])
                        if isinstance(item.get("region_id"), str) and item.get("region_id")
                        else None
                    ),
                    score=float(item["score"]) if isinstance(item.get("score"), (int, float)) else None,
                    matched_queries=[
                        str(query)
                        for query in item.get("matched_queries", [])
                        if isinstance(query, str)
                    ],
                    snippet_preview=snippet_preview,
                )
            )

        state.add_search_history_entry(
            SearchHistoryEntry(
                search_id=search_id,
                action_id=action.action_id,
                queries=[
                    str(query)
                    for query in data.get("queries", [])
                    if isinstance(query, str)
                ],
                mode=str(data["mode"]) if isinstance(data.get("mode"), str) else None,
                searched_pages=[
                    int(page_index)
                    for page_index in data.get("searched_pages", [])
                    if isinstance(page_index, int)
                ] if isinstance(data.get("searched_pages"), list) else [],
                hit_page_ids=hit_page_ids,
                hit_region_ids=hit_region_ids,
                hit_count=len(hit_items),
                hints=hints,
            )
        )

    @abstractmethod
    def search_internal(
        self,
        state: RunState,
        action: Action,
    ) -> dict[str, Any]:
        """Return ranked internal search results and coverage summary."""


def normalize_search_target(
    target: dict[str, Any],
    *,
    document: Any | None = None,
) -> tuple[str, list[int] | None]:
    mode = normalize_search_mode(target.get("mode"))

    explicit_page_ids = _normalize_str_list(
        target.get("page_ids"),
        field_name="page_ids",
    )
    explicit_page_indices = (
        [page_index_from_id(item, document=document) for item in explicit_page_ids]
        if explicit_page_ids is not None
        else _normalize_int_list(
            target.get("page_indices"),
            field_name="page_indices",
        )
    )
    if target.get("region_ids") is not None:
        raise ValueError("internal_search target.region_ids is not supported; use target.page_ids")

    return mode, explicit_page_indices


def normalize_search_mode(value: Any) -> str:
    mode = str(value).strip() if value is not None else ""
    if not mode:
        raise ValueError("internal_search requires target.mode")
    if mode not in SEARCH_MODES:
        raise ValueError(f"unsupported search mode: {mode}")
    return mode


def normalize_internal_search_top_k(top_k: int, *, mode: str) -> int:
    if mode == "pages":
        return max(MIN_PAGE_SEARCH_TOP_K, top_k)
    return top_k


def _normalize_int_list(value: Any, *, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"internal_search {field_name} must be a non-empty list")

    items: list[int] = []
    seen: set[int] = set()
    for raw_item in value:
        item = int(raw_item)
        if item in seen:
            continue
        items.append(item)
        seen.add(item)
    return items


def _normalize_str_list(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"internal_search {field_name} must be a non-empty list")

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in value:
        item = str(raw_item).strip()
        if not item:
            continue
        if item in seen:
            continue
        items.append(item)
        seen.add(item)
    if not items:
        raise ValueError(f"internal_search {field_name} must contain non-empty strings")
    return items
