"""Shared helpers for internal search retrievers and tools."""

from __future__ import annotations

from typing import Any

from docclaw.agent.utils import text_preview
from docclaw.retrieval.corpus import SearchHit, SearchNode, SkippedUnit, normalize_search_text


def normalize_queries(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("internal_search queries must be a non-empty list")
    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in value:
        query = str(raw_query).strip()
        if not query:
            continue
        normalized = normalize_search_text(query)
        if not normalized or normalized in seen:
            continue
        queries.append(query)
        seen.add(normalized)
    if not queries:
        raise ValueError("internal_search queries must contain at least one non-empty string")
    return queries


def merge_query_hits(
    hits_by_query: list[list[SearchHit]],
    *,
    queries: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for query, hits in zip(queries, hits_by_query, strict=False):
        for hit in hits:
            key = hit.node.node_id
            current = merged.get(key)
            if current is None:
                payload = hit.to_dict()
                payload["snippets"] = list(hit.metadata.get("snippets") or [])
                payload["matched_queries"] = [query]
                merged[key] = payload
                continue
            matched_queries = current.setdefault("matched_queries", [])
            if query not in matched_queries:
                matched_queries.append(query)
            current_snippets = current.setdefault("snippets", [])
            for snippet in hit.metadata.get("snippets") or []:
                if snippet not in current_snippets:
                    current_snippets.append(snippet)
            current_score = float(current.get("score", 0.0))
            if hit.score > current_score:
                payload = hit.to_dict()
                payload["snippets"] = current_snippets
                payload["matched_queries"] = matched_queries
                merged[key] = payload

    ordered_hits = sorted(
        merged.values(),
        key=lambda item: (
            -len(item.get("matched_queries", [])),
            -float(item.get("score", 0.0)),
            int(item.get("page_index", 0)),
            str(item.get("node_id", "")),
        ),
    )[:top_k]
    for index, hit in enumerate(ordered_hits, start=1):
        hit["rank"] = index
    return ordered_hits


def ordered_hit_page_ids(hits: list[dict[str, Any]]) -> list[int]:
    ordered_pages: list[int] = []
    seen: set[int] = set()
    for hit in hits:
        page_index = hit.get("page_index")
        if not isinstance(page_index, int) or page_index in seen:
            continue
        ordered_pages.append(page_index)
        seen.add(page_index)
    return ordered_pages


def ordered_hit_region_ids(hits: list[dict[str, Any]]) -> list[str]:
    ordered_regions: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        region_id = hit.get("region_id")
        if not isinstance(region_id, str) or not region_id or region_id in seen:
            continue
        ordered_regions.append(region_id)
        seen.add(region_id)
    return ordered_regions


def matches_mode(node: SearchNode, mode: str) -> bool:
    if mode == "pages":
        return node.node_type == "page"
    if mode == "regions":
        return node.node_type == "region"
    return False


def matches_target(
    node: SearchNode,
    page_filter: set[int] | None,
) -> bool:
    if page_filter is not None and node.page_index not in page_filter:
        return False
    return True


def matches_skipped_target(
    skipped: SkippedUnit,
    mode: str,
    page_filter: set[int] | None,
) -> bool:
    if mode == "pages" and skipped.unit_type != "page":
        return False
    if mode == "regions" and skipped.unit_type != "region":
        return False
    if page_filter is not None and skipped.page_index not in page_filter:
        return False
    return True


def build_snippets(
    node: SearchNode,
    query: str,
    *,
    left_chars: int = 50,
    right_chars: int = 80,
    max_snippets: int | None = None,
) -> list[str]:
    if not node.has_text or node.text is None:
        return []
    query_text = str(query).strip()
    if not query_text:
        preview = text_preview(node.text, limit=left_chars + right_chars)
        return [preview] if preview else []

    raw_lower = node.text.casefold()
    needles = [query_text.casefold()]
    if " " in query_text:
        needles.extend(term.casefold() for term in query_text.split() if term.strip())

    snippets: list[str] = []
    seen: set[str] = set()
    for needle in needles:
        if not needle:
            continue
        start_index = 0
        while True:
            raw_index = raw_lower.find(needle, start_index)
            if raw_index < 0:
                break
            start = max(0, raw_index - left_chars)
            end = min(len(node.text), raw_index + len(needle) + right_chars)
            snippet = node.text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(node.text):
                snippet = snippet + "..."
            if snippet and snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
                if max_snippets is not None and len(snippets) >= max_snippets:
                    return snippets
            start_index = raw_index + max(1, len(needle))

    if snippets:
        return snippets

    preview = text_preview(node.text, limit=left_chars + right_chars)
    return [preview] if preview else []
