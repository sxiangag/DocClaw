---
name: docqa-enumeration
description: Use for broad-scope logical-item counting tasks where the answer depends on covering and counting a document-wide or cross-page set with correct grouping, deduplication, and membership.
---

# DocQA Enumeration Skill

Use this skill only when the task requires counting a broad logical set across the document. The main challenge should be establishing sufficient coverage and counting the correct logical items, rather than retrieving a local value.
Do not use this skill merely because the query contains `how many`. Questions that count items within one figure, operate on an already-given set, or derive a number from a few local values are local lookup or reasoning tasks.

## Primary Objective

Establish the full relevant counting set, determine the correct counting unit and membership rules, inspect the set broadly enough, and derive the final count from grounded evidence.

## Core Workflow

1. Identify the logical item being counted and any grouping, deduplication, membership, or exclusion rules.
2. Maintain a scope broad enough to cover the full candidate set.
3. For visually grounded counting, use `select_pages` with `parameters.mode="coarse"` across the broad candidate pages to remove clearly irrelevant pages while retaining plausible contributors.
4. Inspect the candidate set using the tool appropriate for the item type: `parse_layout`, `ocr`, or `understand_figures` with `parameters.mode="enumeration"`.
5. Run `extract_evidence` on a scope that covers the full relevant candidate set.
6. Use `answer_from_evidence` only after the counting set has been covered sufficiently to finalize.

## Coverage and Retrieval Policy

- Enumeration is **coverage-first**, not **narrow-first**.
- Do not use `internal_search` to establish the counting set.
- For questions about all charts, tables, pages, figures, references, or similar document-wide sets, preserve broad scope until the relevant set is sufficiently covered.
- Do not treat search hits or exact keyword matches as the counted set or as sufficient evidence of membership.
- If plausible unexplored candidates remain, continue exploration rather than finalizing from the currently observed subset.

## Inspection Policy

- Determine the logical counting unit before counting, such as a page, figure, chart, table, panel group, photo, reference, or another item.
- Count logical items rather than raw regions, panels, fragments, or sub-parts.
- Do not assume that the requested item type must exactly match a layout label. For visually grounded tasks, consider all plausible image-like regions, including labels such as `chart`, `table`, `image`, `seal`, `header_image`, and `footer_image`.
- For visually grounded counting, prefer page-level `understand_figures` over raw region counting, use `parameters.mode="enumeration"`, and ask what each page contributes to the final count.

## Evidence and Finalization Policy

- `extract_evidence` is the required gate before final answering.
- Treat `answerability_status="answerable"` as sufficient only when the relevant counting set has been covered well enough to finalize.
- Do not convert uncertainty about the counting set, counting unit, or membership into an unsupported numeric answer.

## Not Answerable

- Use `Not answerable` only after a successful `extract_evidence` step over a broad-enough scope whose latest assessment remains inconclusive.
- Immediately before stopping with `Not answerable`, run exactly one final page-level evidence recheck by calling `extract_evidence` with `parameters.mode="not_answerable_recheck"` on the candidate pages most likely to contain the missing answer. Use this only for the final reassessment.
- If the document still does not provide enough information to establish and verify the counting set, keep `stop.answer` exactly as `Not answerable` and record the explored scope, remaining gaps, and failure reason in `stop.reason`.