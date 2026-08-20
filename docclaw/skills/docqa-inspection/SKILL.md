---
name: docqa-inspection
description: Use for DocQA tasks whose answer should be found within a narrowed local scope, including local lookup, local comparison, and local reasoning after retrieval and inspection.
---

# DocQA Inspection Skill

Use this skill for DocQA tasks whose answer should come from a narrowed local scope, including local lookup, comparison, reasoning, and visual grounding.

## Primary Objective

Identify the local scope that contains the answer, prepare the evidence required by the query, and answer from grounded evidence.

## Core Workflow

1. Run `select_pages` with `parameters.mode="coarse"` over a broad page set to obtain high-recall candidate pages.
2. Run page-mode `ocr` on the selected pages to make their textual content available for subsequent retrieval.
3. Run `select_pages` with `parameters.mode="refine"` on the OCR-prepared pages to obtain a higher-precision candidate set.
4. Inspect the refined candidate scope using the tool appropriate for the required evidence:
   - `ocr`, `parse_table`, or `parse_layout` for exact text, labels, rows, values, tables, or local structure
   - `understand_figures` with `parameters.mode="inspection"` for charts, figures, photos, maps, or other visual semantics
5. Run `extract_evidence` on the best available candidate scope.
6. If `answerability_status="answerable"`, use `answer_from_evidence` to answer the question.
7. If `answerability_status="inconclusive"`, use `missing_information` to determine what evidence remains missing, expand or refine retrieval accordingly, prepare the missing evidence, and run `extract_evidence` again.

## Retrieval Policy

- Retrieval is **narrow-first**.
- Complete the initial `select_pages(mode="coarse") -> ocr(mode="page") -> select_pages(mode="refine")` sequence before inspection, evidence extraction, or final answering.
- After the initial retrieval sequence, use `internal_search` for text-grounded follow-up retrieval or local narrowing when additional evidence is needed.
- Use region-level search only after candidate pages are known.
- When multiple related query variants belong to the same retrieval step, prefer one batched search.
- Adjust `top_k` to the scope of the task: use a moderate value for localized single-fact questions and a broader value for bounded cross-page comparison, aggregation, or synthesis.
- If `missing_information` indicates that the current scope is insufficient, rewrite it into an appropriate retrieval query and either rerun `select_pages(mode="coarse")` on a broader page set or use `internal_search(retriever=keyword)`, depending on the missing evidence.
- When explicit page numbers are mentioned in the query or required in the answer, interpret them as displayed page references rather than physical PDF indices and use `select_pages` to locate the corresponding pages.

## Inspection Policy

- For exact text, labels, rows, values, table cells, or local document structure, use the corresponding parsing tools rather than relying on visual understanding alone.
- For visual semantics, use `understand_figures(mode="inspection")` and ask for the specific page-level judgment required by the query.
- When several candidate pages require the same visual inspection, prefer one batched `understand_figures` call over inspecting pages individually.
- If textual evidence remains incomplete and the missing information may be visual, inspect the candidate pages with `understand_figures`.
- Repeated OCR or layout parsing on the same scope does not by itself provide new evidence.

## Evidence and Finalization Policy

- `extract_evidence` is the required gate before final answering.
- Prefer narrowed candidate regions as the evidence scope once the relevant pages are known.
- When multiple pages provide complementary evidence, combine them in one `extract_evidence` call rather than assessing them independently.
- For figure-anchored questions, retain the relevant page and supporting text regions in the evidence scope when needed.
- Treat `answerability_status="inconclusive"` as a signal that the current evidence scope is insufficient, not as a document-level verdict.
- If plausible missing evidence remains outside the current local scope, expand retrieval before finalizing.

## Not Answerable

- Use `Not answerable` only after a successful `extract_evidence` step whose latest assessment remains inconclusive.
- Immediately before stopping with `Not answerable`, run exactly one final page-level evidence recheck by calling `extract_evidence` with `parameters.mode="not_answerable_recheck"` on the candidate pages most likely to contain the missing answer.
- Before stopping, ensure that the main local candidates have been inspected and that the evidence type required by the query has actually been prepared.
- If the final reassessment remains inconclusive, keep `stop.answer` exactly as `Not answerable` and record the explored scope and remaining missing information in `stop.reason`.