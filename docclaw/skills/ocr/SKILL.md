---
name: ocr
description: OCR-oriented task policy for transcription and recognition-heavy document tasks.
---

# OCR Skill

Use this skill when the task is mainly about reading document text correctly, including transcription, OCR quality improvement, OCR coverage, or recovering text from a specific page or region.

## Primary Objective

Maximize recognition quality and requested text coverage before producing the final transcription.

## Core Workflow

1. Use `parse_layout` first on relevant pages that have not yet been structurally parsed.
2. After layout is available, treat regions as the default recognition unit and prefer batched region OCR over page-level OCR.
3. The first `ocr` pass should cover all parsed regions rather than only regions that appear text-like.
4. After the first region OCR pass, call `inspect_ocr` on the relevant regions.
5. If inspection returns `refinement_actions`, run `ocr_enhancement(target.refinement_actions=...)` on the corresponding regions.
6. If refinement produces multiple candidates, call `select_ocr(target.region_ids=...)` before final transcription.
7. Use `transcribe` once recognition quality and coverage are sufficient for the user request.

## Recognition Policy

- OCR processing is **layout-first** and **region-first**.
- Prefer region-level OCR whenever usable layout regions are available.
- Use page-level OCR only as a fallback when layout parsing fails.
- Do not repeat document-wide OCR when only a local page or region requires recovery.
- When multiple regions require the same operation, prefer batched processing.

## Refinement Policy

Refine a region when any of the following applies:

- the text is small, dense, rotated, faint, or visually cluttered
- recognized text is truncated, garbled, or incomplete
- requested content is missing after the first OCR pass
- a smaller region is likely to be more readable than the full page
- `inspect_ocr` flags the region for refinement

Use the first-pass region outputs and the quality inspection results as the main refinement signals. If refinement actions are proposed, apply them before final transcription and resolve competing candidates with `select_ocr`.

Do not call `transcribe` immediately after the first region OCR pass when quality inspection or unresolved refinement candidates remain.

## Output and Stopping Policy

- Use `transcribe` as the finalization step for page transcription.
- Stop when the requested page or region has sufficient recognition quality and coverage for the user task.
- For region-specific OCR, stop once the target regions are readable enough to satisfy the request.
