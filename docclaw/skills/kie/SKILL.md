---
name: kie
description: Use for key information extraction tasks that ask for one or more named fields to be extracted from a document or an image and returned as JSON.
---

# KIE Skill

Use this skill when the user asks for one or more named fields from a document image.

This includes single-field extraction such as invoice number, total amount, or date, as well as multi-field key-value extraction. The final answer must be a JSON object.

## Primary Objective

Read the page, identify the reliable value for each requested field, and return exactly one JSON object grounded in the document text.

The task prompt is the source of truth for which fields are requested. Do not invent a schema beyond the field names requested in the prompt.

## Core Workflow

1. Run page-level `ocr` on the single input page.
2. Run `answer_json` with `mode="all"` to produce the final
   JSON object from the page image, OCR text, and verification over conflicting
   extracted values.

## Page Context Policy

- OCR is the grounding source for final answering.
- Use the page image, OCR text, and verification together to determine the
  requested values.
- Values should be copied from the document text as directly as possible.

## Output Policy

- The final answer must be a single JSON object only.
- Use exactly the field names requested by the task prompt.
- Do not add extra keys that were not requested.
- Do not include any explanation, prose, or citations outside the JSON object.
- Use `answer_json` for the final KIE answer generation.