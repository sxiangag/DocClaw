# DocClaw Package Overview

This directory contains the DocClaw runtime implementation. The package is
organized around a planner-executor loop over document state: a planner selects
one action, the matching tool executes it, and the resulting observation updates
the document state before the next action is chosen.

## Main Entry Points

- `docclaw.py`: high-level application facade. `DocClaw.from_config()` loads
  `docclaw.toml`, builds providers, registers enabled tools, and constructs the
  runner.
- `cli.py`: command-line interface for one-shot runs and interactive chat.
- `__main__.py`: enables `python -m docclaw`.

## Runtime Components

- `agent/planner.py`: planner interfaces and the LLM-backed planner. The planner
  receives the current document state, active task skill, and available tool
  definitions, then emits the next structured action.
- `agent/executor.py`: dispatches actions to tools and applies observations to
  state through each tool's `update_state()` method.
- `agent/loop.py`: runs the repeated plan-execute-update loop until a final
  answer, explicit stop, failure, or step limit.
- `agent/runner.py`: creates per-run state, attaches session history, invokes
  the loop, and persists the resulting session transcript.
- `agent/utils.py`: shared dataclasses and serialization helpers for actions,
  observations, document state, page state, regions, artifacts, evidence, traces,
  and run results.

## Tools and Actions

All executable operations implement the `Tool` contract in `agent/tool/tool.py`.
The action names are the planner-facing interface and match the paper
supplementary material:

- `answer_from_evidence`
- `answer_json`
- `crop`
- `extract_evidence`
- `inspect_ocr`
- `internal_search`
- `ocr`
- `ocr_enhancement`
- `parse_chart`
- `parse_formula`
- `parse_layout`
- `parse_table`
- `rotate`
- `select_ocr`
- `select_pages`
- `stop`
- `transcribe`
- `understand_figures`
- `zoom`

Tool implementations live under `agent/tool/`:

- `layout/`, `ocr/`, `table/`, `chart/`, `formula/`, and `figure/` parse or
  understand document content.
- `zoom/`, `crop/`, `rotate/`, `enhancement/`, `inspect_ocr/`, and `select_ocr/`
  support OCR refinement.
- `select_pages/`, `internal_search/`, `evidence/`, and `answer/` support
  document QA and KIE workflows.

The enabled tool registry is built in `docclaw.py` from `DocClawConfig`.

## Document and State Flow

- `document/` loads PDFs and images into `DocumentState`.
- Runtime artifacts are stored under the configured storage root, grouped by
  document id.
- Tools add page text, layout regions, parsed tables, formulas, charts, figure
  answers, OCR refinement artifacts, and evidence records to the shared state.
- `exporter/markdown.py` renders page or document state into Markdown for OCR
  experiments.

## Providers and Configuration

- `provider/` contains LLM provider adapters.
- `config/` loads and validates `docclaw.toml`.
- Provider and tool backends are selected through config sections such as
  `[planner]`, `[providers.*]`, and `[tools.*]`.

## Skills

`skills/` contains task-level planner instructions for OCR, document QA, and
KIE. When enabled in config, the planner first selects a relevant skill and then
uses it as task-specific guidance during action planning.

## Sessions

`session/` persists compact interaction history across runs or chat turns.
Session history is optional but useful for interactive document QA.
