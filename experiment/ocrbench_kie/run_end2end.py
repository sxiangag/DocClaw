"""Run DocClaw on a prepared OCRBench KIE bundle."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docclaw import DocClaw
from docclaw.agent.utils import RunResult, _summarize_observation


PREPARED_BUNDLE_FORMAT_VERSION = "docclaw_ocrbench_kie_prepared_v1"
DEFAULT_PREPARED_JSON = "dataset/OCRBench/OCRBench_kie.json"
TRACE_SUMMARY_TEXT_LIMIT = 400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DocClaw on a prepared OCRBench KIE bundle.",
    )
    parser.add_argument(
        "--prepared-json",
        default=DEFAULT_PREPARED_JSON,
        help=(
            "Prepared bundle created by prepare_dataset.py. "
            f"Default: {DEFAULT_PREPARED_JSON}"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to incremental prediction output JSON.",
    )
    parser.add_argument(
        "--config",
        help="Optional DocClaw config path. Defaults to normal DocClaw config discovery.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start from sample index N within the prepared bundle.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of prepared samples to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help=(
            "Optional maximum planner steps override per sample. "
            "When omitted, uses runtime.max_steps from the DocClaw config."
        ),
    )
    parser.add_argument(
        "--session-id-prefix",
        default="ocrbench_kie",
        help="Prefix for per-sample session ids. Default: ocrbench_kie",
    )
    parser.add_argument(
        "--save-full-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist full result.trace for each sample. Default: disabled.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing output JSON instead of resuming from it.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop immediately on the first uncaught sample-level exception.",
    )
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    prepared_bundle = _load_prepared_bundle(args.prepared_json)
    selected_samples = _select_samples(
        prepared_bundle["samples"],
        offset=args.offset,
        limit=args.limit,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    compact_output_path = output_path.with_suffix(".compact.json")
    summary_path = output_path.with_suffix(".summary.json")

    existing_predictions = [] if args.overwrite else _load_existing_predictions(output_path)
    existing_by_sample_id = {
        str(item["sample_id"]): item
        for item in existing_predictions
        if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
    }
    predictions: list[dict[str, Any]] = [
        item for item in existing_predictions
        if isinstance(item, dict)
    ]

    total_selected = len(selected_samples)
    completed_before = sum(
        1 for sample in selected_samples if sample["sample_id"] in existing_by_sample_id
    )
    print(f"Selected samples: {total_selected}")
    print(f"Already present in output: {completed_before}")

    app = DocClaw.from_config(args.config)
    processed = 0
    pending_total = total_selected - completed_before

    for sample in selected_samples:
        if sample["sample_id"] in existing_by_sample_id:
            continue
        processed += 1
        print(f"[{processed}/{pending_total}] {sample['sample_id']} :: {sample['question']}")

        started_at = time.perf_counter()
        session_id = _session_id(prefix=args.session_id_prefix, sample_id=sample["sample_id"])
        try:
            document = app.load_document(
                sample["image_path_absolute"],
                document_id=sample["sample_id"],
            )
            session = app.runner.session_manager.create(
                session_id,
                metadata={
                    "document_id": document.document_id,
                    "ocrbench_id": sample.get("id"),
                    "sample_id": sample["sample_id"],
                },
            )
            result = await app.run(
                document,
                _task_prompt(sample),
                max_steps=args.max_steps,
                session=session,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            result_record = _build_exception_record(
                sample=sample,
                exception=exc,
                elapsed_seconds=time.perf_counter() - started_at,
                session_id=session_id,
            )
        else:
            result_record = _build_prediction_record(
                sample=sample,
                result=result,
                elapsed_seconds=time.perf_counter() - started_at,
                session_id=session_id,
                save_full_trace=args.save_full_trace,
            )

        existing_by_sample_id[sample["sample_id"]] = result_record
        predictions.append(result_record)
        _save_json(predictions, output_path)
        _save_json(_build_compact_predictions(predictions), compact_output_path)
        _save_json(
            _build_summary(
                prepared_json=args.prepared_json,
                output_json=output_path,
                config=args.config,
                max_steps=args.max_steps,
                total_selected=total_selected,
                predictions=predictions,
            ),
            summary_path,
        )

    print(f"Predictions: {output_path}")
    print(f"Compact predictions: {compact_output_path}")
    print(f"Summary: {summary_path}")
    return 0


def _load_prepared_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prepared bundle must be a JSON object")
    if payload.get("format_version") != PREPARED_BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported prepared bundle format_version: {payload.get('format_version')!r}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("prepared bundle must contain a top-level samples list")
    return payload


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = samples[offset : (offset + limit) if limit is not None else None]
    prepared: list[dict[str, Any]] = []
    for relative_index, item in enumerate(selected):
        if not isinstance(item, dict):
            continue
        bundle_index = offset + relative_index
        sample = dict(item)
        sample["bundle_index"] = bundle_index
        sample["sample_id"] = _sample_id(sample, bundle_index=bundle_index)
        image_path_absolute = sample.get("image_path_absolute")
        if not isinstance(image_path_absolute, str) or not image_path_absolute:
            raise ValueError(f"sample {sample['sample_id']} missing image_path_absolute")
        if not Path(image_path_absolute).expanduser().exists():
            raise FileNotFoundError(
                f"sample {sample['sample_id']} image not found: {image_path_absolute}"
            )
        prepared.append(sample)
    return prepared


def _sample_id(sample: dict[str, Any], *, bundle_index: int) -> str:
    raw_id = sample.get("id")
    if raw_id is not None:
        return f"ocrbench_kie_{bundle_index}_{raw_id}"
    return f"ocrbench_kie_bundle_{bundle_index}"


def _task_prompt(sample: dict[str, Any]) -> str:
    question = str(sample.get("question") or "")
    return (
        f"{question}\n"
        "This is a key information extraction task. Extract the requested "
        "information and return the result as one valid JSON object."
    )


def _load_existing_predictions(path: str | Path) -> list[dict[str, Any]]:
    output_path = Path(path).expanduser().resolve()
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), list):
            payload = payload["predictions"]
        else:
            raise ValueError("existing output JSON object must contain a predictions list")
    if not isinstance(payload, list):
        raise ValueError("existing output JSON must be a list or an object with predictions")
    return [item for item in payload if isinstance(item, dict)]


def _build_prediction_record(
    *,
    sample: dict[str, Any],
    result: RunResult,
    elapsed_seconds: float,
    session_id: str,
    save_full_trace: bool,
) -> dict[str, Any]:
    active_skill = result.state.get_active_skill()
    final_answer = result.answer or ""
    record = {
        "sample_id": sample["sample_id"],
        "bundle_index": sample["bundle_index"],
        "id": sample.get("id"),
        "dataset_name": sample.get("dataset_name"),
        "type": sample.get("type"),
        "image_path": sample.get("image_path"),
        "image_path_absolute": sample.get("image_path_absolute"),
        "question": sample.get("question"),
        "answers": sample.get("answers"),
        "predict": final_answer,
        "raw_response": final_answer,
        "final_answer": final_answer,
        "status": result.status,
        "reason": result.reason,
        "error": result.error,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "session_id": session_id,
        "selected_skill": active_skill.to_dict() if active_skill is not None else None,
        "input": {
            "sample_id": sample["sample_id"],
            "id": sample.get("id"),
            "dataset_name": sample.get("dataset_name"),
            "type": sample.get("type"),
            "image_path": sample.get("image_path"),
            "image_path_absolute": sample.get("image_path_absolute"),
            "question": sample.get("question"),
        },
        "trace_summary": [
            _summarize_trace_step(step)
            for step in result.trace
        ],
    }
    if save_full_trace:
        record["trace"] = [step.to_dict() for step in result.trace]
    return record


def _build_exception_record(
    *,
    sample: dict[str, Any],
    exception: Exception,
    elapsed_seconds: float,
    session_id: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "bundle_index": sample["bundle_index"],
        "id": sample.get("id"),
        "dataset_name": sample.get("dataset_name"),
        "type": sample.get("type"),
        "image_path": sample.get("image_path"),
        "image_path_absolute": sample.get("image_path_absolute"),
        "question": sample.get("question"),
        "answers": sample.get("answers"),
        "predict": "",
        "raw_response": "",
        "final_answer": "",
        "status": "failed",
        "reason": None,
        "error": f"{type(exception).__name__}: {exception}",
        "elapsed_seconds": round(elapsed_seconds, 4),
        "session_id": session_id,
        "selected_skill": None,
        "input": {
            "sample_id": sample["sample_id"],
            "id": sample.get("id"),
            "dataset_name": sample.get("dataset_name"),
            "type": sample.get("type"),
            "image_path": sample.get("image_path"),
            "image_path_absolute": sample.get("image_path_absolute"),
            "question": sample.get("question"),
        },
        "trace_summary": [],
    }


def _summarize_trace_step(step: Any) -> dict[str, Any]:
    observation = step.observation
    return {
        "step_index": step.step_index,
        "action_type": step.action.action_type,
        "target": step.action.target,
        "parameters": step.action.parameters,
        "rationale": step.action.rationale,
        "success": observation.success,
        "message": observation.message,
        "error": observation.error,
        "observation_summary": _summarize_observation(
            observation,
            text_limit=TRACE_SUMMARY_TEXT_LIMIT,
        ),
    }


def _build_summary(
    *,
    prepared_json: str,
    output_json: Path,
    config: str | None,
    max_steps: int | None,
    total_selected: int,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in predictions:
        status = item.get("status")
        if isinstance(status, str) and status:
            counts[status] += 1
            dataset_name = str(item.get("dataset_name") or "unknown")
            by_dataset[dataset_name][status] += 1
    return {
        "prepared_json": str(Path(prepared_json).expanduser().resolve()),
        "output_json": str(output_json),
        "compact_output_json": str(output_json.with_suffix(".compact.json")),
        "config": config,
        "max_steps": max_steps,
        "total_selected": total_selected,
        "prediction_count": len(predictions),
        "status_counts": dict(sorted(counts.items())),
        "status_counts_by_dataset": {
            dataset_name: dict(sorted(status_counts.items()))
            for dataset_name, status_counts in sorted(by_dataset.items())
        },
    }


def _build_compact_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in predictions:
        compact.append(
            {
                "dataset_name": item.get("dataset_name"),
                "type": item.get("type"),
                "id": item.get("id"),
                "image_path": item.get("image_path"),
                "image_path_absolute": item.get("image_path_absolute"),
                "question": item.get("question"),
                "answers": item.get("answers"),
                "predict": item.get("final_answer", ""),
                "status": item.get("status"),
                "error": item.get("error"),
            }
        )
    return compact


def _session_id(*, prefix: str, sample_id: str) -> str:
    safe_sample_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in sample_id
    )
    return f"{prefix}_{safe_sample_id}"


def _save_json(data: object, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
