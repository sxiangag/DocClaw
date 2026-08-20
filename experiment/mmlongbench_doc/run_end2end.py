"""Run DocClaw on a prepared MMLongBench-Doc bundle."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docclaw import DocClaw
from docclaw.agent.utils import RunResult, _summarize_observation, page_index_from_id


PREPARED_BUNDLE_FORMAT_VERSION = "docclaw_mmlongbench_doc_prepared_v1"
TRACE_SUMMARY_TEXT_LIMIT = 400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DocClaw on a prepared MMLongBench-Doc bundle. "
            "Documents are loaded once per unique document_path and reused "
            "across all questions for that document."
        ),
    )
    parser.add_argument(
        "--prepared-json",
        required=True,
        help="Prepared bundle created by prepare_dataset.py.",
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
        "--documents-dir",
        help=(
            "Optional document root override. When set, each sample uses "
            "<documents-dir>/<doc_id> instead of sample.document_path."
        ),
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
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Maximum number of documents to process concurrently. "
            "Questions within the same document are still run sequentially. "
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help=(
            "Optional maximum planner steps override per question. "
            "When omitted, uses runtime.max_steps from the DocClaw config."
        ),
    )
    parser.add_argument(
        "--session-id-prefix",
        default="mmlongbench_doc",
        help="Prefix for per-document session ids. Default: mmlongbench_doc",
    )
    parser.add_argument(
        "--carry-session-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Carry multi-turn session history across questions for the same document. "
            "Default: disabled; document state is still reused."
        ),
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
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
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

    grouped_samples = _group_samples_by_document(
        selected_samples,
        documents_dir_override=args.documents_dir,
    )
    total_selected = len(selected_samples)
    completed_before = sum(
        1
        for sample in selected_samples
        if sample["sample_id"] in existing_by_sample_id
    )
    print(f"Selected samples: {total_selected}")
    print(f"Already present in output: {completed_before}")
    pending_groups = [
        (document_path_str, document_samples)
        for document_path_str, document_samples in grouped_samples
        if any(sample["sample_id"] not in existing_by_sample_id for sample in document_samples)
    ]
    shared_state = {
        "processed": 0,
        "pending_total": total_selected - completed_before,
    }
    output_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.n_jobs)

    async def run_document_group(
        document_path_str: str,
        document_samples: list[dict[str, Any]],
    ) -> None:
        async with semaphore:
            await _process_document_group(
                document_path_str=document_path_str,
                document_samples=document_samples,
                args=args,
                existing_by_sample_id=existing_by_sample_id,
                predictions=predictions,
                output_path=output_path,
                compact_output_path=compact_output_path,
                summary_path=summary_path,
                total_selected=total_selected,
                output_lock=output_lock,
                shared_state=shared_state,
            )

    await asyncio.gather(
        *(run_document_group(document_path_str, document_samples) for document_path_str, document_samples in pending_groups)
    )

    print(f"Predictions: {output_path}")
    print(f"Compact predictions: {compact_output_path}")
    print(f"Summary: {summary_path}")
    return 0


async def _process_document_group(
    *,
    document_path_str: str,
    document_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    existing_by_sample_id: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    compact_output_path: Path,
    summary_path: Path,
    total_selected: int,
    output_lock: asyncio.Lock,
    shared_state: dict[str, int],
) -> None:
    pending_samples = [
        sample
        for sample in document_samples
        if sample["sample_id"] not in existing_by_sample_id
    ]
    if not pending_samples:
        return

    document_path = Path(document_path_str)
    if not document_path.exists():
        raise FileNotFoundError(f"document not found: {document_path}")

    app = DocClaw.from_config(args.config)
    document = app.load_document(
        document_path,
        document_id=str(pending_samples[0]["doc_id"]),
    )
    session = app.runner.session_manager.create(
        _session_id(
            prefix=args.session_id_prefix,
            doc_id=str(pending_samples[0]["doc_id"]),
        ),
        metadata={"document_id": document.document_id},
    )
    async with output_lock:
        print(
            f"Loaded document {document.document_id} once for "
            f"{len(document_samples)} question(s): {document_path}"
        )

    for sample in pending_samples:
        if not args.carry_session_history:
            session.reset_active_history()

        async with output_lock:
            shared_state["processed"] += 1
            processed = shared_state["processed"]
            print(
                f"[{processed}/{shared_state['pending_total']}] "
                f"{sample['sample_id']} :: {sample['question']}"
            )

        started_at = time.perf_counter()
        try:
            result = await app.run(
                document,
                sample["question"],
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
                session_id=session.session_id,
            )
        else:
            result_record = _build_prediction_record(
                sample=sample,
                result=result,
                elapsed_seconds=time.perf_counter() - started_at,
                session_id=session.session_id,
                save_full_trace=args.save_full_trace,
            )

        async with output_lock:
            existing_by_sample_id[sample["sample_id"]] = result_record
            predictions.append(result_record)
            _save_json(predictions, output_path)
            _save_json(
                _build_compact_predictions(predictions),
                compact_output_path,
            )
            _save_json(
                _build_summary(
                    prepared_json=args.prepared_json,
                    output_json=output_path,
                    config=args.config,
                    documents_dir=args.documents_dir,
                    max_steps=args.max_steps,
                    carry_session_history=args.carry_session_history,
                    total_selected=total_selected,
                    predictions=predictions,
                    n_jobs=args.n_jobs,
                ),
                summary_path,
            )


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
    return [item for item in selected if isinstance(item, dict)]


def _group_samples_by_document(
    samples: list[dict[str, Any]],
    *,
    documents_dir_override: str | None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        document_path = _resolve_document_path(
            sample,
            documents_dir_override=documents_dir_override,
        )
        grouped[str(document_path)].append(sample)
    return list(grouped.items())


def _resolve_document_path(
    sample: dict[str, Any],
    *,
    documents_dir_override: str | None,
) -> Path:
    if documents_dir_override:
        return (
            Path(documents_dir_override).expanduser().resolve()
            / str(sample["doc_id"])
        )
    return Path(str(sample["document_path"])).expanduser().resolve()


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
    record = {
        "sample_id": sample["sample_id"],
        "doc_id": sample["doc_id"],
        "document_path": sample["document_path"],
        "question": sample["question"],
        "answer": sample["answer"],
        "answer_format": sample["answer_format"],
        "raw_response": result.answer or "",
        "final_answer": result.answer,
        "status": result.status,
        "reason": result.reason,
        "error": result.error,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "session_id": session_id,
        "selected_skill": active_skill.to_dict() if active_skill is not None else None,
        "input": {
            "sample_id": sample["sample_id"],
            "doc_id": sample["doc_id"],
            "document_path": sample["document_path"],
            "question": sample["question"],
        },
        "trace_summary": [
            _summarize_trace_step(step, document=result.state.document, state=result.state)
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
        "doc_id": sample["doc_id"],
        "document_path": sample["document_path"],
        "question": sample["question"],
        "answer": sample["answer"],
        "answer_format": sample["answer_format"],
        "raw_response": "",
        "final_answer": None,
        "status": "failed",
        "reason": None,
        "error": f"{type(exception).__name__}: {exception}",
        "elapsed_seconds": round(elapsed_seconds, 4),
        "session_id": session_id,
        "selected_skill": None,
        "input": {
            "sample_id": sample["sample_id"],
            "doc_id": sample["doc_id"],
            "document_path": sample["document_path"],
            "question": sample["question"],
        },
        "trace_summary": [],
    }


def _summarize_trace_step(step: Any, *, document: Any, state: Any) -> dict[str, Any]:
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
        "page_debug": _safe_trace_step_page_debug(step, document=document, state=state),
    }


def _safe_trace_step_page_debug(step: Any, *, document: Any, state: Any) -> dict[str, list[int]]:
    try:
        return _trace_step_page_debug(step, document=document, state=state)
    except Exception:
        return {
            "target_page_indices": [],
            "candidate_page_indices": [],
            "output_page_indices": [],
            "selected_page_indices": [],
            "evidence_page_indices": [],
            "hit_page_indices": [],
        }


def _trace_step_page_debug(step: Any, *, document: Any, state: Any) -> dict[str, list[int]]:
    action_type = getattr(step.action, "action_type", None)
    target = step.action.target if isinstance(step.action.target, dict) else {}
    data = step.observation.data if isinstance(step.observation.data, dict) else {}

    target_page_indices = _page_indices_from_target(target)
    page_debug: dict[str, list[int]] = {
        "target_page_indices": target_page_indices,
        "candidate_page_indices": [],
        "output_page_indices": [],
        "selected_page_indices": [],
        "evidence_page_indices": [],
        "hit_page_indices": [],
    }

    if action_type == "select_pages":
        page_debug["candidate_page_indices"] = list(target_page_indices)
        page_debug["selected_page_indices"] = _selected_page_indices_from_select_pages(
            data,
            target_page_indices=target_page_indices,
            document=document,
        )
        page_debug["output_page_indices"] = list(page_debug["selected_page_indices"])
    elif action_type in {"ocr", "understand_figures", "parse_layout", "parse_table", "parse_chart", "parse_formula", "transcribe"}:
        page_debug["output_page_indices"] = _page_indices_from_results(data.get("results"))
        if not page_debug["output_page_indices"]:
            page_debug["output_page_indices"] = _page_indices_from_pages(data.get("pages"))
    elif action_type == "extract_evidence":
        page_debug["evidence_page_indices"] = _page_indices_from_evidence(data.get("evidence"))
        page_debug["output_page_indices"] = list(page_debug["evidence_page_indices"])
    elif action_type == "internal_search":
        page_debug["hit_page_indices"] = _page_indices_from_values(data.get("hit_page_ids"))
        page_debug["output_page_indices"] = list(page_debug["hit_page_indices"])
    elif action_type == "answer_from_evidence":
        page_debug["evidence_page_indices"] = _page_indices_from_evidence_ids(
            data.get("evidence_ids"),
            state=state,
        )

    return {key: _dedupe_ints(value) for key, value in page_debug.items()}


def _page_indices_from_target(target: dict[str, Any]) -> list[int]:
    values: list[int] = []
    page_indices = target.get("page_indices")
    if isinstance(page_indices, list):
        values.extend(_page_indices_from_values(page_indices))
    page_index = target.get("page_index")
    if isinstance(page_index, int):
        values.append(page_index)
    return _dedupe_ints(values)


def _selected_page_indices_from_select_pages(
    data: dict[str, Any],
    *,
    target_page_indices: list[int],
    document: Any,
) -> list[int]:
    page_id_by_index = {
        f"page_{page_index + 1:03d}": page_index
        for page_index in target_page_indices
    }
    results = data.get("results")
    values: list[int] = []
    if not isinstance(results, list):
        return values
    for item in results:
        if not isinstance(item, dict):
            continue
        selected_page_ids = item.get("selected_page_ids")
        if not isinstance(selected_page_ids, list):
            continue
        for page_id in selected_page_ids:
            if not isinstance(page_id, str):
                continue
            page_index = _page_index_from_page_id(
                page_id,
                page_id_by_index=page_id_by_index,
                document=document,
            )
            if page_index is not None:
                values.append(page_index)
    return _dedupe_ints(values)


def _page_index_from_page_id(
    page_id: str,
    *,
    page_id_by_index: dict[str, int],
    document: Any,
) -> int | None:
    if page_id in page_id_by_index:
        return page_id_by_index[page_id]
    try:
        return page_index_from_id(page_id, document=document)
    except (TypeError, ValueError):
        return None


def _page_indices_from_results(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_index = item.get("page_index")
        if isinstance(page_index, int):
            values.append(page_index)
    return _dedupe_ints(values)


def _page_indices_from_pages(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_index = item.get("page_index")
        if isinstance(page_index, int):
            values.append(page_index)
    return _dedupe_ints(values)


def _page_indices_from_evidence(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_index = item.get("page_index")
        if isinstance(page_index, int):
            values.append(page_index)
    return _dedupe_ints(values)


def _page_indices_from_evidence_ids(value: Any, *, state: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    evidence_ids = {item for item in value if isinstance(item, str)}
    values: list[int] = []
    for item in getattr(state, "evidence", []) or []:
        if getattr(item, "evidence_id", None) not in evidence_ids:
            continue
        page_index = getattr(item, "page_index", None)
        if isinstance(page_index, int):
            values.append(page_index)
    return _dedupe_ints(values)


def _page_indices_from_values(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            values.append(item)
    return _dedupe_ints(values)


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _build_summary(
    *,
    prepared_json: str,
    output_json: Path,
    config: str | None,
    documents_dir: str | None,
    max_steps: int | None,
    carry_session_history: bool,
    total_selected: int,
    predictions: list[dict[str, Any]],
    n_jobs: int,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for item in predictions:
        status = item.get("status")
        if isinstance(status, str) and status:
            counts[status] += 1
    return {
        "prepared_json": str(Path(prepared_json).expanduser().resolve()),
        "output_json": str(output_json),
        "config": config,
        "documents_dir_override": documents_dir,
        "max_steps": max_steps,
        "n_jobs": n_jobs,
        "carry_session_history": carry_session_history,
        "total_selected": total_selected,
        "prediction_count": len(predictions),
        "status_counts": dict(sorted(counts.items())),
    }


def _build_compact_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "sample_id": item.get("sample_id"),
                "doc_id": item.get("doc_id"),
                "question": item.get("question"),
                "metadata": {
                    "document_path": item.get("document_path"),
                    "answer_format": item.get("answer_format"),
                },
                "status": item.get("status"),
                "reason": item.get("reason"),
                "error": item.get("error"),
                "final_answer": item.get("final_answer"),
            }
        )
    return compact


def _session_id(*, prefix: str, doc_id: str) -> str:
    safe_doc_id = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in doc_id
    ).strip("._")
    if not safe_doc_id:
        safe_doc_id = "document"
    return f"{prefix}_{safe_doc_id}"


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
