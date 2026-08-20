"""Command-line entrypoint for DocClaw."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from docclaw.docclaw import DocClaw
from docclaw.agent.utils import Action, Observation, RunResult


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docclaw",
        description="Run DocClaw on a document and task prompt.",
    )
    parser.add_argument(
        "--config",
        help="Path to docclaw.toml. Defaults to ./docclaw.toml.",
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Path to the input document file.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task prompt to execute against the document.",
    )
    parser.add_argument(
        "--document-id",
        help="Optional explicit document_id override.",
    )
    parser.add_argument(
        "--session-id",
        help="Optional session identifier for persisted history.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional max planner steps override.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory for planner/select_ocr/inspect_ocr JSONL debug dumps.",
    )
    return parser


def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docclaw chat",
        description="Start an interactive DocClaw chat session for one document.",
    )
    parser.add_argument(
        "--config",
        help="Path to docclaw.toml. Defaults to ./docclaw.toml.",
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Path to the input document file.",
    )
    parser.add_argument(
        "--document-id",
        help="Optional explicit document_id override.",
    )
    parser.add_argument(
        "--session-id",
        help="Optional session identifier for persisted history.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional max planner steps override.",
    )
    parser.add_argument(
        "--save-trace-dir",
        help="Optional directory to persist one full trace JSON per chat turn.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory for planner/select_ocr/inspect_ocr JSONL debug dumps.",
    )
    return parser


async def run_cli(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    app_factory: Any = None,
) -> int:
    if argv is None:
        argv = list(sys.argv[1:])
    else:
        argv = list(argv)
    if argv and argv[0] == "chat":
        return await run_chat_cli(
            argv[1:],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            app_factory=app_factory,
        )

    # one-shot runner, for testing purpose
    parser = build_run_parser()
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    factory = app_factory or DocClaw.from_config

    try:
        _configure_debug_dir(args.debug_dir)
        app = factory(args.config)
        document = app.load_document(args.document, document_id=args.document_id)
        _write_line(
            err,
            f"Loaded document {document.document_id} with {len(document.pages)} page(s).",
        )
        _write_line(err, f"Task: {args.task}")
        result = await app.run(
            document,
            args.task,
            max_steps=args.max_steps,
            session_id=args.session_id,
            on_event=lambda event, payload: _emit_cli_event(err, event, payload),
        )
    except Exception as exc:
        err.write(f"Error: {exc}\n")
        return 1

    if result.answer:
        out.write(result.answer + "\n")
    elif result.status != "completed":
        out.write(f"{result.status}\n")

    reason = getattr(result, "reason", None)
    if not result.answer and isinstance(reason, str) and reason:
        _write_line(err, f"No answer, reason: {reason}")

    if result.status == "stopped":
        return 0

    if result.status == "failed":
        failed_action = result.state.metadata.get("failed_action")
        if isinstance(failed_action, dict):
            action_type = failed_action.get("action_type")
            target = failed_action.get("target")
            if isinstance(action_type, str):
                _write_line(
                    err,
                    f"Failed action: {action_type}{_format_target_suffix(target)}",
                )
        if result.error:
            _write_line(err, result.error)
        return 1
    return 0


async def run_chat_cli(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    app_factory: Any = None,
) -> int:
    parser = build_chat_parser()
    args = parser.parse_args(argv)
    in_stream = stdin or sys.stdin
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    factory = app_factory or DocClaw.from_config

    try:
        _configure_debug_dir(args.debug_dir)
        app = factory(args.config)
        document = app.load_document(args.document, document_id=args.document_id)
    except Exception as exc:
        err.write(f"Error: {exc}\n")
        return 1
    session = _get_or_create_chat_session(app, args.session_id, document.document_id)
    last_result: RunResult | None = None
    turn_index = 0

    _write_launch(
        err,
        document_id=document.document_id,
        page_count=len(document.pages),
        session_id=session.session_id,
    )
    _write_chat_separator(err)

    while True:
        line = _read_prompt(in_stream, out, "docclaw> ")
        if line is None:
            _write_line(err, "Exiting chat.")
            return 0

        task = line.strip()
        if not task:
            continue
        if task == "/exit":
            _write_line(err, "Exiting chat.")
            return 0
        if task == "/reset":
            session.reset_active_history()
            app.runner.session_manager.save(session)
            _write_command(err, "Conversation context reset.")
            _write_chat_separator(err)
            continue
        if task == "/reload":
            try:
                document = app.load_document(args.document, document_id=args.document_id)
            except Exception as exc:
                _write_line(err, f"Error: {exc}")
                _write_chat_separator(err)
                continue
            session = _get_or_create_chat_session(
                app,
                args.session_id,
                document.document_id,
                force_new=True,
            )
            last_result = None
            _write_command(
                err,
                f"Reloaded document {document.document_id} with {len(document.pages)} page(s).",
            )
            _write_chat_separator(err)
            continue
        if task == "/status":
            _write_chat_status(err, document=document, session=session)
            _write_chat_separator(err)
            continue
        if task == "/trace":
            _write_trace(err, last_result)
            _write_chat_separator(err)
            continue
        if task.startswith("/"):
            _write_command(err, f"Unknown command: {task}")
            _write_chat_separator(err)
            continue

        turn_index += 1
        _write_prompt(err, turn_index, task)
        _write_block_title(err, "trace", style="bold blue")
        current_trace: list[dict[str, Any]] = []

        def on_event(event: str, payload: dict[str, Any]) -> None:
            _emit_cli_event(err, event, payload)
            if event == "step_finished":
                current_trace.append(payload)

        try:
            result = await app.run(
                document,
                task,
                max_steps=args.max_steps,
                session=session,
                on_event=on_event,
            )
        except Exception as exc:
            _write_line(err, f"Error: {exc}")
            _write_chat_separator(err)
            continue
        result.state.metadata["_cli_last_trace"] = current_trace
        last_result = result
        if args.save_trace_dir:
            trace_path = _save_chat_trace(
                args.save_trace_dir,
                result=result,
                turn_index=turn_index,
                task=task,
                session_id=session.session_id,
                document_id=document.document_id,
            )
            _write_key_value(err, "trace_saved", str(trace_path))

        if result.answer:
            _write_block_title(err, "answer", style="bold green")
            _write_line(out, result.answer)
        elif result.status != "completed":
            _write_line(out, result.status)

        reason = getattr(result, "reason", None)
        if not result.answer and isinstance(reason, str) and reason:
            _write_key_value(err, "reason", reason)

        if result.status == "stopped":
            _write_result_label(err, "status: stopped", "bold yellow")
            _write_chat_separator(err)
            continue

        if result.status == "failed":
            _write_result_label(err, "status: failed", "bold red")
            failed_action = result.state.metadata.get("failed_action")
            if isinstance(failed_action, dict):
                action_type = failed_action.get("action_type")
                target = failed_action.get("target")
                if isinstance(action_type, str):
                    _write_key_value(
                        err,
                        "failed_action",
                        f"{action_type}{_format_target_suffix(target)}",
                    )
            if result.error:
                _write_key_value(err, "error", result.error)
        _write_chat_separator(err)


def _emit_cli_event(stream: TextIO, event: str, payload: dict[str, Any]) -> None:
    if event == "skill_selected":
        skill = payload.get("skill")
        if not isinstance(skill, dict):
            return
        name = skill.get("name")
        reason = skill.get("reason")
        if not isinstance(name, str) or not name:
            return
        line = Text()
        line.append("00. ", style="bold")
        line.append("planner", style="bold cyan")
        line.append(" -> ")
        line.append("select skill", style="bold yellow")
        line.append(": ")
        line.append(name, style="bold magenta")
        _console(stream).print(line)
        if isinstance(reason, str) and reason.strip():
            reason_line = Text("    ")
            reason_line.append("reason", style="bold yellow")
            reason_line.append(": ")
            reason_line.append(reason)
            _console(stream).print(reason_line)
        return

    if event == "step_started":
        action = payload.get("action")
        step_index = payload.get("step_index")
        if isinstance(action, Action) and isinstance(step_index, int):
            line = Text()
            line.append(f"{step_index + 1:02d}. ", style="bold")
            line.append("planner", style="bold cyan")
            line.append(" -> ")
            line.append(action.action_type, style="bold")
            _console(stream).print(line)
            if action.rationale:
                _write_line(stream, f"    {action.rationale}")
        return

    if event == "step_finished":
        action = payload.get("action")
        observation = payload.get("observation")
        step_index = payload.get("step_index")
        if not isinstance(action, Action) or not isinstance(observation, Observation):
            return
        if not isinstance(step_index, int):
            return
        line = Text("    ")
        line.append(action.action_type, style="bold")
        line.append(": ")
        if observation.success:
            detail = observation.message or "ok"
            line.append("OK", style="bold green")
        else:
            detail = observation.error or "action failed"
            line.append("ERR", style="bold red")
        line.append(" | ")
        line.append(detail)
        _console(stream).print(line)


def _format_target_suffix(target: Any) -> str:
    if not isinstance(target, dict) or not target:
        return ""
    parts: list[str] = []
    page_index = target.get("page_index")
    region_id = target.get("region_id")
    if page_index is not None:
        parts.append(f"page={page_index}")
    if isinstance(region_id, str) and region_id.strip():
        parts.append(f"region={region_id}")
    if not parts:
        parts.append(json.dumps(target, ensure_ascii=False, sort_keys=True))
    return " [" + ", ".join(parts) + "]"


def _configure_debug_dir(debug_dir: str | None) -> None:
    if not isinstance(debug_dir, str) or not debug_dir.strip():
        return
    root = Path(debug_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["DOCCLAW_PLANNER_DEBUG_PATH"] = str(root / "planner.jsonl")
    os.environ["DOCCLAW_OCR_QUALITY_DEBUG_PATH"] = str(root / "inspect_ocr.jsonl")
    os.environ["DOCCLAW_OCR_SELECTION_DEBUG_PATH"] = str(root / "select_ocr.jsonl")


def _write_line(stream: TextIO, text: str) -> None:
    stream.write(text + "\n")
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def _read_prompt(stream: TextIO, out: TextIO, prompt: str) -> str | None:
    out.write(prompt)
    flush = getattr(out, "flush", None)
    if callable(flush):
        flush()
    line = stream.readline()
    if line == "":
        return None
    return line.rstrip("\n")


def _get_or_create_chat_session(
    app: Any,
    session_id: str | None,
    document_id: str,
    *,
    force_new: bool = False,
):
    manager = app.runner.session_manager
    if session_id is not None and not force_new:
        existing = manager.get(session_id)
        if existing is not None:
            existing.metadata.setdefault("document_id", document_id)
            return existing
    return manager.create(
        session_id,
        metadata={"document_id": document_id},
    )


def _write_chat_status(
    stream: TextIO,
    *,
    document: Any,
    session: Any,
) -> None:
    region_count = 0
    table_cache_count = 0
    for page in getattr(document, "pages", ()):
        regions = getattr(page, "regions", ())
        region_count += len(regions)
        for region in regions:
            if getattr(region, "type", None) == "table" and getattr(region, "text", None):
                table_cache_count += 1
    ocr_text_count = 0
    for page in getattr(document, "pages", ()):
        if getattr(page, "ocr_text", None):
            ocr_text_count += 1
        for region in getattr(page, "regions", ()):
            if getattr(region, "text", None):
                ocr_text_count += 1
    document_metadata = getattr(document, "metadata", {}) or {}
    if not isinstance(document_metadata, dict):
        document_metadata = {}
    figure_cache = document_metadata.get("figure_cache", {})
    figure_cache_count = len(figure_cache) if isinstance(figure_cache, dict) else 0
    active_count = len(session.get_active_history()) if hasattr(session, "get_active_history") else 0

    console = _console(stream)
    console.print(Text("[status]", style="bold blue"))
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("document_id", str(document.document_id))
    table.add_row("pages", str(len(document.pages)))
    table.add_row("regions", str(region_count))
    table.add_row("ocr_text_entries", str(ocr_text_count))
    table.add_row("table_cache_entries", str(table_cache_count))
    table.add_row("figure_cache_entries", str(figure_cache_count))
    table.add_row("session_id", str(session.session_id))
    table.add_row("active_history_messages", str(active_count))
    console.print(table)


def _write_trace(stream: TextIO, result: RunResult | None) -> None:
    trace = getattr(result, "trace", None) if result is not None else None
    if not trace and result is not None:
        metadata = getattr(getattr(result, "state", None), "metadata", {}) or {}
        if isinstance(metadata, dict):
            trace = metadata.get("_cli_last_trace")
    _write_block_title(stream, "trace", style="bold blue")
    if not trace:
        _write_line(stream, "No trace available.")
        return
    for step in trace:
        if isinstance(step, dict):
            step_index = int(step.get("step_index", 0))
            action = step.get("action")
            observation = step.get("observation")
        else:
            step_index = step.step_index
            action = step.action
            observation = step.observation
        line = Text()
        line.append(f"{step_index + 1:02d}. ", style="bold")
        line.append(f"{action.action_type}{_format_target_suffix(action.target)}", style="bold")
        _console(stream).print(line)
        if action.rationale:
            _write_line(stream, f"    {action.rationale}")
        detail = observation.message if observation.success else observation.error
        if detail:
            status_line = Text("    ")
            if observation.success:
                status_line.append("OK", style="bold green")
            else:
                status_line.append("ERR", style="bold red")
            status_line.append(f": {detail}")
            _console(stream).print(status_line)


def _save_chat_trace(
    save_dir: str | Path,
    *,
    result: RunResult,
    turn_index: int,
    task: str,
    session_id: str,
    document_id: str,
) -> Path:
    output_dir = Path(save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = _trace_rows(result)
    payload = {
        "turn_index": turn_index,
        "task": task,
        "session_id": session_id,
        "document_id": document_id,
        "status": result.status,
        "answer": result.answer,
        "reason": result.reason,
        "error": result.error,
        "trace": trace_rows,
    }
    path = output_dir / f"{session_id}.turn_{turn_index:04d}.trace.json"
    path.write_text(
        json.dumps(_jsonable_trace_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _trace_rows(result: RunResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    trace = getattr(result, "trace", None)
    if trace:
        rows: list[dict[str, Any]] = []
        for step in trace:
            if hasattr(step, "to_dict"):
                rows.append(_jsonable_trace_row(step.to_dict()))
            elif isinstance(step, dict):
                rows.append(_jsonable_trace_row(step))
        if rows:
            return rows
    metadata = getattr(getattr(result, "state", None), "metadata", {}) or {}
    if isinstance(metadata, dict):
        fallback = metadata.get("_cli_last_trace")
        if isinstance(fallback, list):
            return [_jsonable_trace_row(item) for item in fallback if isinstance(item, dict)]
    return []


def _jsonable_trace_row(value: Any) -> dict[str, Any]:
    return dict(_jsonable_trace_value(value))


def _jsonable_trace_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return _jsonable_trace_value(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable_trace_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable_trace_value(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(key): _jsonable_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_jsonable_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_trace_value(item) for item in value]
    if isinstance(value, set):
        return [_jsonable_trace_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_chat_separator(stream: TextIO) -> None:
    _console(stream).print(Rule(style="dim"))


def _write_launch(stream: TextIO, *, document_id: str, page_count: int, session_id: str) -> None:
    console = _console(stream)
    console.print(Text("[launch]", style="bold yellow"))
    console.print(f"document={document_id} pages={page_count} session={session_id}")
    console.print("interactive chat started, type /exit to quit.")


def _write_prompt(stream: TextIO, turn_index: int, task: str) -> None:
    line = Text()
    line.append("[prompt]", style="bold magenta")
    line.append(" ")
    line.append(f"(turn {turn_index})", style="bold")
    _console(stream).print(line)
    _console(stream).print(task)


def _write_command(stream: TextIO, message: str) -> None:
    line = Text()
    line.append("[command]", style="bold magenta")
    line.append(" ")
    line.append(message)
    _console(stream).print(line)


def _write_result_label(stream: TextIO, label: str, style: str) -> None:
    _console(stream).print(Text(label, style=style))


def _write_key_value(stream: TextIO, key: str, value: str) -> None:
    line = Text()
    line.append(key, style="bold")
    line.append(": ")
    line.append(value)
    _console(stream).print(line)


def _write_block_title(stream: TextIO, title: str, *, style: str = "bold blue") -> None:
    _console(stream).print(Text(f"[{title}]", style=style))


def _console(stream: TextIO) -> Console:
    ansi = _supports_ansi(stream)
    return Console(
        file=stream,
        force_terminal=ansi,
        color_system="auto" if ansi else None,
        highlight=False,
        soft_wrap=True,
    )


def _supports_ansi(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run_cli(argv))


if __name__ == "__main__":
    raise SystemExit(main())
