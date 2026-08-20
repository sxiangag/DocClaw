"""Run DocClaw OCR on OmniDocBench page images."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docclaw import DocClaw  # noqa: E402
from docclaw.exporter import export_page_markdown  # noqa: E402
from experiment.omnidocbench_end2end.prepare_dataset import (  # noqa: E402
    BenchmarkPageSample,
    build_manifest_entries,
    build_page_samples,
    default_summary,
    load_dataset_pages,
    load_prepared_bundle,
    save_json,
)


DEFAULT_PROMPT = "ocr this page"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DocClaw OCR on OmniDocBench page images.",
    )
    parser.add_argument("--dataset-json", help="Path to OmniDocBench.json.")
    parser.add_argument(
        "--prepared-json",
        help="Path to a prepared page-level bundle created by prepare_dataset.py.",
    )
    parser.add_argument("--run-root", required=True, help="Output root for this run.")
    parser.add_argument(
        "--dataset-root",
        help="Optional dataset root used to resolve relative image paths.",
    )
    parser.add_argument("--image-root", help="Optional image root override.")
    parser.add_argument("--offset", type=int, default=0, help="Start from page index N.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of pages.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Fixed task prompt for the run. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--config",
        help="Optional DocClaw config path. Defaults to normal DocClaw config discovery.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum DocClaw steps per page.",
    )
    parser.add_argument(
        "--pretty-markdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exporter pretty setting. Default: enabled.",
    )
    return parser


async def main_async() -> int:
    args = build_parser().parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if bool(args.dataset_json) == bool(args.prepared_json):
        raise ValueError("specify exactly one of --dataset-json or --prepared-json")

    run_root = Path(args.run_root).expanduser().resolve()
    prediction_dir = run_root / "predictions"
    manifest_path = run_root / "run_manifest.json"
    summary_path = run_root / "summary.json"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    dataset_json, samples, dataset_root, image_root = _resolve_samples(args)
    manifest_entries = build_manifest_entries(samples, prediction_dir=prediction_dir)
    summary = default_summary(
        dataset_json=dataset_json,
        prediction_dir=prediction_dir,
        sample_count=len(samples),
        prompt=args.prompt,
        pretty=args.pretty_markdown,
    )
    summary.update(
        {
            "method": "docclaw",
            "execution_phase": "running",
            "max_steps": args.max_steps,
            "protocol": {
                "prediction_dir_name": "predictions",
                "prediction_suffix": ".md",
                "failure_policy": "write empty markdown and continue",
                "one_page_per_run": True,
            },
        }
    )
    if dataset_root is not None:
        summary["dataset_root"] = str(Path(dataset_root).expanduser().resolve())
    if image_root is not None:
        summary["image_root"] = str(Path(image_root).expanduser().resolve())
    if args.prepared_json:
        summary["prepared_bundle"] = str(Path(args.prepared_json).expanduser().resolve())
    if args.config:
        summary["docclaw_config"] = str(Path(args.config).expanduser().resolve())

    save_json([entry.to_dict() for entry in manifest_entries], manifest_path)
    save_json(summary, summary_path)

    app = DocClaw.from_config(args.config)
    for index, sample in enumerate(samples):
        entry = manifest_entries[index]
        summary["pages_started"] += 1
        entry.status = "running"
        save_json([item.to_dict() for item in manifest_entries], manifest_path)
        save_json(summary, summary_path)

        output_path = Path(entry.prediction_path)
        try:
            markdown = await _run_page(
                app=app,
                sample=sample,
                prompt=args.prompt,
                max_steps=args.max_steps,
                pretty_markdown=args.pretty_markdown,
            )
            _write_markdown(output_path, markdown)
            entry.status = "completed"
            summary["pages_completed"] += 1
        except Exception as exc:
            _write_markdown(output_path, "")
            entry.status = "failed"
            entry.error = f"{type(exc).__name__}: {exc}"
            summary["pages_failed"] += 1

        save_json([item.to_dict() for item in manifest_entries], manifest_path)
        save_json(summary, summary_path)

    summary["execution_phase"] = "completed"
    save_json([entry.to_dict() for entry in manifest_entries], manifest_path)
    save_json(summary, summary_path)
    print(summary_path)
    return 0


def _resolve_samples(args: argparse.Namespace) -> tuple[Path, list[BenchmarkPageSample], str | None, str | None]:
    if args.prepared_json:
        prepared_bundle = load_prepared_bundle(args.prepared_json)
        return (
            Path(prepared_bundle.source_dataset_json).expanduser().resolve(),
            prepared_bundle.samples,
            prepared_bundle.metadata.get("dataset_root"),
            prepared_bundle.metadata.get("image_root"),
        )

    dataset_json = Path(args.dataset_json).expanduser().resolve()
    dataset_root = args.dataset_root if args.dataset_root is not None else str(dataset_json.parent)
    pages = load_dataset_pages(dataset_json)
    samples = build_page_samples(
        pages=pages,
        dataset_root=dataset_root,
        image_root=args.image_root,
        offset=args.offset,
        limit=args.limit,
    )
    return dataset_json, samples, dataset_root, args.image_root


async def _run_page(
    *,
    app: DocClaw,
    sample: BenchmarkPageSample,
    prompt: str,
    max_steps: int,
    pretty_markdown: bool,
) -> str:
    document = app.load_document(sample.image_path, document_id=sample.page_basename)
    result = await app.run(document, prompt, max_steps=max_steps)
    pages = result.state.document.pages
    if len(pages) != 1:
        raise ValueError(f"expected one page for {sample.sample_id}, got {len(pages)}")
    return export_page_markdown(pages[0], pretty=pretty_markdown)


def _write_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
