"""Prepare a stable question-level MMLongBench-Doc bundle for downstream evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREPARED_BUNDLE_FORMAT_VERSION = "docclaw_mmlongbench_doc_prepared_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a question-level MMLongBench-Doc bundle.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--benchmark-root",
        help="Path to a local MMLongBench-Doc checkout. Uses <root>/data/samples.json and <root>/data/documents.",
    )
    source_group.add_argument(
        "--samples-json",
        help="Path to samples.json if you do not want to point at a full benchmark checkout.",
    )
    parser.add_argument(
        "--documents-dir",
        help="Documents directory. Required with --samples-json. Ignored when --benchmark-root is provided.",
    )
    parser.add_argument("--output-json", required=True, help="Path to prepared bundle JSON.")
    parser.add_argument("--offset", type=int, default=0, help="Start from sample index N.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of samples.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    benchmark_root = (
        Path(args.benchmark_root).expanduser().resolve()
        if args.benchmark_root
        else None
    )
    if benchmark_root is not None:
        samples_json = benchmark_root / "data" / "samples.json"
        documents_dir = benchmark_root / "data" / "documents"
    else:
        if not args.documents_dir:
            raise ValueError("--documents-dir is required with --samples-json")
        samples_json = Path(args.samples_json).expanduser().resolve()
        documents_dir = Path(args.documents_dir).expanduser().resolve()

    raw_samples = _load_dataset_samples(samples_json)
    prepared_samples = _build_question_samples(
        samples=raw_samples,
        documents_dir=documents_dir,
        offset=args.offset,
        limit=args.limit,
    )

    missing_documents = [
        sample["document_path"]
        for sample in prepared_samples
        if not Path(sample["document_path"]).exists()
    ]
    if missing_documents:
        preview = "\n".join(missing_documents[:10])
        raise FileNotFoundError(
            "failed to resolve benchmark documents for prepared bundle.\n"
            f"First missing paths:\n{preview}"
        )

    bundle = {
        "format_version": PREPARED_BUNDLE_FORMAT_VERSION,
        "benchmark": "MMLongBench-Doc",
        "task": "question_answering",
        "source_samples_json": str(samples_json),
        "samples": prepared_samples,
        "metadata": {
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "documents_dir": str(documents_dir),
            "offset": args.offset,
            "limit": args.limit,
            "sample_count_total": len(raw_samples),
            "sample_count_selected": len(prepared_samples),
        },
    }
    _save_json(bundle, args.output_json)
    print(Path(args.output_json).expanduser().resolve())
    print(f"Prepared samples: {len(prepared_samples)}")
    return 0


def _load_dataset_samples(path: str | Path) -> list[dict]:
    dataset_path = Path(path).expanduser().resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MMLongBench-Doc samples JSON must be a top-level list")
    return [item for item in payload if isinstance(item, dict)]


def _build_question_samples(
    *,
    samples: list[dict],
    documents_dir: str | Path,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict]:
    documents_root = Path(documents_dir).expanduser().resolve()
    selected = samples[offset : (offset + limit) if limit is not None else None]
    prepared: list[dict] = []
    for bundle_sample_index, sample in enumerate(selected):
        dataset_sample_index = offset + bundle_sample_index
        doc_id = str(sample["doc_id"])
        prepared.append(
            {
                "sample_id": f"mmlongbench_doc_q_{dataset_sample_index}",
                "bundle_sample_index": bundle_sample_index,
                "dataset_sample_index": dataset_sample_index,
                "doc_id": doc_id,
                "document_path": str(documents_root / doc_id),
                "question": str(sample["question"]),
                "answer": str(sample["answer"]),
                "answer_format": str(sample["answer_format"]),
                "metadata": {
                    "doc_type": sample.get("doc_type"),
                    "evidence_pages": sample.get("evidence_pages"),
                    "evidence_sources": sample.get("evidence_sources"),
                    "source_sample": sample,
                },
            }
        )
    return prepared


def _save_json(data: object, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
