"""Prepare stable OCRBench KIE bundles for downstream experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREPARED_BUNDLE_FORMAT_VERSION = "docclaw_ocrbench_kie_prepared_v1"
KIE_TYPE = "Key Information Extraction"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare OCRBench KIE bundles for DocClaw experiments.",
    )
    parser.add_argument(
        "--dataset-json",
        default="dataset/OCRBench/OCRBench.json",
        help="Path to the official OCRBench.json.",
    )
    parser.add_argument(
        "--image-root",
        default="dataset/OCRBench/data",
        help="Root directory used to resolve relative OCRBench image paths.",
    )
    parser.add_argument(
        "--output-json",
        default="dataset/OCRBench/OCRBench_kie.json",
        help="Path to prepared KIE bundle JSON.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start from sample index N.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of samples.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    dataset_json = Path(args.dataset_json).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    raw_items = _load_dataset_items(dataset_json)
    kie_items = [item for item in raw_items if item.get("type") == KIE_TYPE]
    prepared_samples = _build_samples(
        items=kie_items,
        image_root=image_root,
        offset=args.offset,
        limit=args.limit,
    )

    missing_images = [
        sample["image_path_absolute"]
        for sample in prepared_samples
        if not Path(sample["image_path_absolute"]).exists()
    ]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            "failed to resolve OCRBench KIE images for prepared bundle.\n"
            f"First missing paths:\n{preview}"
        )

    bundle = {
        "format_version": PREPARED_BUNDLE_FORMAT_VERSION,
        "benchmark": "OCRBench",
        "task": "key_information_extraction",
        "source_dataset_json": str(dataset_json),
        "samples": prepared_samples,
        "metadata": {
            "image_root": str(image_root),
            "offset": args.offset,
            "limit": args.limit,
            "sample_count_total": len(kie_items),
            "sample_count_selected": len(prepared_samples),
            "task_types": sorted({sample["type"] for sample in prepared_samples}),
            "dataset_names": sorted(
                {
                    str(sample["dataset_name"])
                    for sample in prepared_samples
                    if sample.get("dataset_name") is not None
                }
            ),
        },
    }
    _save_json(bundle, output_json)
    print(output_json)
    print(f"Prepared samples: {len(prepared_samples)}")
    return 0


def _load_dataset_items(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("OCRBench.json must be a top-level list")
    return [item for item in payload if isinstance(item, dict)]


def _build_samples(
    *,
    items: list[dict[str, Any]],
    image_root: Path,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = items[offset : (offset + limit) if limit is not None else None]
    prepared: list[dict[str, Any]] = []
    for item in selected:
        sample = dict(item)
        image_path = str(sample.get("image_path") or "")
        sample["image_path_absolute"] = str((image_root / image_path).resolve())
        prepared.append(sample)
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
