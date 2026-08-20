"""Prepare a stable page-level OmniDocBench bundle for end-to-end runs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


PREPARED_BUNDLE_FORMAT_VERSION = "docclaw_omnidocbench_end2end_prepared_v1"


@dataclass(slots=True)
class BenchmarkPageSample:
    sample_id: str
    bundle_page_index: int
    dataset_page_index: int
    document_id: str
    image_path: str
    page_basename: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "bundle_page_index": self.bundle_page_index,
            "dataset_page_index": self.dataset_page_index,
            "document_id": self.document_id,
            "image_path": self.image_path,
            "page_basename": self.page_basename,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkPageSample":
        return cls(
            sample_id=str(data["sample_id"]),
            bundle_page_index=int(data["bundle_page_index"]),
            dataset_page_index=int(data["dataset_page_index"]),
            document_id=str(data["document_id"]),
            image_path=str(data["image_path"]),
            page_basename=str(data["page_basename"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class RunManifestEntry:
    sample_id: str
    bundle_page_index: int
    dataset_page_index: int
    document_id: str
    image_path: str
    page_basename: str
    prediction_path: str
    status: str = "pending"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "bundle_page_index": self.bundle_page_index,
            "dataset_page_index": self.dataset_page_index,
            "document_id": self.document_id,
            "image_path": self.image_path,
            "page_basename": self.page_basename,
            "prediction_path": self.prediction_path,
            "status": self.status,
            "error": self.error,
        }


@dataclass(slots=True)
class PreparedEnd2EndBundle:
    format_version: str
    benchmark: str
    task: str
    source_dataset_json: str
    samples: list[BenchmarkPageSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "benchmark": self.benchmark,
            "task": self.task,
            "source_dataset_json": self.source_dataset_json,
            "samples": [sample.to_dict() for sample in self.samples],
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedEnd2EndBundle":
        return cls(
            format_version=str(data["format_version"]),
            benchmark=str(data["benchmark"]),
            task=str(data["task"]),
            source_dataset_json=str(data["source_dataset_json"]),
            samples=[
                BenchmarkPageSample.from_dict(item)
                for item in data.get("samples", [])
                if isinstance(item, dict)
            ],
            metadata=dict(data.get("metadata") or {}),
        )


def load_dataset_pages(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("OmniDocBench dataset JSON must be a top-level list")
    return [item for item in payload if isinstance(item, dict)]


def build_page_samples(
    *,
    pages: list[dict[str, Any]],
    dataset_root: str | Path | None = None,
    image_root: str | Path | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[BenchmarkPageSample]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    dataset_root_path = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else None
    )
    image_root_path = (
        Path(image_root).expanduser().resolve()
        if image_root is not None
        else None
    )
    selected_pages = pages[offset : (offset + limit) if limit is not None else None]
    samples: list[BenchmarkPageSample] = []

    for bundle_page_index, page in enumerate(selected_pages):
        dataset_page_index = offset + bundle_page_index
        page_info = page.get("page_info")
        if not isinstance(page_info, dict):
            raise ValueError(f"page {dataset_page_index} missing page_info")

        image_path = _resolve_image_path(
            page_info,
            dataset_root=dataset_root_path,
            image_root=image_root_path,
            dataset_page_index=dataset_page_index,
        )
        page_basename = _page_basename_from_image_path(image_path)
        document_id = _build_document_id_from_page_info(page_info, fallback=page_basename)
        samples.append(
            BenchmarkPageSample(
                sample_id=f"omnidocbench_page_{dataset_page_index}",
                bundle_page_index=bundle_page_index,
                dataset_page_index=dataset_page_index,
                document_id=document_id,
                image_path=str(image_path),
                page_basename=page_basename,
                metadata={"page_info": copy.deepcopy(page_info)},
            )
        )

    return samples


def build_manifest_entries(
    samples: list[BenchmarkPageSample],
    *,
    prediction_dir: str | Path,
) -> list[RunManifestEntry]:
    prediction_root = Path(prediction_dir).expanduser().resolve()
    return [
        RunManifestEntry(
            sample_id=sample.sample_id,
            bundle_page_index=sample.bundle_page_index,
            dataset_page_index=sample.dataset_page_index,
            document_id=sample.document_id,
            image_path=sample.image_path,
            page_basename=sample.page_basename,
            prediction_path=str(prediction_root / f"{sample.page_basename}.md"),
        )
        for sample in samples
    ]


def default_summary(
    *,
    dataset_json: str | Path,
    prediction_dir: str | Path,
    sample_count: int,
    prompt: str,
    pretty: bool,
) -> dict[str, Any]:
    return {
        "benchmark": "OmniDocBench",
        "task": "end2end",
        "dataset_json": str(Path(dataset_json).expanduser().resolve()),
        "prediction_dir": str(Path(prediction_dir).expanduser().resolve()),
        "sample_count": sample_count,
        "pages_planned": sample_count,
        "pages_started": 0,
        "pages_completed": 0,
        "pages_failed": 0,
        "prompt": prompt,
        "markdown_pretty": pretty,
        "execution_phase": "not_started",
    }


def save_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def save_prepared_bundle(bundle: PreparedEnd2EndBundle, path: str | Path) -> Path:
    return save_json(bundle.to_dict(), path)


def load_prepared_bundle(path: str | Path) -> PreparedEnd2EndBundle:
    bundle_path = Path(path).expanduser().resolve()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prepared end-to-end bundle must be a JSON object")
    bundle = PreparedEnd2EndBundle.from_dict(payload)
    if bundle.format_version != PREPARED_BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported prepared bundle format_version: {bundle.format_version!r}"
        )
    return bundle


def _resolve_image_path(
    page_info: dict[str, Any],
    *,
    dataset_root: Path | None,
    image_root: Path | None,
    dataset_page_index: int,
) -> Path:
    raw_image_path = page_info.get("image_path")
    if not isinstance(raw_image_path, str) or not raw_image_path.strip():
        raise ValueError(f"page {dataset_page_index} missing page_info.image_path")

    raw = Path(raw_image_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if image_root is not None:
        candidates.append((image_root / raw.name).resolve())
        candidates.append((image_root / raw).resolve())
    if dataset_root is not None:
        candidates.append((dataset_root / raw).resolve())
        candidates.append((dataset_root / "images" / raw.name).resolve())
        candidates.append((dataset_root / "images" / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if image_root is not None:
        return (image_root / raw.name).resolve()
    if dataset_root is not None:
        return (dataset_root / raw).resolve()
    return raw.expanduser().resolve()


def _page_basename_from_image_path(image_path: str | Path) -> str:
    stem = Path(image_path).stem
    if not stem:
        raise ValueError(f"cannot derive page basename from image path: {image_path}")
    return stem


def _build_document_id_from_page_info(page_info: dict[str, Any], *, fallback: str) -> str:
    pdf_path = page_info.get("pdf_path")
    if isinstance(pdf_path, str) and pdf_path.strip():
        stem = Path(pdf_path).stem
        if stem:
            return stem
    image_path = page_info.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        stem = Path(image_path).stem
        if stem:
            return stem
    return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a page-level OmniDocBench bundle for end-to-end experiments.",
    )
    parser.add_argument("--dataset-json", required=True, help="Path to OmniDocBench.json.")
    parser.add_argument("--output-json", required=True, help="Path to prepared bundle JSON.")
    parser.add_argument(
        "--dataset-root",
        help="Optional dataset root used to resolve relative image paths. Defaults to the parent of --dataset-json.",
    )
    parser.add_argument(
        "--image-root",
        help="Optional image root override.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start from page index N.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of pages.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    dataset_json = Path(args.dataset_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else dataset_json.parent
    )

    pages = load_dataset_pages(dataset_json)
    samples = build_page_samples(
        pages=pages,
        dataset_root=dataset_root,
        image_root=args.image_root,
        offset=args.offset,
        limit=args.limit,
    )
    missing_images = [
        sample.image_path for sample in samples if not Path(sample.image_path).exists()
    ]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            "failed to resolve benchmark images for prepared bundle.\n"
            f"First missing paths:\n{preview}"
        )

    bundle = PreparedEnd2EndBundle(
        format_version=PREPARED_BUNDLE_FORMAT_VERSION,
        benchmark="OmniDocBench",
        task="end2end",
        source_dataset_json=str(dataset_json),
        samples=samples,
        metadata={
            "dataset_root": str(dataset_root),
            "image_root": (
                str(Path(args.image_root).expanduser().resolve())
                if args.image_root
                else None
            ),
            "offset": args.offset,
            "limit": args.limit,
            "page_count_total": len(pages),
            "page_count_selected": len(samples),
        },
    )
    save_prepared_bundle(bundle, output_json)
    print(output_json)
    print(f"Prepared pages: {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
