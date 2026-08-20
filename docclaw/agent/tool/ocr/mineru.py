"""MinerU2.5Pro-backed OCR candidate recognition."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


class MinerU2_5ProRecognizer:
    """Run MinerU2.5Pro on image crops and return plain-text results."""

    DEFAULT_BACKEND = "vlm-auto-engine"
    DEFAULT_MODEL_NAME = "MinerU2.5-Pro-2604-1.2B"
    _REPO_ROOT_ENVS = ("DOCCLAW_MINERU_REPO_ROOT", "MINERU_REPO_ROOT")
    _PYTHONPATH_ENV = "PYTHONPATH"
    _MODEL_NAME_ENV = "MINERU_VL_MODEL_NAME"

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        backend: str = DEFAULT_BACKEND,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        resolved_repo_root = repo_root or _repo_root_from_env()
        self.repo_root = (
            Path(resolved_repo_root).expanduser().resolve()
            if resolved_repo_root is not None
            else None
        )
        self.backend = backend
        self.model_name = model_name

    def recognize_batch(self, image_paths: Sequence[Path]) -> list[dict[str, Any]]:
        if not image_paths:
            return []
        if self.repo_root is None:
            env_names = ", ".join(self._REPO_ROOT_ENVS)
            raise RuntimeError(
                f"MinerU repo root is not configured; set one of: {env_names}"
            )
        if not self.repo_root.exists():
            raise FileNotFoundError(f"MinerU repo not found: {self.repo_root}")

        do_parse = _import_mineru_do_parse(self.repo_root)
        path_list = [Path(path).expanduser().resolve() for path in image_paths]

        with tempfile.TemporaryDirectory(prefix="docclaw_mineru_alt_") as tempdir:
            output_dir = Path(tempdir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            file_names = [f"alt_candidate_{index:04d}" for index, _ in enumerate(path_list)]
            file_bytes_list = [path.read_bytes() for path in path_list]
            lang_list = ["ch" for _ in path_list]

            previous_model_name = os.environ.get(self._MODEL_NAME_ENV)
            previous_pythonpath = os.environ.get(self._PYTHONPATH_ENV)
            pythonpath_entries = [str(self.repo_root)]
            if previous_pythonpath:
                pythonpath_entries.append(previous_pythonpath)
            os.environ[self._PYTHONPATH_ENV] = os.pathsep.join(pythonpath_entries)
            os.environ[self._MODEL_NAME_ENV] = self.model_name
            try:
                do_parse(
                    output_dir=str(output_dir),
                    pdf_file_names=file_names,
                    pdf_bytes_list=file_bytes_list,
                    p_lang_list=lang_list,
                    backend=self.backend,
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_md=False,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_orig_pdf=False,
                    f_dump_content_list=True,
                    image_analysis=False,
                    client_side_output_generation=False,
                )
            finally:
                if previous_model_name is None:
                    os.environ.pop(self._MODEL_NAME_ENV, None)
                else:
                    os.environ[self._MODEL_NAME_ENV] = previous_model_name
                if previous_pythonpath is None:
                    os.environ.pop(self._PYTHONPATH_ENV, None)
                else:
                    os.environ[self._PYTHONPATH_ENV] = previous_pythonpath

            results: list[dict[str, Any]] = []
            for file_name in file_names:
                content_list_path = next(
                    output_dir.glob(f"{file_name}/**/{file_name}_content_list.json"),
                    None,
                )
                content_list = []
                if content_list_path is not None and content_list_path.exists():
                    content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
                results.append(
                    {
                        "text": _content_list_to_text(content_list),
                        "confidence": None,
                    }
                )
            return results


def _import_mineru_do_parse(repo_root: Path):
    repo_root_text = str(repo_root)
    inserted = False
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
        inserted = True
    try:
        from mineru.cli.common import do_parse
    except Exception:
        if inserted:
            try:
                sys.path.remove(repo_root_text)
            except ValueError:
                pass
        raise
    return do_parse


def _repo_root_from_env() -> str | None:
    for env_name in MinerU2_5ProRecognizer._REPO_ROOT_ENVS:
        value = os.environ.get(env_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _content_list_to_text(content_list: Any) -> str:
    if not isinstance(content_list, list):
        return ""
    parts: list[str] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue
        if item.get("type") == "table":
            table_body = item.get("table_body")
            if isinstance(table_body, str) and table_body.strip():
                parts.append(table_body.strip())
    deduped: list[str] = []
    for part in parts:
        if deduped and deduped[-1] == part:
            continue
        deduped.append(part)
    return "\n".join(deduped)
