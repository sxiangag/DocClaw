"""Utilities for optional JSONL debug dumps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def dump_jsonl_from_env(env_var: str, payload: dict[str, Any]) -> None:
    path_value = os.environ.get(env_var)
    if not isinstance(path_value, str) or not path_value.strip():
        return
    try:
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False))
            handle.write("\n")
    except Exception:
        return


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
