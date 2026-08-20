"""TOML config loading for DocClaw."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import tomllib

from docclaw.config.schema import DocClawConfig

_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def get_config_path(path: str | Path | None = None) -> Path:
    if path is None:
        return (Path.cwd() / "docclaw.toml").resolve()
    return Path(path).expanduser().resolve()


def load_config(path: str | Path | None = None) -> DocClawConfig:
    config_path = get_config_path(path)
    if not config_path.exists():
        return DocClawConfig()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    resolved = _resolve_env_vars(raw)
    return DocClawConfig.from_dict(resolved)


def save_config(config: DocClawConfig, path: str | Path | None = None) -> Path:
    config_path = get_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_dump_toml(config.to_dict()), encoding="utf-8")
    return config_path


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF_PATTERN.sub(_replace_env_var, value)
    if isinstance(value, dict):
        return {key: _resolve_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _replace_env_var(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name)
    if value is None:
        raise ValueError(
            f"Environment variable '{name}' referenced in config is not set"
        )
    return value


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_table(lines, [], data)
    return "\n".join(lines).rstrip() + "\n"


def _append_table(lines: list[str], prefix: list[str], data: dict[str, Any]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    child_tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            child_tables.append((key, value))
            continue
        scalar_items.append((key, value))

    if prefix:
        if lines:
            lines.append("")
        lines.append(f"[{'.'.join(prefix)}]")

    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")

    for key, value in child_tables:
        _append_table(lines, prefix + [key], value)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")
