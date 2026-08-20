"""Path resolution helpers for DocClaw runtime storage."""

from __future__ import annotations

from pathlib import Path

from docclaw.config.loader import get_config_path
from docclaw.config.schema import DocClawConfig


def resolve_storage_root(
    config: DocClawConfig,
    *,
    config_path: str | Path | None = None,
) -> Path:
    path = _resolve_path(config.storage.root_dir, config_path=config_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_sessions_dir(
    config: DocClawConfig,
    *,
    config_path: str | Path | None = None,
) -> Path:
    raw = config.storage.sessions_dir
    path = _resolve_path(raw, config_path=config_path) if raw else resolve_storage_root(
        config,
        config_path=config_path,
    ) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_artifacts_dir(
    config: DocClawConfig,
    *,
    config_path: str | Path | None = None,
) -> Path:
    raw = config.storage.artifacts_dir
    path = _resolve_path(raw, config_path=config_path) if raw else resolve_storage_root(
        config,
        config_path=config_path,
    ) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_document_artifact_dir(
    config: DocClawConfig,
    document_id: str,
    *,
    config_path: str | Path | None = None,
) -> Path:
    path = resolve_artifacts_dir(config, config_path=config_path) / _safe_stem(document_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_skills_dir(
    config: DocClawConfig,
    *,
    config_path: str | Path | None = None,
) -> Path | None:
    raw = config.skills.workspace_dir
    if raw is None:
        return None
    return _resolve_path(raw, config_path=config_path)


def _resolve_path(raw: str | Path | None, *, config_path: str | Path | None) -> Path:
    if raw is None:
        raise ValueError("path value must not be None")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = get_config_path(config_path).parent
    return (base / path).resolve()


def _safe_stem(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    return clean or "document"
