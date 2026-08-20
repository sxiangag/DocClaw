"""Loader for DocClaw task skills."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


BUILTIN_TASK_SKILLS_DIR = Path(__file__).resolve().parent
_FRONTMATTER_RE = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)


@dataclass(slots=True, frozen=True)
class TaskSkillInfo:
    """Resolved metadata for one task skill."""

    name: str
    path: str
    source: str
    description: str | None = None


class TaskSkillsLoader:
    """Discover and read task-skill definitions.

    Task skills are stored as ``<skill-name>/SKILL.md`` directories. Workspace
    skills override built-in skills with the same name.
    """

    def __init__(
        self,
        *,
        workspace_skills_dir: str | Path | None = None,
        builtin_skills_dir: str | Path | None = None,
    ) -> None:
        self.workspace_skills_dir = (
            Path(workspace_skills_dir).expanduser().resolve()
            if workspace_skills_dir is not None
            else None
        )
        self.builtin_skills_dir = (
            Path(builtin_skills_dir).expanduser().resolve()
            if builtin_skills_dir is not None
            else BUILTIN_TASK_SKILLS_DIR
        )

    def list_skills(self) -> list[TaskSkillInfo]:
        entries = self._entries_from_dir(self.workspace_skills_dir, "workspace")
        workspace_names = {entry.name for entry in entries}
        entries.extend(
            self._entries_from_dir(
                self.builtin_skills_dir,
                "builtin",
                skip_names=workspace_names,
            )
        )
        return sorted(entries, key=lambda entry: entry.name)

    def load_skill(self, name: str) -> str | None:
        for base in self._roots():
            path = base / name / "SKILL.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_skill_body(self, name: str) -> str | None:
        content = self.load_skill(name)
        if content is None:
            return None
        return self._strip_frontmatter(content).strip()

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        content = self.load_skill(name)
        if not content:
            return None
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None
        return _parse_frontmatter(match.group(1))

    def build_skills_summary(self) -> str:
        lines: list[str] = []
        for entry in self.list_skills():
            description = entry.description or entry.name
            lines.append(f"- **{entry.name}** — {description}  `{entry.path}`")
        return "\n".join(lines)

    def _entries_from_dir(
        self,
        base: Path | None,
        source: str,
        *,
        skip_names: set[str] | None = None,
    ) -> list[TaskSkillInfo]:
        if base is None or not base.exists():
            return []

        entries: list[TaskSkillInfo] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            metadata = self.get_skill_metadata_from_path(skill_file)
            description = metadata.get("description") if metadata else None
            entries.append(
                TaskSkillInfo(
                    name=name,
                    path=str(skill_file),
                    source=source,
                    description=description if isinstance(description, str) else None,
                )
            )
        return entries

    def get_skill_metadata_from_path(self, path: str | Path) -> dict[str, object] | None:
        skill_path = Path(path).expanduser().resolve()
        if not skill_path.exists():
            return None
        content = skill_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None
        return _parse_frontmatter(match.group(1))

    def _roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.workspace_skills_dir is not None:
            roots.append(self.workspace_skills_dir)
        roots.append(self.builtin_skills_dir)
        return roots

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return content
        return content[match.end():]


def _parse_frontmatter(raw: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    pending_key: str | None = None
    pending_lines: list[str] = []

    def flush_pending() -> None:
        nonlocal pending_key, pending_lines
        if pending_key is None:
            return
        metadata[pending_key] = _parse_frontmatter_value("\n".join(pending_lines).strip())
        pending_key = None
        pending_lines = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            if pending_key is not None:
                pending_lines.append(stripped)
            continue
        flush_pending()
        key, value = stripped.split(":", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            continue
        pending_key = normalized_key
        pending_lines = [normalized_value]
    flush_pending()
    return metadata


def _parse_frontmatter_value(raw: str) -> object:
    if not raw:
        return ""
    if raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        return raw[1:-1]
    if raw in {"true", "false"}:
        return raw == "true"
    if raw[0] in "[{" and raw[-1] in "]}":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw
