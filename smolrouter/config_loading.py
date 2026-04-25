"""Helpers for loading provider configuration values from files."""

from pathlib import Path
from typing import List


def _apply_assignment_parsing(line: str) -> str:
    if line.startswith("export "):
        line = line[len("export ") :].strip()

    if "=" in line:
        _, value = line.split("=", 1)
        return value.strip()

    return line


def _normalize_config_entry(
    raw_line: str,
    *,
    skip_comments: bool,
    allow_assignments: bool,
    strip_inline_comments: bool,
) -> str:
    line = raw_line.strip()
    if not line:
        return ""

    if skip_comments and line.startswith("#"):
        return ""

    if allow_assignments:
        line = _apply_assignment_parsing(line)

    if strip_inline_comments:
        line = line.split("#", 1)[0].strip()

    return line.strip().strip('"').strip("'")


def load_config_entries(
    file_path: str,
    *,
    skip_comments: bool = True,
    allow_assignments: bool = False,
    strip_inline_comments: bool = False,
) -> List[str]:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise ValueError(f"API key file not found: {file_path}")

    entries: List[str] = []
    for raw_line in path.read_text().splitlines():
        line = _normalize_config_entry(
            raw_line,
            skip_comments=skip_comments,
            allow_assignments=allow_assignments,
            strip_inline_comments=strip_inline_comments,
        )
        if line:
            entries.append(line)

    return entries


def load_first_config_entry(
    file_path: str,
    *,
    skip_comments: bool = True,
    allow_assignments: bool = False,
    strip_inline_comments: bool = False,
    value_label: str = "API key",
) -> str:
    entries = load_config_entries(
        file_path,
        skip_comments=skip_comments,
        allow_assignments=allow_assignments,
        strip_inline_comments=strip_inline_comments,
    )
    if entries:
        return entries[0]

    raise ValueError(f"No {value_label} found in file: {file_path}")