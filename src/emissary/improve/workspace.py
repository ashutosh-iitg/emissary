"""The only tools the improver gets: read, list, search and write, inside one worktree.

Least privilege is the point (Prime Agent §3.5: an unconstrained refiner saved
an exploit of its own metric as a skill). So there is no shell — a shell could
run the evaluator or write anywhere — and protected paths are invisible, not
just read-only: they hold the graders and the holdout scenarios, and an
improver that can read the holdout has turned it into a search set.
"""

import re
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from ..harness.tools import Tool, ToolResult

MAX_LINES = 400
MAX_MATCHES = 100
MAX_LISTED = 2000


class Workspace:
    def __init__(self, root: Path, *, editable: Sequence[str], protected: Sequence[str]):
        if not editable:
            raise ValueError("at least one editable glob is required")
        self.root = root.resolve()
        self.editable = tuple(editable)
        self.protected = tuple(protected)

    def is_protected(self, relative: str) -> bool:
        return _matches(relative, self.protected)

    def is_editable(self, relative: str) -> bool:
        return _matches(relative, self.editable) and not self.is_protected(relative)

    def resolve(self, path: str) -> tuple[Path, str]:
        """Return the absolute path and its root-relative POSIX form, or raise."""
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"{path} is outside the workspace")
        relative = candidate.relative_to(self.root).as_posix()
        if relative == ".git" or relative.startswith(".git/") or self.is_protected(relative):
            raise PermissionError(f"{path} is not available")
        return candidate, relative

    def files(self) -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            inside = path.relative_to(self.root)
            visible = ".git" not in inside.parts and not self.is_protected(inside.as_posix())
            if visible and path.is_file():
                yield inside.as_posix()

    def tools(self) -> tuple[Tool, ...]:
        return (
            Tool(
                "list_files",
                "List the files you may read. Files you may also change are marked (editable).",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self._list_files,
                api_scope="none",
            ),
            Tool(
                "read_file",
                f"Read up to {MAX_LINES} lines of a file, starting at `start_line` (1-based).",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "start_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                self._read_file,
                api_scope="none",
            ),
            Tool(
                "search",
                f"Search every readable file for a regular expression; up to {MAX_MATCHES} "
                "matches as path:line: text.",
                {
                    "type": "object",
                    "properties": {"pattern": {"type": "string", "minLength": 1}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                self._search,
                api_scope="none",
            ),
            Tool(
                "write_file",
                "Replace a file's entire content, creating it if needed. Only editable paths.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                self._write_file,
                side_effect="local",
            ),
        )

    def _list_files(self) -> ToolResult:
        lines = [f"{f} (editable)" if self.is_editable(f) else f for f in self.files()]
        if len(lines) > MAX_LISTED:
            shown = "\n".join(lines[:MAX_LISTED])
            return ToolResult("warning", f"first {MAX_LISTED} of {len(lines)} files", shown)
        return ToolResult("success", f"{len(lines)} files", "\n".join(lines))

    def _read_file(self, path: str, start_line: int = 1) -> ToolResult:
        try:
            absolute, relative = self.resolve(path)
            lines = absolute.read_text().splitlines()
        except (PermissionError, OSError, UnicodeDecodeError) as exc:
            return ToolResult("error", str(exc) or f"cannot read {path}")
        window = lines[start_line - 1 : start_line - 1 + MAX_LINES]
        numbered = "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(window))
        return ToolResult("success", f"{relative}: {len(lines)} lines", numbered)

    def _search(self, pattern: str) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult("error", f"invalid pattern: {exc}")
        matches = []
        for relative in self.files():
            try:
                text = (self.root / relative).read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{relative}:{number}: {line.strip()}")
                    if len(matches) == MAX_MATCHES:
                        return ToolResult("warning", "match limit reached", "\n".join(matches))
        return ToolResult("success", f"{len(matches)} matches", "\n".join(matches))

    def _write_file(self, path: str, content: str) -> ToolResult:
        try:
            absolute, relative = self.resolve(path)
        except PermissionError as exc:
            return ToolResult("error", str(exc))
        if not self.is_editable(relative):
            return ToolResult("error", f"{relative} is not editable")
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_text(content)
        return ToolResult("success", f"wrote {relative}")


def _matches(relative: str, patterns: Sequence[str]) -> bool:
    path = PurePosixPath(relative)
    return any(path.full_match(pattern) for pattern in patterns)


__all__ = ["Workspace"]
