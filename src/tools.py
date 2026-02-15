"""Tool functions for the rank-bot judge agents.

Each tool is registered via ``@function_tool`` from the OpenAI Agents SDK.
They return ``str`` — either the content on success, or an ``"Error: …"``
prefixed message on failure.  Errors are never raised; they are returned as
values (errors-as-return-values pattern).
"""

import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from agents import function_tool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base directories — resolved once at import time from env or heuristic
# ---------------------------------------------------------------------------

_BASE_DIR = Path(os.environ.get("RANK_BOT_BASE", Path(__file__).resolve().parents[1]))
_C4_DIR = _BASE_DIR / "Submissions-C4"
_C3_DIR = _BASE_DIR / "Submissions_C3"

_GIT_TIMEOUT = int(os.environ.get("RANK_BOT_GIT_TIMEOUT", "30"))
_MAX_FILE_LINES = int(os.environ.get("RANK_BOT_MAX_FILE_LINES", "300"))

_IGNORE_PATTERNS = {"__pycache__", ".pyc", "node_modules", ".git", ".DS_Store"}


def _filter_paths(paths: list[str]) -> list[str]:
    """Remove noisy paths from file listings.

    Args:
        paths: Raw list of file paths.

    Returns:
        Filtered list with noise directories removed.
    """
    return [
        p for p in paths
        if not any(ig in p for ig in _IGNORE_PATTERNS)
    ]


def _resolve_repo_dir(repo: str) -> Path:
    """Map a repo label to its local path.

    Args:
        repo: Either 'c3' or 'c4'.

    Returns:
        Path to the corresponding submissions directory.

    Raises:
        AssertionError: If repo is not 'c3' or 'c4'.
    """
    match repo:
        case "c4":
            return _C4_DIR
        case "c3":
            return _C3_DIR
        case _:
            assert False, f"Unknown repo: {repo!r}, expected 'c3' or 'c4'"


# ---------------------------------------------------------------------------
# Tool 1: git_list_files
# ---------------------------------------------------------------------------


@function_tool
def git_list_files(repo: str, branch: str, path: str = "") -> str:
    """List all files on a given git branch, optionally under a subdirectory.

    Args:
        repo: Which repository to query — 'c3' or 'c4'.
        branch: The git branch name (e.g. 'Group_1', 'main').
        path: Optional subdirectory path to scope the listing.

    Returns:
        Newline-separated file listing, or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    ref = f"origin/{branch}" if not branch.startswith("origin/") else branch

    cmd = ["git", "ls-tree", "-r", "--name-only", ref]
    if path:
        cmd.append(path)

    log.info("git_list_files: repo=%s branch=%s path=%s", repo, branch, path)
    result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )

    if result.returncode != 0:
        log.warning("git_list_files failed: %s", result.stderr.strip())
        return f"Error: {result.stderr.strip()}"

    lines = _filter_paths(result.stdout.strip().splitlines())
    if len(lines) > 200:
        lines = lines[:200]
        lines.append(f"... (truncated, {len(lines)} of many files shown)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: git_read_file
# ---------------------------------------------------------------------------


@function_tool
def git_read_file(repo: str, branch: str, filepath: str) -> str:
    """Read the contents of a single file from a git branch without checkout.

    Args:
        repo: Which repository — 'c3' or 'c4'.
        branch: The git branch name.
        filepath: Path to the file within the branch.

    Returns:
        File contents (possibly truncated), or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    ref = f"origin/{branch}" if not branch.startswith("origin/") else branch
    object_spec = f"{ref}:{filepath}"

    log.info("git_read_file: repo=%s spec=%s", repo, object_spec)
    result = subprocess.run(
        ["git", "show", object_spec],
        cwd=repo_dir, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )

    if result.returncode != 0:
        log.warning("git_read_file failed: %s", result.stderr.strip())
        return f"Error: {result.stderr.strip()}"

    lines = result.stdout.splitlines()
    if len(lines) > _MAX_FILE_LINES:
        lines = lines[:_MAX_FILE_LINES]
        lines.append(f"\n... (truncated at {_MAX_FILE_LINES} lines)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: read_local_file
# ---------------------------------------------------------------------------


@function_tool
def read_local_file(repo: str, filepath: str) -> str:
    """Read a file from the local filesystem (for projects on the main branch).

    Args:
        repo: Which repository — 'c3' or 'c4'.
        filepath: Relative path within the repo directory.

    Returns:
        File contents (possibly truncated), or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    full = (repo_dir / filepath).resolve()
    assert full.is_relative_to(repo_dir), f"Path traversal blocked: {filepath}"

    log.info("read_local_file: %s", full)
    if not full.is_file():
        return f"Error: file not found: {filepath}"

    content = full.read_text(errors="replace")
    lines = content.splitlines()
    if len(lines) > _MAX_FILE_LINES:
        lines = lines[:_MAX_FILE_LINES]
        lines.append(f"\n... (truncated at {_MAX_FILE_LINES} lines)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: list_local_directory
# ---------------------------------------------------------------------------


@function_tool
def list_local_directory(repo: str, dirpath: str) -> str:
    """Recursively list files in a local directory with sizes.

    Args:
        repo: Which repository — 'c3' or 'c4'.
        dirpath: Relative directory path within the repo.

    Returns:
        Newline-separated listing of 'path  (size bytes)', or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    full = (repo_dir / dirpath).resolve()
    assert full.is_relative_to(repo_dir), f"Path traversal blocked: {dirpath}"

    log.info("list_local_directory: %s", full)
    if not full.is_dir():
        return f"Error: directory not found: {dirpath}"

    entries: list[str] = []
    for p in sorted(full.rglob("*")):
        if p.is_file() and not any(ig in str(p) for ig in _IGNORE_PATTERNS):
            rel = p.relative_to(repo_dir)
            size = p.stat().st_size
            entries.append(f"{rel}  ({size} bytes)")
        if len(entries) >= 200:
            entries.append("... (truncated at 200 files)")
            break

    return "\n".join(entries) if entries else "Error: directory is empty"


# ---------------------------------------------------------------------------
# Tool 5: extract_zip_and_list
# ---------------------------------------------------------------------------


@function_tool
def extract_zip_and_list(repo: str, branch: str, zip_path: str) -> str:
    """Extract a .zip file from a git branch to a temp directory and list its contents.

    Uses ``git show`` to pipe the zip bytes, then extracts with ``zipfile``.

    Args:
        repo: Which repository — 'c3' or 'c4'.
        branch: Git branch containing the zip file.
        zip_path: Path to the .zip file within the branch.

    Returns:
        Newline-separated file listing from the extracted zip, or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    ref = f"origin/{branch}" if not branch.startswith("origin/") else branch
    object_spec = f"{ref}:{zip_path}"

    log.info("extract_zip_and_list: repo=%s spec=%s", repo, object_spec)
    result = subprocess.run(
        ["git", "show", object_spec],
        cwd=repo_dir, capture_output=True, timeout=_GIT_TIMEOUT,
    )

    if result.returncode != 0:
        return f"Error: could not read zip: {result.stderr.decode(errors='replace').strip()}"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    extract_dir = tempfile.mkdtemp(prefix="rankbot_zip_")
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(extract_dir)

        entries: list[str] = []
        for p in sorted(Path(extract_dir).rglob("*")):
            if p.is_file() and not any(ig in str(p) for ig in _IGNORE_PATTERNS):
                rel = p.relative_to(extract_dir)
                size = p.stat().st_size
                entries.append(f"{rel}  ({size} bytes)")
            if len(entries) >= 200:
                entries.append("... (truncated at 200 files)")
                break

        return "\n".join(entries) if entries else "Error: zip was empty"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tool 6: read_file_from_zip
# ---------------------------------------------------------------------------


@function_tool
def read_file_from_zip(repo: str, branch: str, zip_path: str, file_inside_zip: str) -> str:
    """Read a specific file from inside a .zip archive on a git branch.

    Args:
        repo: Which repository — 'c3' or 'c4'.
        branch: Git branch containing the zip file.
        zip_path: Path to the .zip file within the branch.
        file_inside_zip: Path of the file inside the zip to read.

    Returns:
        File contents (possibly truncated), or an error string.
    """
    repo_dir = _resolve_repo_dir(repo)
    ref = f"origin/{branch}" if not branch.startswith("origin/") else branch
    object_spec = f"{ref}:{zip_path}"

    result = subprocess.run(
        ["git", "show", object_spec],
        cwd=repo_dir, capture_output=True, timeout=_GIT_TIMEOUT,
    )

    if result.returncode != 0:
        return f"Error: could not read zip: {result.stderr.decode(errors='replace').strip()}"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            if file_inside_zip not in zf.namelist():
                return f"Error: {file_inside_zip} not found in zip. Available: {zf.namelist()[:20]}"
            content = zf.read(file_inside_zip).decode(errors="replace")

        lines = content.splitlines()
        if len(lines) > _MAX_FILE_LINES:
            lines = lines[:_MAX_FILE_LINES]
            lines.append(f"\n... (truncated at {_MAX_FILE_LINES} lines)")

        return "\n".join(lines)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Convenience: all tools as a list for agent registration
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    git_list_files,
    git_read_file,
    read_local_file,
    list_local_directory,
    extract_zip_and_list,
    read_file_from_zip,
]
