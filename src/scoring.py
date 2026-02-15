"""Pure functions for CSV parsing, URL parsing, and report generation.

No side effects — all I/O is done via arguments and return values.
"""

import csv
import logging
import re
from pathlib import Path
from urllib.parse import unquote

from models import (CodeQualityResult, ConceptScoreResult,
                    DifficultyScoreEntry, GroupInfo)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL / project-link parsing
# ---------------------------------------------------------------------------


def parse_project_link(link: str) -> tuple[str | None, str | None, bool, bool]:
    """Parse a GitHub project URL into (branch, path, is_zip, is_commit).

    Handles patterns:
      - tree/<branch>/<path>        → (branch, path, False, False)
      - tree/<commit_hash>/<path>   → (commit_hash, path, False, True)
      - blob/<branch>/<path>.zip    → (branch, path, True, False)
      - commit/<hash>               → (hash, None, False, True)
      - empty / None                → (None, None, False, False)

    Args:
        link: Raw URL from the scorecard CSV.

    Returns:
        Tuple of (branch, path, is_zip, is_commit).
    """
    if not link or not link.strip():
        return (None, None, False, False)

    link = link.strip()

    # commit/<hash>
    m = re.search(r"/commit/([0-9a-f]{7,40})", link)
    if m:
        return (m.group(1), None, False, True)

    # blob/<branch>/<path>.zip
    m = re.search(r"/blob/([^/]+)/(.+\.zip)$", link)
    if m:
        branch = unquote(m.group(1))
        path = unquote(m.group(2))
        return (branch, path, True, False)

    # tree/<ref>/<path>  or  tree/<ref>
    m = re.search(r"/tree/([^/]+)(?:/(.+))?$", link)
    if m:
        ref = unquote(m.group(1))
        path = unquote(m.group(2)) if m.group(2) else None

        # Check if ref looks like a commit hash (40 hex chars)
        is_commit = bool(re.fullmatch(r"[0-9a-f]{7,40}", ref))
        return (ref, path, False, is_commit)

    # blob/<branch>/<path>  (non-zip blob, e.g. a folder link mis-categorised)
    m = re.search(r"/blob/([^/]+)/(.+)$", link)
    if m:
        branch = unquote(m.group(1))
        path = unquote(m.group(2))
        return (branch, path, False, False)

    return (None, None, False, False)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_groups_from_csv(csv_path: Path) -> list[GroupInfo]:
    """Parse the scorecard CSV into a list of GroupInfo objects.

    Args:
        csv_path: Path to the CSV file (C3 or C4).

    Returns:
        List of GroupInfo, one per group row that has a numeric group number.
    """
    groups: list[GroupInfo] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_group = row.get("Group", "").strip()
            if not raw_group or not raw_group.isdigit():
                continue

            project_link = row.get("Project Link", "").strip()
            video_link = row.get("Video Link", "").strip()
            branch, path, is_zip, is_commit = parse_project_link(project_link)

            groups.append(
                GroupInfo(
                    group=int(raw_group),
                    project_link=project_link,
                    video_link=video_link,
                    branch=branch,
                    path=path,
                    is_zip=is_zip,
                    is_commit=is_commit,
                )
            )

    return groups


def load_syllabus(csv_path: Path) -> str:
    """Load the syllabus CSV and format it as a readable text block.

    Args:
        csv_path: Path to the syllabus CSV.

    Returns:
        Formatted string summarising each sprint's topics and outcomes.
    """
    lines: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sprint_title = row.get("Sprint Title", "").strip()
            topics = row.get("Topics", "").strip()
            description = row.get("Description", "").strip()
            outcomes = row.get("Outcomes", "").strip()
            tools = row.get("Tools - Sprint Wise", "").strip()

            if not sprint_title and not topics:
                continue

            block = []
            if sprint_title:
                block.append(f"## {sprint_title}")
            if topics:
                block.append(f"**Topics:** {topics}")
            if description:
                block.append(f"**Description:** {description}")
            if outcomes:
                block.append(f"**Outcomes:** {outcomes}")
            if tools:
                block.append(f"**Tools:** {tools}")
            block.append("")

            lines.append("\n".join(block))

    return "\n".join(lines)


def load_c3_reference(csv_path: Path) -> str:
    """Load C3 scores and format as a reference table for calibration.

    Args:
        csv_path: Path to the C3 scorecard CSV.

    Returns:
        Formatted reference table string.
    """
    lines: list[str] = [
        "# C3 Reference Scores (for calibration)",
        "",
        "| Group | Concept | Difficulty | Code Quality | Total | Comments |",
        "|-------|---------|------------|--------------|-------|----------|",
    ]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grp = row.get("Group", "").strip()
            if not grp or not grp.isdigit():
                continue
            concept = row.get("Concept Score (10)", "").strip() or "-"
            diff = row.get("Difficulty Level (10)", "").strip() or "-"
            quality = row.get("Code Quality (10)", "").strip() or "-"
            total = row.get("Total (30)", "").strip() or "-"
            comments = row.get("Comments", "").strip() or ""
            lines.append(
                f"| {grp} | {concept} | {diff} | {quality} | {total} | {comments} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building for per-project evaluation
# ---------------------------------------------------------------------------


def build_project_prompt(group: GroupInfo, repo: str = "c4") -> str:
    """Build the evaluation prompt for a single group, with tool-use hints.

    Uses ``match`` on the group's link characteristics to provide the right
    instructions for tool calls.

    Args:
        group: Parsed group metadata.
        repo: Which repo this group belongs to ('c3' or 'c4').

    Returns:
        Prompt string to send to the judge agents.
    """
    header = f"# Evaluate Group {group.group}\n\n"

    match (group.branch, group.is_zip, group.is_commit):
        case (None, _, _):
            return header + (
                "This group has no submission link. "
                "Score 0/10 and explain that no code was available for review."
            )
        case (_, True, _):
            assert group.branch is not None
            assert group.path is not None
            return header + (
                f"This group submitted a .zip file.\n"
                f"- Repo: '{repo}'\n"
                f"- Branch: '{group.branch}'\n"
                f"- Zip path: '{group.path}'\n\n"
                f"Use the `extract_zip_and_list` tool to list files in the zip, "
                f"then use `read_file_from_zip` to read key files like README, "
                f"main app files, and agent/graph definitions.\n\n"
                f"Note: .zip submissions indicate poor code quality practices "
                f"(should have been committed properly to git)."
            )
        case (_, _, True):
            assert group.branch is not None
            path_hint = f"  Path hint: '{group.path}'" if group.path else ""
            return header + (
                f"This group's link points to a specific commit.\n"
                f"- Repo: '{repo}'\n"
                f"- Commit/Branch ref: '{group.branch}'\n"
                f"{path_hint}\n\n"
                f"Use `git_list_files` with branch='{group.branch}' to list files, "
                f"then `git_read_file` to read key files.\n"
                f"Look for README, app entry points, agent definitions, and graph files."
            )
        case _:
            assert group.branch is not None
            use_local = group.branch == "main"
            path_hint = group.path or ""

            if use_local:
                return header + (
                    f"This project is on the main branch.\n"
                    f"- Repo: '{repo}'\n"
                    f"- Directory: '{path_hint}'\n\n"
                    f"Use `list_local_directory` with repo='{repo}' and "
                    f"dirpath='{path_hint}' to see the file structure, "
                    f"then use `read_local_file` to read key files.\n"
                    f"Look for README, app entry points, agent definitions, and graph files."
                )
            return header + (
                f"This project is on a feature branch.\n"
                f"- Repo: '{repo}'\n"
                f"- Branch: '{group.branch}'\n"
                f"- Path: '{path_hint}'\n\n"
                f"Use `git_list_files` with repo='{repo}', branch='{group.branch}' "
                f"and path='{path_hint}' to see the file structure, "
                f"then use `git_read_file` to read key files.\n"
                f"Look for README, app entry points, agent definitions, and graph files."
            )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    groups: list[GroupInfo],
    concept_scores: dict[int, ConceptScoreResult],
    difficulty_scores: dict[int, DifficultyScoreEntry],
    quality_scores: dict[int, CodeQualityResult],
) -> str:
    """Generate the final markdown evaluation report with rankings.

    Args:
        groups: All parsed group metadata.
        concept_scores: Concept scores keyed by group number.
        difficulty_scores: Difficulty scores keyed by group number.
        quality_scores: Code quality scores keyed by group number.

    Returns:
        Complete markdown report string.
    """
    lines: list[str] = [
        "# C4 Hackathon Evaluation Report",
        "",
        "## Summary",
        "",
        "| Rank | Group | Concept | Difficulty | Code Quality | Total |",
        "|------|-------|---------|------------|--------------|-------|",
    ]

    # Build rows with totals for sorting
    rows: list[tuple[int, int, int, int, int]] = []
    for g in groups:
        gn = g.group
        c = concept_scores.get(gn)
        d = difficulty_scores.get(gn)
        q = quality_scores.get(gn)
        cs = c.score if c else 0
        ds = d.score if d else 0
        qs = q.score if q else 0
        total = cs + ds + qs
        rows.append((gn, cs, ds, qs, total))

    rows.sort(key=lambda r: r[4], reverse=True)

    for rank, (gn, cs, ds, qs, total) in enumerate(rows, 1):
        lines.append(f"| {rank} | {gn} | {cs} | {ds} | {qs} | {total} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed per-group sections
    lines.append("## Detailed Evaluations")
    lines.append("")

    for gn, _cs_val, _ds_val, _qs_val, total in rows:
        lines.append(f"### Group {gn} (Total: {total}/30)")
        lines.append("")

        c = concept_scores.get(gn)
        if c:
            lines.append(f"**Concept Score: {c.score}/10**")
            lines.append(f"- Concepts found: {', '.join(c.concepts_found)}")
            lines.append(f"- Concepts missing: {', '.join(c.concepts_missing)}")
            lines.append(f"- Justification: {c.justification}")
        else:
            lines.append("**Concept Score: 0/10** — No submission")

        lines.append("")

        d = difficulty_scores.get(gn)
        if d:
            lines.append(f"**Difficulty Score: {d.score}/10**")
            lines.append(f"- Justification: {d.justification}")
        else:
            lines.append("**Difficulty Score: 0/10** — No submission")

        lines.append("")

        q = quality_scores.get(gn)
        if q:
            lines.append(f"**Code Quality Score: {q.score}/10**")
            lines.append(f"- Folder structure: {'✓' if q.has_proper_folders else '✗'}")
            lines.append(
                f"- README: {'✓' if q.has_readme else '✗'} ({q.readme_quality})"
            )
            lines.append(f"- Requirements: {'✓' if q.has_requirements_txt else '✗'}")
            lines.append(f"- Env handling: {'✓' if q.has_env_handling else '✗'}")
            lines.append(f"- Organization: {q.code_organization}")
            lines.append(f"- Justification: {q.justification}")
        else:
            lines.append("**Code Quality Score: 0/10** — No submission")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV writer — updates the scorecard CSV with computed scores
# ---------------------------------------------------------------------------


def write_scores_to_csv(
    csv_path: Path,
    concept_scores: dict[int, ConceptScoreResult],
    difficulty_scores: dict[int, DifficultyScoreEntry],
    quality_scores: dict[int, CodeQualityResult],
) -> None:
    """Read the scorecard CSV, fill in scores, compute totals and positions, write back.

    Updates columns: Concept Score (10), Difficulty Level (10),
    Code Quality (10), Total (30), Position.

    Args:
        csv_path: Path to the scorecard CSV to update in-place.
        concept_scores: Concept scores keyed by group number.
        difficulty_scores: Difficulty scores keyed by group number.
        quality_scores: Code quality scores keyed by group number.
    """
    # Read existing rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None, "CSV has no header row"
        rows = list(reader)

    # Fill in scores — only overwrite if we have a new score, preserve existing
    for row in rows:
        raw_group = row.get("Group", "").strip()
        if not raw_group or not raw_group.isdigit():
            continue
        gn = int(raw_group)

        c = concept_scores.get(gn)
        d = difficulty_scores.get(gn)
        q = quality_scores.get(gn)

        if c:
            row["Concept Score (10)"] = str(c.score)
        if d:
            row["Difficulty Level (10)"] = str(d.score)
        if q:
            row["Code Quality (10)"] = str(q.score)

        # Recompute total from whatever is in the row now
        cs = (
            int(row["Concept Score (10)"])
            if row.get("Concept Score (10)", "").strip().isdigit()
            else 0
        )
        ds = (
            int(row["Difficulty Level (10)"])
            if row.get("Difficulty Level (10)", "").strip().isdigit()
            else 0
        )
        qs = (
            int(row["Code Quality (10)"])
            if row.get("Code Quality (10)", "").strip().isdigit()
            else 0
        )
        total = cs + ds + qs
        row["Total (30)"] = str(total) if total > 0 else ""

    # Compute positions based on Total (descending)
    scored_rows = [
        (row, int(row["Total (30)"]))
        for row in rows
        if row.get("Total (30)", "").strip().isdigit()
    ]
    scored_rows.sort(key=lambda pair: pair[1], reverse=True)

    current_position = 0
    prev_total = None
    for i, (row, total) in enumerate(scored_rows):
        if total != prev_total:
            current_position = i + 1
        row["Position"] = str(current_position)
        prev_total = total

    # Write back
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Updated CSV: %s", csv_path)
