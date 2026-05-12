"""Orchestrator for the rank-bot hackathon judge agent.

Entry point: ``uv run python src/main.py``

Phases:
    1. Load config, syllabus, C3 reference, and C4 groups.
    2. Collect project summaries for the difficulty judge.
    3. Score each project on Concept and Code Quality (per-project).
    4. Score all projects on Difficulty (relative, all at once).
    5. Generate report and JSON output.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from config import Config
from cursor_runner import _schema_suffix, run_cursor_agent, run_structured_agent
from models import (
    AllDifficultyScores,
    CodeQualityResult,
    ConceptScoreResult,
    DifficultyScoreEntry,
    GroupInfo,
)
from prompts import (
    build_code_quality_judge_instructions,
    build_concept_judge_instructions,
    build_difficulty_judge_instructions,
)
from scoring import (
    build_project_prompt,
    generate_report,
    load_c3_reference,
    load_groups_from_csv,
    load_groups_from_xlsx,
    load_syllabus,
    write_scores_to_csv,
    write_scores_to_xlsx,
)

log = logging.getLogger("rank_bot")


# ---------------------------------------------------------------------------
# Summary collector — builds a text summary of each project for the
# difficulty judge (which has no tools and works from text only)
# ---------------------------------------------------------------------------


def _repo_workspace(config: Config, repo: str) -> Path:
    """Resolve the workspace directory for a submissions repo label.

    Args:
        config: Application configuration.
        repo: Which repository to evaluate ('c3' or 'c4').

    Returns:
        Path to the local submissions repository.

    Raises:
        AssertionError: If repo is not 'c3' or 'c4'.
    """
    match repo:
        case "c4":
            return config.repo_c4_path
        case "c3":
            return config.repo_c3_path
        case "c6":
            # External repos are cloned by the agent; use base dir as workspace
            return config.repo_c4_path.parent
        case _:
            assert False, f"Unknown repo: {repo!r}, expected 'c3', 'c4', or 'c6'"


async def collect_project_summary(
    group: GroupInfo,
    config: Config,
    repo: str = "c4",
) -> str:
    """Collect a text summary of a project by running a lightweight probe.

    For groups with no submission, returns a placeholder. Otherwise, uses
    Cursor CLI to inspect the project and return a textual summary.

    Args:
        group: Parsed group info.
        config: Application configuration.
        repo: Which repo to probe ('c3' or 'c4').

    Returns:
        A text summary string suitable for the difficulty judge.

    Raises:
        RuntimeError: If Cursor CLI fails.
        json.JSONDecodeError: If Cursor CLI emits malformed JSON.
        KeyError: If Cursor CLI omits the result field.
    """
    if group.branch is None:
        return f"Group {group.group}: No submission — no code available."

    prompt = build_project_prompt(group, repo=repo)
    summary_prompt = (
        "You are a technical project summarizer. Use Cursor CLI's shell and "
        "file-reading capabilities to explore the project. Do NOT score "
        "anything.\n\n"
        f"{prompt}\n\n"
        "DO NOT SCORE. Instead, provide a brief technical summary of this project:\n"
        "1. What does the project do? (1-2 sentences)\n"
        "2. What key technologies/frameworks are used? (list them)\n"
        "3. How is the agent/graph structured? (linear, conditional, loops?)\n"
        "4. How many agents/nodes are there?\n"
        "5. What external integrations exist?\n"
        "6. Any notable patterns (RAG, multimodal, debate, reflection)?\n\n"
        "Keep it concise — 10-15 lines max."
    )

    result = await run_cursor_agent(
        summary_prompt,
        workspace=_repo_workspace(config, repo),
        model=config.cursor_model,
        api_key=config.cursor_api_key,
    )
    return f"## Group {group.group}\n{result}"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def run_evaluation(
    config: Config,
    repo: str = "c4",
    groups_override: list[int] | None = None,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        config: Application configuration.
        repo: Which repository to evaluate ('c3', 'c4', or 'c6').
        groups_override: If set, only evaluate these group numbers.
    """
    # --- Phase 0: Load data ---
    log.info("Phase 0: Loading data")
    syllabus = load_syllabus(config.syllabus_csv_path)
    c3_ref = load_c3_reference(config.c3_csv_path)

    match repo:
        case "c6":
            all_groups = load_groups_from_xlsx(
                config.c6_xlsx_path, config.c6_sheet_name
            )
            csv_path: Path | None = None
        case "c4":
            csv_path = config.c4_csv_path
            all_groups = load_groups_from_csv(csv_path)
        case _:
            csv_path = config.c3_csv_path
            all_groups = load_groups_from_csv(csv_path)

    if groups_override:
        groups = [g for g in all_groups if g.group in groups_override]
    else:
        groups = all_groups

    evaluable = [g for g in groups if g.branch is not None]
    log.info(
        "Loaded %d groups (%d evaluable)",
        len(groups),
        len(evaluable),
    )

    # --- Phase 0.5: Build Cursor CLI prompts ---
    concept_instructions = build_concept_judge_instructions(
        syllabus, c3_ref
    ) + _schema_suffix(ConceptScoreResult)
    quality_instructions = build_code_quality_judge_instructions(
        c3_ref
    ) + _schema_suffix(CodeQualityResult)
    difficulty_instructions = build_difficulty_judge_instructions(
        c3_ref
    ) + _schema_suffix(AllDifficultyScores)

    # Semaphore caps concurrent Cursor CLI subprocesses to avoid API overload
    sem = asyncio.Semaphore(5)
    workspace = _repo_workspace(config, repo)

    # --- Phase 1: Collect project summaries for difficulty judge (parallel) ---
    log.info("Phase 1: Collecting project summaries (parallel)")

    async def _summary_task(g: GroupInfo) -> tuple[int, str]:
        async with sem:
            log.info("Collecting summary for Group %d", g.group)
            try:
                result = await collect_project_summary(g, config, repo=repo)
                log.info("Summary collected for Group %d", g.group)
                return g.group, result
            except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
                log.error("Failed to collect summary for Group %d: %s", g.group, exc)
                return g.group, f"Group {g.group}: Summary collection failed."

    summary_results = await asyncio.gather(*[_summary_task(g) for g in groups])
    summaries: dict[int, str] = dict(summary_results)

    # --- Phase 2: Per-project scoring (Concept + Code Quality, parallel) ---
    log.info("Phase 2: Scoring Concept and Code Quality (parallel)")
    concept_scores: dict[int, ConceptScoreResult] = {}
    quality_scores: dict[int, CodeQualityResult] = {}

    async def _concept_task(g: GroupInfo) -> tuple[int, ConceptScoreResult | None]:
        prompt = build_project_prompt(g, repo=repo)
        async with sem:
            log.info("Scoring Group %d — Concept", g.group)
            try:
                score = await run_structured_agent(
                    f"{concept_instructions}\n\n{prompt}",
                    workspace=workspace,
                    model=config.cursor_model,
                    output_type=ConceptScoreResult,
                    api_key=config.cursor_api_key,
                )
                log.info("Group %d Concept: %d/10", g.group, score.score)
                return g.group, score
            except (
                RuntimeError,
                json.JSONDecodeError,
                ValidationError,
                KeyError,
            ) as exc:
                log.error("Concept scoring failed for Group %d: %s", g.group, exc)
                return g.group, None

    async def _quality_task(g: GroupInfo) -> tuple[int, CodeQualityResult | None]:
        prompt = build_project_prompt(g, repo=repo)
        async with sem:
            log.info("Scoring Group %d — Code Quality", g.group)
            try:
                score = await run_structured_agent(
                    f"{quality_instructions}\n\n{prompt}",
                    workspace=workspace,
                    model=config.cursor_model,
                    output_type=CodeQualityResult,
                    api_key=config.cursor_api_key,
                )
                log.info("Group %d Code Quality: %d/10", g.group, score.score)
                return g.group, score
            except (
                RuntimeError,
                json.JSONDecodeError,
                ValidationError,
                KeyError,
            ) as exc:
                log.error("Quality scoring failed for Group %d: %s", g.group, exc)
                return g.group, None

    scoring_tasks = [_concept_task(g) for g in evaluable] + [
        _quality_task(g) for g in evaluable
    ]
    scoring_results = await asyncio.gather(*scoring_tasks)

    for gn, score in scoring_results:
        if score is None:
            continue
        match score:
            case ConceptScoreResult():
                concept_scores[gn] = score
            case CodeQualityResult():
                quality_scores[gn] = score

    # --- Phase 3: Relative difficulty scoring (all at once) ---
    log.info("Phase 3: Scoring Difficulty (relative)")
    all_summaries_text = "\n\n---\n\n".join(summaries[g.group] for g in groups)
    difficulty_scores: dict[int, DifficultyScoreEntry] = {}
    try:
        diff_result = await run_structured_agent(
            f"{difficulty_instructions}\n\n{all_summaries_text}",
            workspace=config.repo_c4_path.parent,
            model=config.cursor_model,
            output_type=AllDifficultyScores,
            api_key=config.cursor_api_key,
        )
        difficulty_scores = {s.group: s for s in diff_result.scores}
        for gn, entry in sorted(difficulty_scores.items()):
            log.info("Group %d Difficulty: %d/10", gn, entry.score)
    except (RuntimeError, json.JSONDecodeError, ValidationError, KeyError) as exc:
        log.error("Difficulty scoring failed: %s", exc)

    # --- Phase 4: Generate report ---
    log.info("Phase 4: Generating report")
    cohort = repo.upper()
    report = generate_report(
        groups, concept_scores, difficulty_scores, quality_scores, cohort=cohort
    )

    base_dir = config.repo_c4_path.parent
    report_path = base_dir / f"{cohort.lower()}_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", report_path)

    # JSON scores
    scores_list = []
    for g in groups:
        gn = g.group
        c = concept_scores.get(gn)
        d = difficulty_scores.get(gn)
        q = quality_scores.get(gn)
        scores_list.append(
            {
                "group": gn,
                "concept_score": c.score if c else 0,
                "concept_justification": c.justification if c else "No submission",
                "concept_concepts_found": c.concepts_found if c else [],
                "difficulty_score": d.score if d else 0,
                "difficulty_justification": d.justification if d else "No submission",
                "code_quality_score": q.score if q else 0,
                "code_quality_justification": q.justification if q else "No submission",
                "total": (c.score if c else 0)
                + (d.score if d else 0)
                + (q.score if q else 0),
            }
        )

    scores_list.sort(key=lambda x: x["total"], reverse=True)
    json_path = base_dir / f"{cohort.lower()}_scores.json"
    json_path.write_text(json.dumps(scores_list, indent=2), encoding="utf-8")
    log.info("Scores written to %s", json_path)

    # --- Phase 5: Update scorecard with scores ---
    log.info("Phase 5: Updating scorecard")
    if repo == "c6":
        write_scores_to_xlsx(
            config.c6_xlsx_path,
            config.c6_sheet_name,
            concept_scores,
            difficulty_scores,
            quality_scores,
        )
    else:
        assert csv_path is not None
        write_scores_to_csv(csv_path, concept_scores, difficulty_scores, quality_scores)

    # Print summary table
    print(f"\n{'='*60}")
    print(f" {cohort} Hackathon Evaluation Results")
    print(f"{'='*60}")
    print(
        f"{'Rank':<5} {'Group':<7} {'Concept':<9} {'Diff':<6} {'Quality':<9} {'Total':<6}"
    )
    print(f"{'-'*5} {'-'*7} {'-'*9} {'-'*6} {'-'*9} {'-'*6}")
    for rank, entry in enumerate(scores_list, 1):
        print(
            f"{rank:<5} {entry['group']:<7} "
            f"{entry['concept_score']:<9} {entry['difficulty_score']:<6} "
            f"{entry['code_quality_score']:<9} {entry['total']:<6}"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli() -> None:
    """CLI entry point for rank-bot.

    Usage:
        uv run python src/main.py                          # Evaluate all C4 groups
        uv run python src/main.py --repo c3                # Evaluate C3 instead
        uv run python src/main.py --repo c6                # Evaluate C6 (xlsx, external repos)
        uv run python src/main.py --groups 2 4 13          # Only specific groups
        uv run python src/main.py --repo c6 --groups 1 2   # C6 subset
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    config = Config.from_env()

    # Parse CLI args (simple, no argparse needed)
    args = sys.argv[1:]
    repo = "c4"
    groups_override: list[int] | None = None

    i = 0
    while i < len(args):
        match args[i]:
            case "--repo":
                repo = args[i + 1]
                i += 2
            case "--groups":
                groups_override = []
                i += 1
                while i < len(args) and not args[i].startswith("--"):
                    groups_override.append(int(args[i]))
                    i += 1
            case _:
                log.warning("Unknown argument: %s", args[i])
                i += 1

    log.info("Starting evaluation: repo=%s groups=%s", repo, groups_override)
    asyncio.run(run_evaluation(config, repo=repo, groups_override=groups_override))


if __name__ == "__main__":
    cli()
