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

from agents import Runner
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from dotenv import load_dotenv
from openai import APIStatusError

from agents_factory import create_agents
from config import Config
from models import (
    CodeQualityResult,
    ConceptScoreResult,
    DifficultyScoreEntry,
    GroupInfo,
)
from scoring import (
    build_project_prompt,
    generate_report,
    load_c3_reference,
    load_groups_from_csv,
    load_syllabus,
    write_scores_to_csv,
)

log = logging.getLogger("rank_bot")


# ---------------------------------------------------------------------------
# Summary collector — builds a text summary of each project for the
# difficulty judge (which has no tools and works from text only)
# ---------------------------------------------------------------------------


async def collect_project_summary(
    group: GroupInfo,
    concept_judge: object,
    repo: str = "c4",
) -> str:
    """Collect a text summary of a project by running a lightweight probe.

    For groups with no submission, returns a placeholder.  Otherwise, uses
    the concept judge's tools (via a temporary agent run) to list files and
    read the README, then returns a textual summary.

    Args:
        group: Parsed group info.
        concept_judge: An Agent with tools attached (reused for tool access).
        repo: Which repo to probe ('c3' or 'c4').

    Returns:
        A text summary string suitable for the difficulty judge.
    """
    if group.branch is None:
        return f"Group {group.group}: No submission — no code available."

    prompt = build_project_prompt(group, repo=repo)
    # We use the concept judge here just to get tool access for file listing.
    # The actual scoring is done separately.
    summary_prompt = (
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

    from agents import Agent, ModelSettings
    from tools import ALL_TOOLS

    # Create a lightweight summarizer agent (no structured output)
    summarizer = Agent(
        name="ProjectSummarizer",
        model=concept_judge.model,
        model_settings=ModelSettings(max_tokens=4096),
        instructions="You are a technical project summarizer. Use the provided tools to explore the project and return a concise summary. Do NOT score anything.",
        tools=ALL_TOOLS,
    )

    result = await Runner.run(summarizer, summary_prompt, max_turns=15)
    return f"## Group {group.group}\n{result.final_output}"


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
        repo: Which repository to evaluate ('c3' or 'c4').
        groups_override: If set, only evaluate these group numbers.
    """
    # --- Phase 0: Load data ---
    log.info("Phase 0: Loading data")
    syllabus = load_syllabus(config.syllabus_csv_path)
    c3_ref = load_c3_reference(config.c3_csv_path)

    csv_path = config.c4_csv_path if repo == "c4" else config.c3_csv_path
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

    # --- Phase 0.5: Create agents ---
    concept_judge, quality_judge, difficulty_judge = create_agents(
        config, syllabus, c3_ref,
    )

    # --- Phase 1: Collect project summaries for difficulty judge ---
    log.info("Phase 1: Collecting project summaries")
    summaries: dict[int, str] = {}
    for g in groups:
        log.info("Collecting summary for Group %d", g.group)
        try:
            summaries[g.group] = await collect_project_summary(g, concept_judge, repo=repo)
        except (APIStatusError, MaxTurnsExceeded, ModelBehaviorError) as exc:
            log.error("Failed to collect summary for Group %d: %s", g.group, exc)
            summaries[g.group] = f"Group {g.group}: Summary collection failed."
        log.info("Summary collected for Group %d", g.group)

    # --- Phase 2: Per-project scoring (Concept + Code Quality) ---
    log.info("Phase 2: Scoring Concept and Code Quality")
    concept_scores: dict[int, ConceptScoreResult] = {}
    quality_scores: dict[int, CodeQualityResult] = {}

    for g in evaluable:
        prompt = build_project_prompt(g, repo=repo)

        try:
            log.info("Scoring Group %d — Concept", g.group)
            concept_result = await Runner.run(concept_judge, prompt, max_turns=20)
            concept_scores[g.group] = concept_result.final_output
            log.info(
                "Group %d Concept: %d/10",
                g.group,
                concept_scores[g.group].score,
            )
        except (APIStatusError, MaxTurnsExceeded, ModelBehaviorError) as exc:
            log.error("Concept scoring failed for Group %d: %s", g.group, exc)

        try:
            log.info("Scoring Group %d — Code Quality", g.group)
            quality_result = await Runner.run(quality_judge, prompt, max_turns=20)
            quality_scores[g.group] = quality_result.final_output
            log.info(
                "Group %d Code Quality: %d/10",
                g.group,
                quality_scores[g.group].score,
            )
        except (APIStatusError, MaxTurnsExceeded, ModelBehaviorError) as exc:
            log.error("Quality scoring failed for Group %d: %s", g.group, exc)

    # --- Phase 3: Relative difficulty scoring (all at once) ---
    log.info("Phase 3: Scoring Difficulty (relative)")
    all_summaries_text = "\n\n---\n\n".join(
        summaries[g.group] for g in groups
    )
    difficulty_scores: dict[int, DifficultyScoreEntry] = {}
    try:
        diff_result = await Runner.run(difficulty_judge, all_summaries_text, max_turns=5)
        difficulty_scores = {
            s.group: s for s in diff_result.final_output.scores
        }
        for gn, entry in sorted(difficulty_scores.items()):
            log.info("Group %d Difficulty: %d/10", gn, entry.score)
    except (APIStatusError, MaxTurnsExceeded, ModelBehaviorError) as exc:
        log.error("Difficulty scoring failed: %s", exc)

    # --- Phase 4: Generate report ---
    log.info("Phase 4: Generating report")
    report = generate_report(groups, concept_scores, difficulty_scores, quality_scores)

    base_dir = config.repo_c4_path.parent
    cohort = repo.upper()
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
        scores_list.append({
            "group": gn,
            "concept_score": c.score if c else 0,
            "concept_justification": c.justification if c else "No submission",
            "concept_concepts_found": c.concepts_found if c else [],
            "difficulty_score": d.score if d else 0,
            "difficulty_justification": d.justification if d else "No submission",
            "code_quality_score": q.score if q else 0,
            "code_quality_justification": q.justification if q else "No submission",
            "total": (c.score if c else 0) + (d.score if d else 0) + (q.score if q else 0),
        })

    scores_list.sort(key=lambda x: x["total"], reverse=True)
    json_path = base_dir / f"{cohort.lower()}_scores.json"
    json_path.write_text(json.dumps(scores_list, indent=2), encoding="utf-8")
    log.info("Scores written to %s", json_path)

    # --- Phase 5: Update scorecard CSV with scores ---
    log.info("Phase 5: Updating scorecard CSV")
    write_scores_to_csv(csv_path, concept_scores, difficulty_scores, quality_scores)

    # Print summary table
    print(f"\n{'='*60}")
    print(f" {cohort} Hackathon Evaluation Results")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Group':<7} {'Concept':<9} {'Diff':<6} {'Quality':<9} {'Total':<6}")
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
        uv run python src/main.py                     # Evaluate all C4 groups
        uv run python src/main.py --repo c3            # Evaluate C3 instead
        uv run python src/main.py --groups 2 4 13      # Only specific groups
        uv run python src/main.py --repo c3 --groups 4 5 12  # C3 calibration
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
