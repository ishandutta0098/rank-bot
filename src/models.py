"""Domain types and Pydantic output models for the rank-bot judge agents.

Domain types are frozen dataclasses (product types).
Agent output types are Pydantic BaseModels for structured output validation.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain types (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupInfo:
    """Parsed metadata for a single hackathon submission group.

    Attributes:
        group: Group number (1-15).
        project_link: Raw URL from the scorecard CSV.
        video_link: Raw URL to the project demo video.
        branch: Git branch name extracted from the project link, or None.
        path: Subdirectory path within the branch, or None.
        is_zip: Whether the submission is a .zip file.
        is_commit: Whether the link points to a specific commit hash.
    """

    group: int
    project_link: str
    video_link: str
    branch: str | None
    path: str | None
    is_zip: bool
    is_commit: bool


# ---------------------------------------------------------------------------
# Agent output types (Pydantic BaseModel for structured output)
# ---------------------------------------------------------------------------


class ConceptScoreResult(BaseModel):
    """Output from the Concept Judge agent.

    Attributes:
        score: Concept score from 1 to 10.
        concepts_found: List of syllabus concepts identified in the project.
        concepts_missing: List of syllabus concepts not used.
        justification: Explanation of the score.
    """

    score: int = Field(ge=1, le=10)
    concepts_found: list[str]
    concepts_missing: list[str]
    justification: str


class CodeQualityResult(BaseModel):
    """Output from the Code Quality Judge agent.

    Attributes:
        score: Code quality score from 1 to 10.
        has_proper_folders: Whether the project uses a sensible folder structure.
        has_readme: Whether a README file exists.
        readme_quality: Brief assessment of the README.
        has_requirements_txt: Whether dependency management is present.
        has_env_handling: Whether .env / environment variables are handled.
        code_organization: Brief assessment of code modularity.
        justification: Explanation of the score.
    """

    score: int = Field(ge=1, le=10)
    has_proper_folders: bool
    has_readme: bool
    readme_quality: str
    has_requirements_txt: bool
    has_env_handling: bool
    code_organization: str
    justification: str


class DifficultyScoreEntry(BaseModel):
    """A single group's difficulty score within the relative evaluation.

    Attributes:
        group: Group number.
        score: Difficulty score from 1 to 10.
        justification: Explanation of the score relative to other projects.
    """

    group: int
    score: int = Field(ge=1, le=10)
    justification: str


class AllDifficultyScores(BaseModel):
    """Output from the Difficulty Judge agent — scores for all groups at once.

    Attributes:
        scores: List of per-group difficulty score entries.
    """

    scores: list[DifficultyScoreEntry]
