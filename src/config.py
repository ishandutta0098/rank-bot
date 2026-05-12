"""Configuration for the rank-bot hackathon judge agent.

Loads all settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the rank-bot pipeline.

    Attributes:
        cursor_model: Model identifier for Cursor CLI.
        cursor_api_key: Optional API key for Cursor CLI automation.
        repo_c4_path: Local path to the cloned C4 submissions repo.
        repo_c3_path: Local path to the cloned C3 submissions repo.
        c3_csv_path: Path to the C3 scorecard CSV.
        c4_csv_path: Path to the C4 scorecard CSV.
        syllabus_csv_path: Path to the syllabus CSV.
        max_file_lines: Maximum lines to read from a single file (truncation limit).
        git_timeout: Timeout in seconds for git subprocess calls.
    """

    cursor_model: str
    cursor_api_key: str | None
    repo_c4_path: Path
    repo_c3_path: Path
    c3_csv_path: Path
    c4_csv_path: Path
    syllabus_csv_path: Path
    max_file_lines: int = 300
    git_timeout: int = 30

    @classmethod
    def from_env(cls) -> "Config":
        """Build Config from environment variables.

        Returns:
            Config: Fully populated configuration.
        """
        base = Path(
            os.environ.get("RANK_BOT_BASE", Path(__file__).resolve().parents[1])
        )

        return cls(
            cursor_model=os.environ.get("CURSOR_MODEL", "sonnet-4.6"),
            cursor_api_key=os.environ.get("CURSOR_API_KEY"),
            repo_c4_path=base / "Submissions-C4",
            repo_c3_path=base / "Submissions_C3",
            c3_csv_path=base
            / "sheets"
            / "Outskill Eng Accelerator Score Card - C3.csv",
            c4_csv_path=base
            / "sheets"
            / "Outskill Eng Accelerator Score Card - C4.csv",
            syllabus_csv_path=base
            / "sheets"
            / "Engineering Accelerator Program - Schedule + Roadmap.csv",
        )
