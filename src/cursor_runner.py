"""Cursor CLI subprocess runner for rank-bot judge calls.

All LLM-backed work is delegated to Cursor CLI headless mode. The rest of the
pipeline treats the CLI result as text and validates structured judge outputs
with Pydantic.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """Extract a JSON object or array from text that may have preamble/postamble.

    Args:
        text: Raw model response text.

    Returns:
        Cleaned string containing only the JSON portion.
    """
    stripped = text.strip()

    if stripped.startswith(("{", "[")):
        return stripped

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    first_brace = None
    for i, ch in enumerate(stripped):
        if ch in "{[":
            first_brace = i
            break

    if first_brace is not None:
        candidate = stripped[first_brace:]
        for j in range(len(candidate) - 1, -1, -1):
            if candidate[j] in "}]":
                return candidate[: j + 1]

    return stripped


def _schema_suffix(model_cls: type[BaseModel]) -> str:
    """Generate a JSON output instruction block from a Pydantic model's schema.

    Args:
        model_cls: Pydantic model class for the expected judge output.

    Returns:
        A formatted instruction string with the JSON schema.
    """
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "\n\n## CRITICAL: Output Format\n\n"
        "When you have finished using tools and are ready to give your final "
        "answer, you MUST respond with ONLY a valid JSON object matching this "
        "schema -- no markdown, no explanation text, no code fences:\n\n"
        f"```\n{schema}\n```\n\n"
        "Do NOT include any text before or after the JSON object."
    )


async def run_cursor_agent(
    prompt: str,
    workspace: Path,
    model: str,
    api_key: str | None = None,
) -> str:
    """Run Cursor CLI in headless mode and return its final text result.

    Args:
        prompt: Complete prompt to send to Cursor CLI.
        workspace: Workspace path passed to the CLI.
        model: Cursor model identifier, for example 'sonnet-4.6'.
        api_key: Optional Cursor API key. If omitted, CLI auth or env auth is used.

    Returns:
        The final assistant text from the CLI JSON result.

    Raises:
        RuntimeError: If the CLI exits non-zero or returns an error result.
        json.JSONDecodeError: If the CLI stdout is not valid JSON.
        KeyError: If the successful CLI JSON result is missing the result field.
    """
    command = [
        "agent",
        "-p",
        "--force",
        "--trust",
        "--model",
        model,
        "--output-format",
        "json",
        "--workspace",
        str(workspace),
    ]
    if api_key:
        command.extend(["--api-key", api_key])

    log.info("Running Cursor CLI: workspace=%s model=%s", workspace, model)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=prompt.encode())

    stderr_text = stderr.decode(errors="replace").strip()
    stdout_text = stdout.decode(errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cursor CLI failed with code {proc.returncode}: {stderr_text}"
        )

    payload = json.loads(stdout_text)
    if payload.get("is_error"):
        raise RuntimeError(f"Cursor CLI returned an error result: {payload}")

    return payload["result"]


async def run_structured_agent(
    prompt: str,
    workspace: Path,
    model: str,
    output_type: type[T],
    api_key: str | None = None,
) -> T:
    """Run Cursor CLI and validate its final text as a Pydantic model.

    Args:
        prompt: Complete prompt to send to Cursor CLI.
        workspace: Workspace path passed to the CLI.
        model: Cursor model identifier, for example 'sonnet-4.6'.
        output_type: Pydantic model class expected from the response.
        api_key: Optional Cursor API key. If omitted, CLI auth or env auth is used.

    Returns:
        The validated Pydantic model instance.

    Raises:
        RuntimeError: If the CLI exits non-zero or returns an error result.
        json.JSONDecodeError: If either the CLI result or extracted response JSON is invalid.
        pydantic.ValidationError: If the extracted JSON does not match output_type.
        KeyError: If the successful CLI JSON result is missing the result field.
    """
    result = await run_cursor_agent(
        prompt=prompt,
        workspace=workspace,
        model=model,
        api_key=api_key,
    )
    cleaned = _extract_json(result)
    return output_type.model_validate_json(cleaned)
