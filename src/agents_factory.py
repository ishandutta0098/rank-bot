"""Agent factory — creates the three judge agents backed by OpenRouter.

Uses the OpenAI Agents SDK with an ``AsyncOpenAI`` client pointed at
OpenRouter's ``/api/v1`` endpoint.

Two SDK patches for OpenRouter compatibility:

1. **Response format**: downgrades ``json_schema`` (OpenAI-only) to
   ``json_object`` which OpenRouter supports for all major models.
2. **JSON extraction**: the SDK's ``validate_json`` is monkey-patched so
   that if the model returns preamble text before the JSON object, the
   JSON is extracted before Pydantic validation.  This handles Claude's
   tendency to wrap JSON in explanatory text.

The Pydantic ``output_type`` on each agent still validates the response;
the JSON schema is also injected into agent instructions so the model
knows what structure to produce.
"""

import json
import logging
import re

from agents import Agent, ModelSettings, set_tracing_disabled
from agents.models.chatcmpl_converter import Converter
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.util import _json as _agents_json
from openai import AsyncOpenAI
from pydantic import BaseModel, TypeAdapter

from config import Config
from models import AllDifficultyScores, CodeQualityResult, ConceptScoreResult
from prompts import (
    build_code_quality_judge_instructions,
    build_concept_judge_instructions,
    build_difficulty_judge_instructions,
)
from tools import ALL_TOOLS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: downgrade json_schema → json_object for OpenRouter compatibility
# ---------------------------------------------------------------------------

_original_convert_response_format = Converter.convert_response_format.__func__


@classmethod  # type: ignore[misc]
def _openrouter_convert_response_format(cls, output_schema):  # noqa: ANN001, ANN201
    """Convert output schema to ``json_object`` instead of ``json_schema``.

    Args:
        cls: The Converter class.
        output_schema: The agent output schema, or None.

    Returns:
        A response format dict or Omit sentinel.
    """
    original = _original_convert_response_format(cls, output_schema)
    if isinstance(original, dict) and original.get("type") == "json_schema":
        return {"type": "json_object"}
    return original


Converter.convert_response_format = _openrouter_convert_response_format


# ---------------------------------------------------------------------------
# Patch 2: extract JSON from model responses with preamble text
# ---------------------------------------------------------------------------

_original_validate_json = _agents_json.validate_json


def _extract_json(text: str) -> str:
    """Extract a JSON object or array from text that may have preamble/postamble.

    Tries the raw text first (fast path), then progressively strips non-JSON
    content.

    Args:
        text: Raw model response text.

    Returns:
        Cleaned string containing only the JSON portion.
    """
    stripped = text.strip()

    # Fast path: already starts with { or [
    if stripped.startswith(("{", "[")):
        return stripped

    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Find the first { or [ and take everything from there
    first_brace = None
    for i, ch in enumerate(stripped):
        if ch in "{[":
            first_brace = i
            break

    if first_brace is not None:
        candidate = stripped[first_brace:]
        # Find the matching closing brace from the end
        for j in range(len(candidate) - 1, -1, -1):
            if candidate[j] in "}]":
                return candidate[: j + 1]

    return stripped


def _patched_validate_json(json_str: str, type_adapter: TypeAdapter, partial: bool):  # noqa: ANN001, ANN201
    """Validate JSON with automatic extraction of JSON from preamble text.

    Args:
        json_str: Raw model response (may contain preamble).
        type_adapter: Pydantic TypeAdapter for validation.
        partial: Whether to allow partial JSON.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ModelBehaviorError: If JSON is invalid even after extraction.
    """
    cleaned = _extract_json(json_str)
    return _original_validate_json(cleaned, type_adapter, partial)


_agents_json.validate_json = _patched_validate_json


# ---------------------------------------------------------------------------
# Schema suffix generator — derives JSON instructions from the Pydantic model
# ---------------------------------------------------------------------------


def _schema_suffix(model_cls: type[BaseModel]) -> str:
    """Generate a JSON output instruction block from a Pydantic model's schema.

    This is appended to agent instructions so the model knows the exact JSON
    structure expected.  The Pydantic ``output_type`` on the Agent validates
    the response after it is received.

    Args:
        model_cls: The Pydantic model class used as the agent's output_type.

    Returns:
        A formatted instruction string with the JSON schema.
    """
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "\n\n## CRITICAL: Output Format\n\n"
        "When you have finished using tools and are ready to give your final "
        "answer, you MUST respond with ONLY a valid JSON object matching this "
        "schema — no markdown, no explanation text, no code fences:\n\n"
        f"```\n{schema}\n```\n\n"
        "Do NOT include any text before or after the JSON object."
    )


# ---------------------------------------------------------------------------
# Model and agent creation
# ---------------------------------------------------------------------------


def create_model(config: Config) -> OpenAIChatCompletionsModel:
    """Create an OpenRouter-backed chat completions model.

    Args:
        config: Application configuration with API key and model name.

    Returns:
        OpenAIChatCompletionsModel wrapping the AsyncOpenAI client.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key,
    )
    return OpenAIChatCompletionsModel(
        model=config.model_name,
        openai_client=client,
    )


def create_agents(
    config: Config,
    syllabus: str,
    c3_ref: str,
) -> tuple[Agent, Agent, Agent]:
    """Create the three judge agents: concept, code quality, and difficulty.

    Each agent gets:
    - Domain-specific instructions from ``prompts.py``
    - A Pydantic-derived JSON schema suffix appended to its instructions
    - A Pydantic ``output_type`` for response validation

    Args:
        config: Application configuration.
        syllabus: Formatted syllabus text for the concept judge.
        c3_ref: Formatted C3 reference scores table for calibration.

    Returns:
        Tuple of (concept_judge, code_quality_judge, difficulty_judge).
    """
    # Disable OpenAI tracing (we're using OpenRouter, not OpenAI)
    set_tracing_disabled(True)

    model = create_model(config)

    # Cap max_tokens to avoid credit exhaustion on OpenRouter
    settings = ModelSettings(max_tokens=4096)

    concept_judge = Agent(
        name="ConceptJudge",
        model=model,
        model_settings=settings,
        output_type=ConceptScoreResult,
        instructions=(
            build_concept_judge_instructions(syllabus, c3_ref)
            + _schema_suffix(ConceptScoreResult)
        ),
        tools=ALL_TOOLS,
    )

    code_quality_judge = Agent(
        name="CodeQualityJudge",
        model=model,
        model_settings=settings,
        output_type=CodeQualityResult,
        instructions=(
            build_code_quality_judge_instructions(c3_ref)
            + _schema_suffix(CodeQualityResult)
        ),
        tools=ALL_TOOLS,
    )

    # Difficulty judge gets no tools — it receives pre-collected summaries
    difficulty_judge = Agent(
        name="DifficultyJudge",
        model=model,
        model_settings=settings,
        output_type=AllDifficultyScores,
        instructions=(
            build_difficulty_judge_instructions(c3_ref)
            + _schema_suffix(AllDifficultyScores)
        ),
    )

    log.info(
        "Created agents: model=%s, tools=%d",
        config.model_name,
        len(ALL_TOOLS),
    )

    return concept_judge, code_quality_judge, difficulty_judge
