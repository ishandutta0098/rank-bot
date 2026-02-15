# 🤖 Rank-Bot: AI-Powered Hackathon Judge

An intelligent agent system built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) that automatically evaluates and ranks hackathon submissions based on concept usage, implementation difficulty, and code quality.

## 📋 Overview

Rank-Bot was designed to judge hackathon projects from an AI Accelerator program that teaches:
- Prompt Engineering
- Multimodal AI
- RAG (Retrieval Augmented Generation)
- AI Agents (LangChain, LangGraph)

The system uses a **multi-agent architecture** with three specialized judge agents that evaluate submissions on:

1. **Concept Score (1-10)**: How many advanced concepts from the syllabus were used
2. **Difficulty Level (1-10)**: Relative implementation complexity compared to other submissions
3. **Code Quality (1-10)**: Project structure, documentation, and organization

### Key Features

- ✅ **Non-destructive Git inspection** using `git ls-tree` and `git show`
- ✅ **Handles multiple submission formats**: branches, commits, zip files
- ✅ **Calibrated scoring** using historical reference data
- ✅ **OpenRouter API** for cost-effective LLM access
- ✅ **Pydantic structured outputs** for reliable JSON parsing
- ✅ **Robust error handling** with partial run support
- ✅ **Detailed evaluation reports** in Markdown and JSON

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Orchestrator                        │
│                      (src/main.py)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Concept  │  │Difficulty│  │  Code    │
        │  Judge   │  │  Judge   │  │ Quality  │
        │  Agent   │  │  Agent   │  │  Judge   │
        └──────────┘  └──────────┘  └──────────┘
             │             │             │
             └─────────────┼─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Tools     │
                    │ (Git/FS)    │
                    └─────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| **`main.py`** | Orchestrates the 3-phase evaluation pipeline |
| **`agents_factory.py`** | Creates specialized judge agents with OpenRouter compatibility patches |
| **`prompts.py`** | Contains detailed instructions for each judge agent |
| **`tools.py`** | Git and filesystem tools for non-destructive code inspection |
| **`models.py`** | Pydantic models for structured LLM outputs |
| **`scoring.py`** | CSV parsing, URL parsing, report generation |
| **`config.py`** | Environment-based configuration management |

### Evaluation Pipeline

```
Phase 1: Collect Project Summaries
  ├─ Parse C4 scorecard CSV
  ├─ Extract branch/commit/path from GitHub URLs
  └─ Generate summaries for difficulty calibration

Phase 2: Per-Project Scoring
  ├─ Concept Judge: Evaluate syllabus concept usage
  └─ Code Quality Judge: Assess structure and docs

Phase 3: Relative Difficulty Scoring
  └─ Difficulty Judge: Compare all projects and rank

Phase 4: Generate Outputs
  ├─ Update CSV with scores and rankings
  ├─ Generate detailed Markdown report
  └─ Export structured JSON results
```

## 🚀 Setup

### Prerequisites

- **Python 3.12+**
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **OpenRouter API key** ([get one here](https://openrouter.ai/))
- **Git repositories**: Cloned C3 and C4 submission repos

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rank-bot
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # Required
   OPENROUTER_API_KEY=your_api_key_here
   
   # Optional (defaults shown)
   RANK_BOT_MODEL=anthropic/claude-sonnet-4
   RANK_BOT_BASE=/path/to/rank-bot  # Auto-detected if not set
   ```

4. **Verify setup**
   ```bash
   uv run python -c "from src.config import Config; print('✅ Config loaded')"
   ```

### Directory Structure

Ensure your project has the following structure:

```
rank-bot/
├── src/                    # Source code
├── sheets/                 # CSV files
│   ├── reference_scores.csv
│   ├── current_scores.csv
│   └── syllabus.csv
├── submissions_reference/  # Reference submission repo
├── submissions_current/    # Current submission repo
├── output/                 # Generated reports (auto-created)
├── .env                    # Your API keys (gitignored)
├── pyproject.toml          # Project config
└── README.md
```

## 💻 Usage

### Basic Evaluation

Evaluate all submissions:

```bash
uv run python src/main.py
```

### Partial Runs

Resume evaluation for specific groups (useful after API errors):

```bash
# Evaluate only groups 3, 4, and 5
uv run python src/main.py --groups 3 4 5
```

### Output Files

After running, you'll find:

- **`sheets/current_scores.csv`**: Updated with scores and rankings
- **`output/evaluation_report.md`**: Detailed Markdown report with justifications
- **`output/scores.json`**: Structured JSON data for programmatic access

### Example Output

```markdown
# Hackathon Evaluation Report

## Summary

| Rank | Group | Concept | Difficulty | Code Quality | Total |
|------|-------|---------|------------|--------------|-------|
| 1    | 12    | 10      | 10         | 9            | 29    |
| 2    | 5     | 9       | 9          | 10           | 28    |
| 3    | 8     | 10      | 10         | 8            | 28    |
...
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | **Required**. Your OpenRouter API key | - |
| `RANK_BOT_MODEL` | Model to use for evaluation | `anthropic/claude-sonnet-4` |
| `RANK_BOT_BASE` | Base directory for the project | Auto-detected |

### Model Selection

Recommended models for cost/quality balance:

- **Best quality**: `anthropic/claude-sonnet-4` (default)
- **Budget-friendly**: `google/gemini-2.0-flash-001`
- **Balanced**: `openai/gpt-4o-mini`

Change model in `.env`:
```bash
RANK_BOT_MODEL=google/gemini-2.0-flash-001
```

### Scoring Calibration

The system uses historical reference scores for calibration. Key calibration points:

- **10/10 Concept**: Multi-agent LangGraph + RAG + multimodal + advanced orchestration
- **9/10 Concept**: LangGraph + RAG + external integrations
- **8/10 Concept**: Multi-agent with RAG but fewer concept areas
- **7/10 Concept**: Multi-agent with some RAG or LangGraph
- **≤6/10 Concept**: Simpler implementations or minimal concept usage

## 🛠️ Development

### Project Structure

```
src/
├── main.py              # Entry point and orchestration
├── agents_factory.py    # Agent creation with SDK patches
├── prompts.py           # Judge agent instructions
├── tools.py             # Git/filesystem tools
├── models.py            # Pydantic schemas
├── scoring.py           # CSV/report utilities
└── config.py            # Configuration management
```

### Key Design Patterns

1. **Functional Core, Imperative Shell**: Pure functions in `scoring.py`, side effects in `main.py`
2. **Errors as Values**: No nested try-except blocks; graceful degradation
3. **Frozen Dataclasses**: Immutable domain models (`GroupInfo`, `Config`)
4. **Pydantic for I/O**: Structured LLM outputs with validation
5. **Monkey Patches**: SDK compatibility fixes for OpenRouter

### SDK Compatibility Patches

The system includes two critical patches in `agents_factory.py`:

1. **Response Format Patch**: Forces `json_object` mode (OpenRouter doesn't support `json_schema`)
2. **JSON Extraction Patch**: Strips preamble text that Claude adds before JSON

### Adding New Tools

Tools are defined in `tools.py` using the `@function_tool` decorator:

```python
from openai_agents import function_tool

@function_tool
def my_new_tool(param: str) -> str:
    """Tool description for the LLM.
    
    Args:
        param: Parameter description
        
    Returns:
        str: Result description
    """
    # Implementation
    return result
```

Register in `tools.py`:
```python
ALL_TOOLS = [
    git_list_files,
    git_read_file,
    read_local_file,
    list_local_directory,
    extract_zip_and_list,
    my_new_tool,  # Add here
]
```

## 📊 Evaluation Criteria

### Concept Score (1-10)

Evaluates usage of accelerator concepts:

- ✅ Prompt Engineering (XML/JSON structured prompts)
- ✅ LLM APIs (OpenAI, Groq, OpenRouter)
- ✅ RAG (Vector stores, embeddings, LlamaIndex/LangChain)
- ✅ Multimodal AI (Image/audio/video processing)
- ✅ AI Agents (LangChain tools, structured output)
- ✅ Advanced Agents (LangGraph orchestration, conditional edges, loops)

**Scoring**: More concepts + sophisticated combinations = higher score

### Difficulty Level (1-10)

Relative assessment of implementation complexity:

- Graph orchestration patterns (conditional edges, loops, fan-out/fan-in)
- Multi-source RAG with parallel retrieval
- Custom vector store implementations
- External API integrations
- Hallucination detection and quality gates
- Domain-specific validation logic

**Scoring**: Comparative ranking across all submissions

### Code Quality (1-10)

Evaluates project organization:

- ✅ Proper folder structure (`agents/`, `utils/`, `tests/`, `docs/`)
- ✅ Comprehensive README with setup instructions
- ✅ `requirements.txt` or `pyproject.toml`
- ✅ `.env.example` and `.gitignore`
- ✅ No committed secrets, cache files, or binaries
- ✅ Architecture documentation

**Penalties**: ZIP submissions, committed secrets, missing docs

## 🐛 Troubleshooting

### Common Issues

**Issue**: `OPENROUTER_API_KEY environment variable must be set`
```bash
# Solution: Create .env file with your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

**Issue**: `ModuleNotFoundError: No module named 'agents'`
```bash
# Solution: Install dependencies
uv sync
```

**Issue**: `APIStatusError: Error code: 402 - insufficient credits`
```bash
# Solutions:
# 1. Top up credits at https://openrouter.ai/settings/keys
# 2. Switch to a cheaper model in .env:
echo "RANK_BOT_MODEL=google/gemini-2.0-flash-001" >> .env
# 3. Resume with --groups flag to skip already-scored groups
uv run python src/main.py --groups 10 11 12
```

**Issue**: `MaxTurnsExceeded: Max turns (20) exceeded`
```bash
# This means the agent is reading too many files
# The system has efficiency guidance built-in, but you can:
# 1. Increase max_turns in main.py (already set to 20)
# 2. Check if the project has excessive files
```

### Debug Mode

Enable detailed logging:

```bash
# In your .env
LOG_LEVEL=DEBUG

# Or inline
LOG_LEVEL=DEBUG uv run python src/main.py
```

## 📝 License

This project is for educational and evaluation purposes as part of the AI Accelerator program.

## 🙏 Acknowledgments

- Built with [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- Powered by [OpenRouter](https://openrouter.ai/) for multi-model access
- Uses [uv](https://github.com/astral-sh/uv) for fast Python package management

---

**Made with ❤️ for the AI Accelerator Hackathon**
