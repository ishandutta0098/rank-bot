# 🤖 Rank-Bot: AI-Powered Hackathon Judge

An intelligent agent system powered by [Cursor CLI](https://cursor.com/docs/cli/headless) that automatically evaluates and ranks hackathon submissions based on concept usage, implementation difficulty, and code quality.

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
- ✅ **Cursor CLI headless mode** (`agent -p`) for all LLM calls
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
        │  Prompt  │  │  Prompt  │  │  Prompt  │
        └──────────┘  └──────────┘  └──────────┘
             │             │             │
             └─────────────┼─────────────┘
                           │
                    ┌──────▼──────┐
                    │ Cursor CLI  │
                    │  agent -p   │
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
        ┌─────────────┐       ┌─────────────────┐
        │  Shell tool │       │  File read tool  │
        │ (git, unzip)│       │  (local files)   │
        └─────────────┘       └─────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| **`main.py`** | Orchestrates the 3-phase evaluation pipeline |
| **`cursor_runner.py`** | Wraps Cursor CLI headless invocations; validates Pydantic outputs |
| **`prompts.py`** | Contains detailed instructions for each judge |
| **`models.py`** | Pydantic models for structured judge outputs |
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
- **Cursor CLI** installed and authenticated ([installation guide](https://cursor.com/docs/cli/installation))
- **Git repositories**: Cloned C3 and C4 submission repos

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rank-bot
   ```

2. **Install Cursor CLI** (if not already installed)
   ```bash
   curl https://cursor.com/install -fsS | bash
   # Then authenticate
   agent login
   ```

3. **Install Python dependencies**
   ```bash
   uv sync
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   # Optional — Cursor CLI uses browser login by default
   # Set only if authenticating via API key (CI/CD, automation)
   CURSOR_API_KEY=your_cursor_api_key_here

   # Optional (defaults shown)
   CURSOR_MODEL=sonnet-4.6
   RANK_BOT_BASE=/path/to/rank-bot  # Auto-detected if not set
   ```

5. **Verify setup**
   ```bash
   agent status          # confirm Cursor CLI is authenticated
   uv run python -c "import sys; sys.path.append('src'); from config import Config; print('Config loaded')"
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
| `CURSOR_MODEL` | Cursor model to use for evaluation | `sonnet-4.6` |
| `CURSOR_API_KEY` | API key for Cursor CLI (optional; browser login used otherwise) | - |
| `RANK_BOT_BASE` | Base directory for the project | Auto-detected |

### Model Selection

The model is passed directly to Cursor CLI via `--model`. Use `agent models` to list models available on your account.

Change model in `.env`:
```bash
CURSOR_MODEL=sonnet-4.6
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
├── cursor_runner.py     # Cursor CLI subprocess wrapper + Pydantic validation
├── prompts.py           # Judge instructions (shell-command oriented)
├── models.py            # Pydantic schemas
├── scoring.py           # CSV/report utilities
└── config.py            # Configuration management
```

### Key Design Patterns

1. **Functional Core, Imperative Shell**: Pure functions in `scoring.py`, side effects in `main.py`
2. **Errors as Values**: No nested try-except blocks; graceful degradation
3. **Frozen Dataclasses**: Immutable domain models (`GroupInfo`, `Config`)
4. **Pydantic for I/O**: Structured outputs validated from Cursor CLI's final JSON response

### How Cursor CLI Is Invoked

Each judge phase runs Cursor CLI in headless mode and pipes the prompt via stdin:

```python
proc = await asyncio.create_subprocess_exec(
    "agent", "-p", "--force", "--trust",
    "--model", config.cursor_model,
    "--output-format", "json",
    "--workspace", str(workspace),
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
stdout, _ = await proc.communicate(input=prompt.encode())
result_text = json.loads(stdout.decode())["result"]
```

The CLI's built-in shell and file tools handle all git inspection — no custom tool layer needed.

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

**Issue**: `Cursor CLI failed with code 1` / `Not authenticated`
```bash
# Solution: Log in via browser
agent login
# Or set an API key from https://cursor.com/dashboard/integrations
echo "CURSOR_API_KEY=your_key_here" > .env
```

**Issue**: `agent: command not found`
```bash
# Solution: Install Cursor CLI and add it to PATH
curl https://cursor.com/install -fsS | bash
# Then follow the PATH setup instructions printed by the installer
```

**Issue**: `RuntimeError: Cursor CLI returned an error result`
```bash
# The CLI ran but the agent reported an error — check stderr output.
# Resume with --groups flag to retry only failed groups:
uv run python src/main.py --groups 10 11 12
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

- Powered by [Cursor CLI](https://cursor.com/docs/cli/headless) for headless agent execution
- Uses [uv](https://github.com/astral-sh/uv) for fast Python package management

---

**Made with ❤️ for the AI Accelerator Hackathon**
