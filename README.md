# SQL Agent Web Application
A FastAPI web application that leverages a Large Language Model (LLM) to interact with a SQL database using natural language queries. The application supports both direct querying and Retrieval-Augmented Generation (RAG) for enhanced responses.

## Demo
### Asking simple questions:
![Web page](images\Demo.png)

### Step-by-step reasoning:
![Reasoning timeline](images\Step.png)

### Follow-up questions:
![Follow-up questions](images\Followup.png)

## Project Structure

```
sqlagent_web/
├── app/
│   ├── __init__.py         # Module initialization (exports app)
│   ├── config.py           # Configuration management with Pydantic Settings
│   ├── main.py             # FastAPI app + CORS configuration
│   ├── routes.py           # API endpoints (query, stream, static)
│   ├── agent_service.py    # SQLAgentService class
│   ├── parsers.py          # Data extraction & transformation
│   └── debug_utils.py      # Agent reasoning timeline display
├── data/                   # Generated files (schema descriptions, FAISS index)
├── scripts/
│   ├── init_schema.py      # Generate schema descriptions using LLM
│   └── build_index.py      # Build FAISS vector index for RAG
├── .env                    # Environment variables (not in version control)
├── .gitignore              # Git ignore rules (includes .env)
├── app.py                  # Entry point (imports from app.main)
├── index.html              # Frontend SPA (no build process required)
├── Chinook.db              # SQLite sample database (read-only)
├── pyproject.toml          # Python 3.12 + dependencies (includes pydantic-settings)
└── uv.lock                 # Locked dependency versions
```

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager
- AWS Bedrock access with Claude Sonnet 4.5

### Installation

```bash
# Install dependencies
uv sync

# Initialize schema descriptions (required for RAG)
uv run python scripts/init_schema.py

# Build FAISS vector index (required for RAG)
uv run python scripts/build_index.py
```

### Run

```bash
# Development mode (auto-reload)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python app.py
```

Access at: http://localhost:8000

## Usage Examples

### Simple Queries
```
Show me all artists
List customers from USA
```

### Complex Queries
```
What are the top 5 albums by total sales?
Which tracks are longer than 5 minutes?
Show total sales by country
```

## Configuration

All settings via `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_URI` | Database connection string | `sqlite:///./Chinook.db` |
| `MODEL_NAME` | LLM model identifier | `bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `TOP_K` | Default result limit | `20` |
| `RAG_TOP_K` | RAG retrieval limit | `5` |
| `RECURSION_LIMIT` | Max agent iterations | `15` |

