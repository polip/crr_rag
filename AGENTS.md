# AGENTS.md

## Critical Gotchas

- **Env var names are non-standard**: `OPENAI_EMBEDD_KEY` (embeddings) and `OPENAI_CHAT_KEY` (chat). NOT `OPENAI_API_KEY`.
- **Two dependency lists diverge**: `requirements.txt` (used by Dockerfiles) is leaner than `pyproject.toml`. Docker builds use `requirements.txt`; local dev with `uv` uses `pyproject.toml`.
- **`docker-compose.yml` is stale**: references `NVIDIA_API_KEY` and `GEMINI_API_KEY` which no longer exist in code. Use `uv run streamlit run st_crr.py` for local dev instead.
- **`main.py` is empty** — not an entrypoint.
- **No test suite exists** — no pytest config, no test files.

## Developer Commands

```bash
uv sync                        # Install deps (uses pyproject.toml)
uv run streamlit run st_crr.py # Run the app locally
uv run process_document.py <pdf> <doc_id> <doc_name>          # Process + upload
uv run process_document.py <pdf> <doc_id> <doc_name> --pickle-only  # Safer for large docs
uv run upload_from_pickle.py <pickle_path> [--skip-validation]
uv run check_collection.py     # List documents/chunks in AstraDB
uv run delete_collection_data.py list|<doc_id>|all  # Delete documents
```

## Architecture

Single Python app, three entrypoints:
- `st_crr.py` — Streamlit UI (imports `DocumentRouter` from `document_router.py`)
- `document_router.py` — Core RAG logic, runnable standalone for testing
- `process_document.py` — PDF processing pipeline, runnable standalone

Data flow: PDF → PyMuPDF extraction → article-boundary chunking → AstraDB vector store → OpenAI embeddings → OpenAI LLM response.

## Chunking Pipeline

Oversized chunks use a **two-stage split**:
1. Split at internal article boundaries first (preserves metadata)
2. Fall back to `RecursiveCharacterTextSplitter` if still oversized

Safety threshold: **870 tokens** (85% of 1024 max). Token estimation: `len(text) / 4`.

## Deployment

- **CI**: Push to `main` → GitHub Actions → build `Dockerfile.production` → deploy to Google Cloud Run (`europe-west12`)
- **Production Dockerfile** only copies `st_crr.py` and `document_router.py` — processing scripts are NOT included in the production image
- Cloud Run config: 2Gi memory, 2 CPU, 300s timeout, min 0 instances

## File Conventions

- `pdfs/` and `pickle_files/` are gitignored — source PDFs and backups are not in the repo
- `.env` is required but gitignored; see README.md for template
- Streamlit caches `DocumentRouter` via `@st.cache_resource` and doc stats via `@st.cache_data(ttl=3600)`

## Env Vars Required

| Variable | Purpose |
|---|---|
| `OPENAI_EMBEDD_KEY` | OpenAI API key for embeddings (`text-embedding-3-small`) |
| `OPENAI_CHAT_KEY` | OpenAI API key for chat (`gpt-4o-mini` default) |
| `ASTRA_DB_TOKEN` | AstraDB authentication token |
| `ASTRA_DB_API_ENDPOINT` | AstraDB API endpoint URL |
| `ASTRA_DB_COLLECTION_NAME` | AstraDB collection name |
| `OPENAI_MODEL_NAME` | Optional, default: `gpt-4o-mini` |
