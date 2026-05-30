# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Document Legal RAG (Retrieval-Augmented Generation) system for querying financial regulation documents (CRR, CRD) with intelligent document routing. Built with LangChain, AstraDB vector store, and OpenAI.

## Core Architecture

**Three-layer architecture:**

1. **Document Processing Layer** ([process_document.py](process_document.py))
   - Extracts articles from PDF regulation documents using PyMuPDF (fitz)
   - Uses regex pattern matching for article boundaries: `^(Članak|Article)\s+(\d+)`
   - Implements automatic chunk splitting with 870 token safety threshold (85% of 1024 token limit)
   - Splits oversized chunks at article boundaries first, then falls back to recursive character splitting
   - Searches entire chunk content for article re-detection (not just first 300 chars)
   - Saves backup to pickle files before uploading to vector store

2. **Document Routing Layer** ([document_router.py](document_router.py))
   - AI-powered routing that analyzes queries to select relevant documents
   - Uses OpenAI LLM for routing decisions
   - Supports three modes: auto-route, all documents, or specific document selection
   - Implements metadata filtering by `document_id` for targeted retrieval
   - Detects article number references in queries and filters by `article_number` metadata

3. **UI Layer** ([st_crr.py](st_crr.py))
   - Streamlit-based chat interface with persistent conversation history
   - Caches RAG system and document statistics (TTL: 1 hour)
   - Displays source citations with document name, article number, and page

**Data Flow:**
```
PDF → article extraction → token validation → chunk splitting → AstraDB (with OpenAI embeddings)
User query → document routing → vector search → context retrieval → OpenAI LLM → response
```

## Development Commands

### Environment Setup
```bash
# Using UV (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

### Running the Application
```bash
# Local development
uv run streamlit run st_crr.py

# Docker
docker compose up -d --build
docker compose logs -f crr-rag-app
docker compose down
```

### Document Processing

**Process and upload (default workflow):**
```bash
uv run process_document.py <pdf_path> <document_id> <document_name>
# Example: uv run process_document.py pdfs/regulation.pdf CRR 'CRR'
```

**Process to pickle only (safer for large docs):**
```bash
uv run process_document.py <pdf_path> <document_id> <document_name> --pickle-only
```

**Upload from pickle file:**
```bash
uv run upload_from_pickle.py <pickle_file_path>
# With validation disabled: add --skip-validation flag
```

**Check collection:**
```bash
uv run check_collection.py
```

**Delete collection data:**
```bash
# List all document IDs in collection
uv run delete_collection_data.py list

# Delete specific document by ID
uv run delete_collection_data.py CRR

# Delete ALL data (requires confirmation)
uv run delete_collection_data.py all
```

### Testing Document Router
```bash
uv run document_router.py
```

## Key Implementation Details

### Token Management
- **Max tokens:** 1024 (OpenAI embedding model limit)
- **Safety threshold:** 870 tokens (85% of max)
- Automatic splitting when chunks exceed safety threshold
- Token estimation: 1 token ≈ 4 characters

### Article Detection
- Primary pattern: `^(Članak|Article)\s+(\d+)` (case-insensitive, multiline)
- Supports bilingual documents (Croatian "Članak" and English "Article")
- Re-detection performed after chunk splitting to maintain accurate article numbers
- Oversized chunks split at article boundaries first, then recursive splitting if needed
- Article number detection in user queries enables targeted retrieval

### Metadata Structure
Each chunk includes:
- `document_id`: Unique identifier for document
- `document_name`: Human-readable document name
- `article_no`: Full article reference (e.g., "Article 123")
- `article_number`: Numeric article number (int)
- `page`: Page number in source PDF
- `language`: "hr" or "en"
- `type`: "article"
- Optional: `sub_chunk`, `total_sub_chunks`, `original_chunk_tokens`, `article_redetected`

### Vector Store Configuration
- **Embeddings:** OpenAI `text-embedding-3-small` model
- **Batch size:** 25 chunks per upload
- **Retrieval:** Top-k=6-8 chunks per query with optional document filtering
- **Collection:** Single AstraDB collection with multi-document support

### Deployment
- **GitHub Actions:** Auto-deploys to Google Cloud Run on push to main
- **Production Dockerfile:** [Dockerfile.production](Dockerfile.production)
- **Environment variables required:**
  - `OPENAI_EMBEDD_KEY` (for embeddings)
  - `OPENAI_CHAT_KEY` (for chat/completions)
  - `ASTRA_DB_TOKEN`
  - `ASTRA_DB_API_ENDPOINT`
  - `ASTRA_DB_COLLECTION_NAME`
  - `OPENAI_MODEL_NAME` (optional, default: gpt-4o-mini)

## Project Structure
```
├── st_crr.py                 # Streamlit UI with chat interface
├── document_router.py        # AI-powered document routing logic
├── process_document.py       # PDF processing & chunking pipeline
├── upload_from_pickle.py     # Pickle-to-database upload utility
├── check_collection.py       # Database statistics viewer
├── delete_collection_data.py # Delete documents from collection
├── list_collections.py       # List all collections in database
├── requirements.txt          # Pip dependencies
├── pyproject.toml           # UV project config
├── Dockerfile                # Local development container
├── Dockerfile.production     # Production deployment container
├── docker-compose.yml       # Local Docker Compose setup
├── .github/workflows/        # CI/CD configuration
├── pdfs/                    # Source PDF documents
└── pickle_files/            # Processed chunk backups
```

## Important Notes

- Always use `--pickle-only` when processing large documents for the first time (safer workflow)
- Pickle files serve as backups and enable re-upload without reprocessing
- The system maintains article number accuracy through re-detection after splitting
- Document routing uses cached document list (fetches 100 samples at init)
- Streamlit caches RAG system initialization and document statistics (1 hour TTL)
- All chunks are validated against token limits before upload (unless `--skip-validation`)
- Uses OpenAI for both embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
- Article boundary splitting prioritizes semantic boundaries before falling back to character-based splitting
