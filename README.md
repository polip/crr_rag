# 📚 Multi-Document Legal RAG System

A Retrieval-Augmented Generation (RAG) system for querying financial regulation documents (CRR, CRD) with intelligent document routing and multi-document support.

## 🎯 Features

- **Multi-Document Support**: Query across multiple regulation documents (CRR, CRD, etc.)
- **Intelligent Document Routing**: AI automatically selects relevant documents for each query
- **Article-Level Chunking**: Extracts and processes documents by article boundaries
- **Smart Token Management**: Automatic chunk splitting with 435-token safety threshold
- **Flexible Processing**: Process to pickle only or directly upload to vector database
- **Vector Storage**: Powered by AstraDB for efficient similarity search
- **Modern UI**: Streamlit-based chat interface
- **Docker Support**: Easy deployment with Docker Compose
- **Pickle Backup**: Processed chunks saved for backup and re-upload

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────────────┐
│  Document Router        │
│  (AI-powered routing)   │
└────────┬────────────────┘
         │
┌────────▼────────────────┐
│  AstraDB Vector Store   │
│  (NVIDIA Embeddings)    │
└────────┬────────────────┘
         │
┌────────▼────────────────┐
│  Google Gemini LLM      │
│  (Response Generation)  │
└─────────────────────────┘
```

## 🛠️ Tech Stack

- **PDF Processing**: pypdf (lightweight text extraction)
- **Embeddings**: NVIDIA AI Endpoints (nv-embedqa-e5-v5)
- **Vector Store**: Astra DB (Cassandra-based)
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain
- **UI**: Streamlit
- **Infrastructure**: Docker, Docker Compose

## 📋 Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- Docker & Docker Compose (for containerized deployment)
- API Keys:
  - NVIDIA API Key (for embeddings)
  - Google Gemini API Key (for LLM)
  - Astra DB credentials (token, endpoint, collection name)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd crr_rag
```

### 2. Configure Environment

Create a `.env` file:

```env
# NVIDIA API (Embeddings)
NVIDIA_API_KEY=your_nvidia_api_key

# Google Gemini (LLM)
GEMINI_API_KEY=your_gemini_api_key

# Astra DB (Vector Store)
ASTRA_DB_TOKEN=your_astra_db_token
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_COLLECTION_NAME=your_collection_name
```

### 3. Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 4. Process Documents

Process your PDF documents:

```bash
# Process and upload directly to AstraDB
uv run process_document.py pdfs/CELEX_02013R0575-20250629_EN_TXT.pdf CRR 'CRR'

# Or process to pickle file only (without database upload)
uv run process_document.py pdfs/CELEX_02013L0036-20260111_EN_TXT.pdf CRD 'CRD' --pickle-only

# Later, upload from pickle file
uv run upload_from_pickle.py pickle_files/chunks_CRD_20260306_143022.pkl
```

### 5. Run the Application

**Local:**
```bash
uv run streamlit run st_crr.py
```

**Docker:**
```bash
docker compose up -d --build
```

Access at: `http://localhost:8501`

## 📖 Usage Guide

### Common Workflows

**Workflow 1: Direct Upload (Default)**
```bash
# Process and upload in one step
uv run process_document.py pdfs/regulation.pdf DOC_ID 'Document Name'
```
Use when: You have a stable database connection and want immediate availability.

**Workflow 2: Pickle-First (Safer)**
```bash
# Step 1: Process to pickle only
uv run process_document.py pdfs/regulation.pdf DOC_ID 'Document Name' --pickle-only

# Step 2: Review pickle file if needed
# ...

# Step 3: Upload when ready
uv run upload_from_pickle.py pickle_files/chunks_DOC_ID_20260306_143022.pkl
```
Use when: Processing large documents, testing chunking strategies, or working offline.

**Workflow 3: Re-upload from Backup**
```bash
# Upload existing pickle file (e.g., after database reset)
uv run upload_from_pickle.py pickle_files/chunks_CRR_20260306_143022.pkl
```
Use when: Recovering from database issues or migrating to a new collection.

### Processing Documents

The `process_document.py` script:
- Extracts text from PDF using pypdf
- Identifies article boundaries with regex
- Splits large articles into smaller chunks (max 435 tokens)
- Uploads chunks to AstraDB with metadata
- Saves backup to pickle files

```bash
uv run process_document.py <pdf_path> <document_id> <document_name> [--pickle-only]
```

**Options:**
- `--pickle-only`: Process document and save to pickle file only (skip vector DB upload)

**Examples:**
```bash
# Process and upload to vector database
uv run process_document.py pdfs/regulation.pdf MiFID_II 'MiFID II Regulation'

# Process to pickle only (no database upload)
uv run process_document.py pdfs/regulation.pdf MiFID_II 'MiFID II Regulation' --pickle-only
```

**What happens:**
1. PDF text extracted page by page
2. Articles identified by regex pattern (Article/Članak + number)
3. Text grouped by article with metadata
4. Oversized chunks automatically split (safety threshold: 435 tokens)
5. Chunks uploaded to AstraDB in batches (unless `--pickle-only` is used)
6. Backup saved to `pickle_files/chunks_{document_id}_{timestamp}.pkl`

### Uploading from Pickle Files

If you processed documents with `--pickle-only`, or want to re-upload existing chunks:

```bash
uv run upload_from_pickle.py <pickle_file_path> [--skip-validation]
```

**Options:**
- `--skip-validation`: Skip token validation and upload chunks as-is

**Examples:**
```bash
# Upload with automatic validation and chunk splitting
uv run upload_from_pickle.py pickle_files/chunks_CRR_20260306_143022.pkl

# Upload without validation (faster, but may fail on oversized chunks)
uv run upload_from_pickle.py pickle_files/chunks_CRR_20260306_143022.pkl --skip-validation
```

**What happens:**
1. Loads chunks from pickle file
2. Validates each chunk against 512 token limit (unless `--skip-validation` is used)
3. Automatically splits any oversized chunks
4. Uploads to AstraDB in batches of 25
5. Reports success/failure statistics

### Check Collection

View what documents are in your database:

```bash
uv run check_collection.py
```

**Output:**
```
📚 Documents found in collection:
============================================================
  Document ID: CRR
  Name: CRR
  Chunks: 520
------------------------------------------------------------
  Document ID: CRD
  Name: CRD
  Chunks: 408
------------------------------------------------------------
```

## 🎨 Streamlit Interface Features

- **Auto-Route Mode**: AI automatically selects relevant documents based on query
- **All Documents Mode**: Search across all available documents
- **Specific Documents Mode**: Query only selected documents
- **Chat History**: Persistent conversation within session
- **Document Stats**: View available documents and chunk counts
- **Sample Queries**: Pre-loaded example questions

## 🐳 Docker Deployment

```bash
# Build and run
docker compose up -d --build

# View logs
docker compose logs -f crr-rag-app

# Stop
docker compose down
```

The Dockerfile includes:
- Python 3.12 slim base
- All required dependencies
- Health checks
- Streamlit on port 8501

## 📁 Project Structure

```
crr_rag/
├── st_crr.py                    # Streamlit UI application
├── document_router.py           # Intelligent document routing logic
├── process_document.py          # PDF processing & upload to AstraDB
├── upload_from_pickle.py        # Upload chunks from pickle files
├── check_collection.py          # View database collection stats
├── main.py                      # Empty main entry point
├── requirements.txt             # Python dependencies (pip)
├── pyproject.toml              # UV project configuration
├── Dockerfile                   # Container definition
├── docker-compose.yml          # Docker Compose setup
├── .env                        # Environment variables (create this)
├── pdfs/                       # Source PDF documents
├── pickle_files/               # Processed chunk backups
├── DOCKER_README.md            # Docker-specific documentation
└── MULTI_DOCUMENT_SETUP.md     # Multi-document setup guide
```

## ⚙️ Configuration

### Token Management

The system uses a **safety threshold** (85% of 512 tokens = 435 tokens) to ensure chunks stay within the embedding model's limit:

- Initial chunk size: 1600 characters (~400 tokens)
- Safety threshold: 435 tokens
- Maximum allowed: 512 tokens
- Chunks exceeding the threshold are automatically split

### Document Router

The AI-powered router:
- Analyzes user queries to determine relevant documents
- Uses metadata filtering by `document_id`
- Performs top-k retrieval (default: 10 chunks per query)
- Generates context-aware responses using retrieved chunks

## 🔍 Monitoring

### Verify Processing

After processing a document, check the output:

**With database upload (default):**
```
✅ Extracted {n} articles
🔧 Checking chunks for 512 token limit (using safety threshold: 435)
✅ Processed {n} chunks:
   - {x} within limit
   - {y} split into sub-chunks
   - {z} articles re-detected
   - Final total: {total} chunks
📚 Adding {total} chunks to Astra DB...
✅ Successfully added: {n} chunks
💾 Backup saved to: pickle_files/chunks_{id}_{timestamp}.pkl
```

**With --pickle-only:**
```
✅ Extracted {n} articles
🔧 Checking chunks for 512 token limit (using safety threshold: 435)
✅ Processed {n} chunks
💾 Backup saved to: pickle_files/chunks_{id}_{timestamp}.pkl
💡 To upload to vector database later, use upload_from_pickle.py
```

### Database Check

Verify documents in your collection:
```bash
uv run check_collection.py
```

## 🐛 Troubleshooting

### Token Limit Errors

If batches fail with "Input length exceeds maximum allowed token size 512":
- The safety threshold should prevent this, but if it occurs:
- Use `--pickle-only` to process first, then review chunks
- Upload with validation enabled: `uv run upload_from_pickle.py <pickle_file>`
- Check your chunk size settings in `process_document.py`
- Verify the `safe_token_limit` is set to `int(max_tokens * 0.85)`

**Recommended approach:**
```bash
# Process to pickle first (safer for large documents)
uv run process_document.py pdfs/document.pdf DOC_ID 'Name' --pickle-only

# Upload with automatic validation and splitting
uv run upload_from_pickle.py pickle_files/chunks_DOC_ID_timestamp.pkl
```

### Docker Build Issues

```bash
# Clean rebuild
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### Missing Documents in Router

If documents don't appear:
1. Verify they're in AstraDB: `uv run check_collection.py`
2. Check metadata includes `document_id` and `document_name`
3. Restart Streamlit to refresh cached data

### PDF Processing Errors

If pypdf fails to extract text:
- Ensure PDF is text-based (not scanned images)
- Check PDF file is not corrupted
- Verify file path is correct

## 📊 Performance

- **PDF Processing**: ~2-5 seconds per document (pypdf)
- **Embedding**: ~100 chunks/minute (NVIDIA API rate limits)
- **Query Response**: ~2-3 seconds average
- **Document Loading**: Cached after first load in Streamlit

## 🔐 Security

- All API keys stored in `.env` file (never commit to git!)
- `.env` is in `.gitignore`
- Docker secrets support available for production
- HTTPS recommended for production deployments

## 📝 Available Scripts

| Script | Purpose |
|--------|---------|
| `process_document.py` | Process PDF and upload to AstraDB (or pickle only) |
| `upload_from_pickle.py` | Upload chunks from pickle files to AstraDB |
| `check_collection.py` | View documents in database |
| `st_crr.py` | Run Streamlit web interface |
| `document_router.py` | Document routing logic (imported) |

## 📧 Support

For issues or questions, check existing documentation:
- `DOCKER_README.md` - Docker deployment details
- `MULTI_DOCUMENT_SETUP.md` - Multi-document configuration

---

Built with ❤️ using LangChain, AstraDB, pypdf, and Google Gemini
