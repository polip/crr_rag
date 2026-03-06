# 📚 Multi-Document Legal RAG System

A Retrieval-Augmented Generation (RAG) system for querying financial regulation documents (CRR, CRD) with intelligent document routing and multi-document support.

## 🎯 Features

- **Multi-Document Support**: Query across multiple regulation documents (CRR, CRD, etc.)
- **Intelligent Document Routing**: AI automatically selects relevant documents for each query
- **Article-Level Chunking**: Extracts and processes documents by article boundaries
- **Smart Token Management**: Automatic chunk splitting with 435-token safety threshold
- **Vector Storage**: Powered by AstraDB for efficient similarity search
- **Modern UI**: Streamlit-based chat interface
- **Docker Support**: Easy deployment with Docker Compose
- **Pickle Backup**: Processed chunks saved for backup

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
# Process CRR (Capital Requirements Regulation)
uv run process_document.py pdfs/CELEX_02013R0575-20250629_EN_TXT.pdf CRR 'CRR'

# Process CRD (Capital Requirements Directive)
uv run process_document.py pdfs/CELEX_02013L0036-20260111_EN_TXT.pdf CRD 'CRD'
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

### Processing Documents

The `process_document.py` script:
- Extracts text from PDF using pypdf
- Identifies article boundaries with regex
- Splits large articles into smaller chunks (max 435 tokens)
- Uploads chunks to AstraDB with metadata
- Saves backup to pickle files

```bash
uv run process_document.py <pdf_path> <document_id> <document_name>
```

**Example:**
```bash
uv run process_document.py pdfs/regulation.pdf MiFID_II 'MiFID II Regulation'
```

**What happens:**
1. PDF text extracted page by page
2. Articles identified by regex pattern (Article/Članak + number)
3. Text grouped by article with metadata
4. Oversized chunks automatically split (safety threshold: 435 tokens)
5. Chunks uploaded to AstraDB in batches
6. Backup saved to `pickle_files/chunks_{document_id}_{timestamp}.pkl`

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

### Database Check

Verify documents in your collection:
```bash
uv run check_collection.py
```

## 🐛 Troubleshooting

### Token Limit Errors

If batches fail with "Input length exceeds maximum allowed token size 512":
- The safety threshold should prevent this, but if it occurs:
- Check your chunk size settings in `process_document.py`
- Verify the `safe_token_limit` is set to `int(max_tokens * 0.85)`

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
| `process_document.py` | Process PDF and upload to AstraDB |
| `check_collection.py` | View documents in database |
| `st_crr.py` | Run Streamlit web interface |
| `document_router.py` | Document routing logic (imported) |

## 📧 Support

For issues or questions, check existing documentation:
- `DOCKER_README.md` - Docker deployment details
- `MULTI_DOCUMENT_SETUP.md` - Multi-document configuration

---

Built with using LangChain, AstraDB, pypdf, and Google Gemini
