# Multi-Document RAG System - Setup Guide

This guide explains how to add new regulation PDFs and use the multi-document RAG system.

## 📋 Overview

Your RAG system now supports:
- **Multiple PDF documents** with separate metadata tracking
- **Intelligent document routing** - AI automatically selects relevant documents
- **Cross-document queries** - Combine knowledge from multiple regulations
- **Flexible querying modes** - Auto-route, all documents, or specific documents

## 🚀 Quick Start: Adding a New Document

### Step 1: Place Your PDF File

Put your new regulation PDF in the project directory:
```bash
# Example: Add MiFID II regulation
cp /path/to/your/regulation.pdf /home/ivan/Documents/crr_rag/
```

### Step 2: Process the Document

Run the document processor script:

```bash
python process_document.py <pdf_path> <document_id> <document_name>
```

**Parameters:**
- `pdf_path`: Path to your PDF file
- `document_id`: Short unique identifier (e.g., `MiFID_II`, `UCITS`, `AIFMD`)
- `document_name`: Full document name (e.g., `"MiFID II Regulation"`)

**Example:**
```bash
# Process a new regulation
python process_document.py regulation.pdf MiFID_II "MiFID II Regulation"
```

### Step 3: Verify the Upload

The script will:
1. ✅ Extract articles from the PDF using Docling
2. ✅ Split large chunks to fit embedding model limits
3. ✅ Add document metadata (document_id, document_name)
4. ✅ Upload chunks to Astra DB vector store
5. ✅ Create a backup pickle file

You'll see output like:
```
📄 Processing: regulation.pdf
🆔 Document ID: MiFID_II
📝 Document Name: MiFID II Regulation
...
✅ Successfully stored: 1245 chunks
```

### Step 4: Use in Streamlit App

Run your Streamlit app:
```bash
streamlit run st_crr.py
```

The new document will automatically appear in the sidebar!

## 🎯 Query Modes

### 1. Auto-Route (Recommended)
The AI automatically determines which documents are relevant to your question.

**Example:**
- Question: *"What are capital requirements?"*
- AI routes to: **CRR** only
- Question: *"How do CRR and MiFID requirements interact?"*
- AI routes to: **CRR + MiFID_II**

### 2. All Documents
Search across all available documents simultaneously.

**Use when:**
- Comparing regulations
- Finding common themes
- Comprehensive research

### 3. Specific Documents
Manually select which documents to query.

**Use when:**
- You know exactly which regulation applies
- Focused analysis on specific frameworks

## 📊 How It Works

### Document Processing Pipeline

1. **PDF Conversion** (Docling)
   - Extracts text and structure from PDF
   - Preserves article numbers and page references

2. **Article Extraction**
   - Detects articles by pattern: `Article 123` or `Članak 123`
   - Groups content by article

3. **Chunk Splitting**
   - Splits large articles to fit NVIDIA embedding model (512 tokens)
   - Re-detects article numbers in sub-chunks
   - Maintains metadata integrity

4. **Metadata Enrichment**
   - Adds `document_id` - unique identifier
   - Adds `document_name` - human-readable name
   - Preserves `article_no`, `page`, `language`, etc.

5. **Vector Storage**
   - Generates embeddings with NVIDIA NV-Embed-QA
   - Stores in Astra DB with full metadata
   - Enables semantic search across documents

### Intelligent Routing

The `DocumentRouter` class:
1. Analyzes your question
2. Compares against available documents
3. Routes to the most relevant source(s)
4. Retrieves and combines results
5. Generates answer with proper citations

## 🗄️ Database Schema

Each chunk in Astra DB contains:

```python
{
    "content": "Article text content...",
    "metadata": {
        "document_id": "MiFID_II",           # Unique document identifier
        "document_name": "MiFID II Regulation",  # Human-readable name
        "article_no": "Article 25",
        "article_number": 25,
        "page": 42,
        "type": "article",
        "language": "en",
        # ... other metadata
    },
    "$vector": [0.123, 0.456, ...]  # Embedding vector
}
```

## 🔍 Example Workflows

### Workflow 1: Add Second Regulation

```bash
# 1. Download or place your PDF
cp ~/Downloads/UCITS_Directive.pdf .

# 2. Process it
python process_document.py UCITS_Directive.pdf UCITS "UCITS Directive"

# 3. Run Streamlit
streamlit run st_crr.py

# 4. Ask cross-document questions
# "Compare fund requirements between CRR and UCITS"
```

### Workflow 2: Query Specific Documents

1. Open Streamlit app
2. Select "🎯 Specific Documents" mode
3. Choose documents from multiselect
4. Ask your question
5. See results only from selected documents

### Workflow 3: Automatic Document Discovery

1. Open Streamlit app
2. Keep "🤖 Auto-Route" mode (default)
3. Ask: *"What are the requirements for investment firms?"*
4. AI automatically routes to relevant document(s)
5. See which documents were queried in the response

## 🔧 Advanced Usage

### Test Document Router

```python
from document_router import DocumentRouter

router = DocumentRouter()

# Get document statistics
stats = router.get_document_stats()
print(stats)

# Manual routing test
doc_ids = router.route_query("What are capital requirements?")
print(f"Routing to: {doc_ids}")

# Query with routing
answer, docs, queried_ids = router.answer_with_routing(
    "What are capital requirements?",
    use_routing=True
)
print(answer)
```

### Query Specific Documents Programmatically

```python
# Query only CRR
answer, docs, _ = router.answer_with_routing(
    "What is Tier 1 capital?",
    specific_documents=["CRR"]
)

# Query multiple specific documents
answer, docs, _ = router.answer_with_routing(
    "Compare liquidity requirements",
    specific_documents=["CRR", "MiFID_II"]
)
```

## 📝 Current Documents

Your system currently has:
- **CRR**: Capital Requirements Regulation (CELEX_02013R0575-20250629_EN_TXT.pdf)

## ⚙️ Configuration

All configuration is in `.env`:
```env
NVIDIA_API_KEY=your_nvidia_api_key
GEMINI_API_KEY=your_gemini_api_key
ASTRA_DB_TOKEN=your_astra_token
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
```

## 🐛 Troubleshooting

### Document not appearing in Streamlit
- Check that chunks were successfully added to Astra DB
- Verify `document_id` and `document_name` in metadata
- Restart Streamlit app to refresh cache

### Routing not working as expected
- Check that documents have distinct content
- Verify document_id is properly set in all chunks
- Try "All Documents" mode to see if retrieval works

### Processing fails
- Check PDF is not encrypted or password-protected
- Ensure Docling can read the PDF format
- Verify sufficient memory for large PDFs

## 📚 Next Steps

1. **Add more documents**: Process additional regulations
2. **Customize routing**: Edit routing prompt in `document_router.py`
3. **Adjust retrieval**: Modify `k` parameter for more/fewer chunks
4. **Fine-tune prompts**: Customize system prompts for specific domains

## 🎉 You're Ready!

Your multi-document RAG system is now set up. To add a new regulation:

```bash
python process_document.py your_regulation.pdf YOUR_ID "Your Regulation Name"
```

Then query across documents in the Streamlit app!
