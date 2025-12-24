# 🔴 CRITICAL ISSUE: Why Articles 32 & 33 Are Missing

## 🐛 Root Cause Identified

Your chunking strategy has a **fundamental flaw** in the article extraction logic that causes articles to be **silently lost** during processing.

---

## 📋 Current Workflow

### Step 1: Article Extraction (Lines 70-160)
```python
# Pattern for Croatian articles
article_pattern = re.compile(r'^(Članak|Article)\s+(\d+)', re.IGNORECASE)

for item, level in docling_doc.iterate_items():
    # Check if this is an article heading
    article_match = article_pattern.match(txt)
    
    if article_match:
        # Save PREVIOUS chunk before starting new one
        if current:
            chunks.append(Document(**current))
        
        # Start NEW article chunk
        current = {
            "page_content": txt + "\n",
            "metadata": {
                "article_no": f"{article_type} {article_num}",
                "article_number": int(article_num),
                ...
            }
        }
    elif current:
        # Append content to CURRENT article
        current["page_content"] += txt + "\n"

# Don't forget the last chunk
if current:
    chunks.append(Document(**current))
```

**Result:** Creates chunks where **each chunk = one complete article**

---

### Step 2: Token-Based Splitting (Lines 259-320)
```python
def split_large_chunks(chunks, max_tokens=512):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,  # ~450 tokens
        separators=[
            "\n\nArticle ",   # ← Splits at article boundaries
            "\n\n",
            ...
        ]
    )
    
    for chunk in chunks:
        if estimated_tokens > max_tokens:
            # Split the oversized chunk
            sub_chunks = text_splitter.split_documents([chunk])
            
            # Add metadata to sub-chunks
            for sub_chunk in sub_chunks:
                sub_chunk.metadata = chunk.metadata.copy()  # ← PROBLEM!
                sub_chunk.metadata['sub_chunk'] = j + 1
```

---

## 🔴 THE CRITICAL FLAW

### Problem: Metadata Inheritance Without Re-Detection

When an article is split into sub-chunks:

1. **Original chunk:** Has `article_number: 92` in metadata
2. **Split happens:** Creates 3 sub-chunks
3. **All sub-chunks inherit:** `article_number: 92`
4. **BUT:** If sub-chunk starts with "Article 93", metadata still says 92!

### Example: Article 92 is Long

```
Original Chunk (article_number: 92):
┌─────────────────────────────────────────┐
│ Article 92                              │
│ Own funds requirements                  │
│ 1. Institutions shall meet...           │
│ (2000 tokens - TOO LARGE)              │
│                                         │
│ Article 93  ← New article starts here! │
│ Leverage ratio                          │
│ ...                                     │
└─────────────────────────────────────────┘

After Splitting (all have article_number: 92):
┌────────────────────┐  ┌────────────────────┐
│ Sub-chunk 1        │  │ Sub-chunk 2        │
│ Article 92         │  │ Article 93  ← BUG! │
│ Own funds...       │  │ Leverage...        │
│ (article_number:92)│  │ (article_number:92)│ ← WRONG!
└────────────────────┘  └────────────────────┘
```

**Result:** Article 93 text exists but has wrong metadata (says it's Article 92)!

---

## 🎯 Why Articles 32 & 33 Specifically?

### Hypothesis 1: They're Part of a Long Article

If Article 31 or Article 32 is very long:

```
Article 31 (very long, 3000 tokens)
├─ Sub-chunk 1: "Article 31..." (metadata: 31) ✅
├─ Sub-chunk 2: "...continuation..." (metadata: 31) ✅
└─ Sub-chunk 3: "Article 32..." (metadata: 31) ❌ WRONG!
    └─ Article 32 content has metadata saying it's Article 31!
```

### Hypothesis 2: Articles 32-33 Never Matched the Pattern

Possible reasons:
- Different formatting: "32. Article" instead of "Article 32"
- Extra whitespace: "Article  32" (two spaces)
- Different language: "Artikel 32" or encoding issue
- OCR error: "Articl e 32" or "Artic1e 32"
- Skipped pages during PDF extraction

---

## 🔬 How to Verify

### Test 1: Check Original Extraction

```python
# Before splitting, check if articles 32 & 33 exist
article_numbers = [c.metadata.get('article_number') for c in chunks]
print(f"Articles extracted: {sorted(set(article_numbers))}")

if 32 not in article_numbers:
    print("❌ Article 32 never extracted from Docling!")
if 33 not in article_numbers:
    print("❌ Article 33 never extracted from Docling!")
```

### Test 2: Check After Splitting

```python
# After splitting, check metadata
for chunk in valid_chunks:
    if "Article 32" in chunk.page_content or "Članak 32" in chunk.page_content:
        print(f"Found Article 32 text with metadata: {chunk.metadata}")
```

---

## ✅ SOLUTIONS

### Solution 1: Re-Detect Articles After Splitting (RECOMMENDED)

```python
def split_large_chunks_with_redetection(chunks, max_tokens=512):
    """Split chunks and RE-DETECT article numbers in each sub-chunk"""
    article_pattern = re.compile(r'^(Članak|Article)\s+(\d+)', re.IGNORECASE)
    
    valid_chunks = []
    
    for chunk in chunks:
        estimated_tokens = estimate_tokens(chunk.page_content)
        
        if estimated_tokens <= max_tokens:
            valid_chunks.append(chunk)
        else:
            # Split oversized chunk
            sub_chunks = text_splitter.split_documents([chunk])
            
            for j, sub_chunk in enumerate(sub_chunks):
                # Start with original metadata
                sub_chunk.metadata = chunk.metadata.copy()
                sub_chunk.metadata['sub_chunk'] = j + 1
                
                # RE-DETECT article number in this sub-chunk
                first_line = sub_chunk.page_content[:200].strip()
                article_match = article_pattern.match(first_line)
                
                if article_match:
                    # Update metadata with ACTUAL article in this chunk
                    article_type = article_match.group(1)
                    article_num = article_match.group(2)
                    sub_chunk.metadata['article_no'] = f"{article_type} {article_num}"
                    sub_chunk.metadata['article_number'] = int(article_num)
                    sub_chunk.metadata['article_detected_in_subchunk'] = True
                    print(f"  ✅ Re-detected {article_type} {article_num} in sub-chunk")
                
                valid_chunks.append(sub_chunk)
    
    return valid_chunks
```

---

### Solution 2: Don't Split at Article Boundaries

```python
# Change separator list to KEEP articles together
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    separators=[
        # ❌ DON'T split at articles: "\n\nArticle ",
        "\n\n",          # Split at paragraph breaks instead
        "\n",
        ". ",
        " ",
    ]
)
```

**Problem:** This might create chunks with partial articles.

---

### Solution 3: Chunk FIRST, Then Detect Articles (BEST)

```python
# 1. Load document with Docling
docling_doc = result.document

# 2. Get ALL text as one big string
full_text = docling_doc.export_to_markdown()

# 3. Split into token-sized chunks FIRST
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". "]
)
raw_chunks = text_splitter.create_documents([full_text])

# 4. THEN detect articles in each chunk
article_pattern = re.compile(r'^(Članak|Article)\s+(\d+)', re.IGNORECASE)

for chunk in raw_chunks:
    # Detect article in THIS specific chunk
    first_lines = chunk.page_content[:300]
    article_match = article_pattern.search(first_lines)  # Use search, not match
    
    if article_match:
        article_type = article_match.group(1)
        article_num = article_match.group(2)
        chunk.metadata['article_no'] = f"{article_type} {article_num}"
        chunk.metadata['article_number'] = int(article_num)
```

---

## 🎯 Immediate Action Plan

### Step 1: Diagnose
```bash
# Run your debug notebook
jupyter notebook debug_articles.ipynb
```

Check:
- Are Articles 32 & 33 in `chunks` (before splitting)?
- Are they in `valid_chunks` (after splitting)?
- Do they have correct `article_number` metadata?

### Step 2: Fix
Choose Solution 1 (re-detection) and update your notebook:

```python
# Replace the split_large_chunks function with the version that re-detects
```

### Step 3: Re-Process
```python
# Re-run the extraction and splitting
chunks = extract_articles(docling_doc)  # Your original extraction
valid_chunks = split_large_chunks_with_redetection(chunks, max_tokens=512)

# Verify
article_nums = {c.metadata.get('article_number') for c in valid_chunks}
print(f"Articles in valid_chunks: {sorted(article_nums)}")
assert 32 in article_nums, "Article 32 missing!"
assert 33 in article_nums, "Article 33 missing!"
```

### Step 4: Re-Upload to Astra DB
```python
# Clear old collection or create new one
vectorstore.delete_collection()

# Upload fixed chunks
vectorstore.add_documents(valid_chunks)
```

---

## 📊 Expected Results

### Before Fix:
```
Articles 1-31: ✅ Found
Articles 32-33: ❌ Missing (metadata says 31 or 34)
Articles 34-521: ✅ Found
```

### After Fix:
```
Articles 1-521: ✅ All found with correct metadata
```

---

## 💡 Prevention for Future

### Best Practice: Stateless Chunking

```python
# DON'T rely on metadata inheritance during splitting
# DO re-detect metadata for each final chunk

def enhance_chunk_metadata(chunk):
    """Add article detection to ANY chunk"""
    article_pattern = re.compile(r'(Article|Članak)\s+(\d+)')
    matches = article_pattern.findall(chunk.page_content)
    
    if matches:
        # Get first article mentioned
        article_type, article_num = matches[0]
        chunk.metadata['article_number'] = int(article_num)
        chunk.metadata['mentions_articles'] = [int(m[1]) for m in matches]
    
    return chunk
```

---

## 🎓 Key Takeaway

**Metadata inheritance during splitting is DANGEROUS** when:
1. Original chunks are article-based
2. Splitting might cross article boundaries
3. No re-detection happens after splitting

**Solution:** Always re-detect domain-specific metadata (like article numbers) in final chunks!

---

## 🚀 Next Steps

1. ✅ Run `debug_articles.ipynb` to confirm diagnosis
2. ✅ Implement Solution 1 (re-detection) in `crr_rag.ipynb`
3. ✅ Re-process the PDF
4. ✅ Re-upload to Astra DB
5. ✅ Test queries for Articles 32 & 33
6. ✅ Verify all 521 articles are present

Good luck! 🎯
