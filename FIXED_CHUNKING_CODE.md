# Fixed Chunking Functions for CRR RAG

## Use this code to replace your split_large_chunks function

```python
def split_large_chunks_with_redetection(chunks, max_tokens=512):
    """
    Split chunks that exceed token limit AND re-detect article numbers.
    
    CRITICAL: This fixes the metadata inheritance bug where sub-chunks
    incorrectly inherit article numbers from their parent chunk.
    """
    import re
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print(f"🔧 Checking chunks for NVIDIA's {max_tokens} token limit...")
    print(f"🔍 Re-detecting article numbers in sub-chunks...")
    
    valid_chunks = []
    oversized_count = 0
    redetected_count = 0
    
    # Article pattern for re-detection
    article_pattern = re.compile(r'^(Članak|Article)\s+(\d+)', re.IGNORECASE | re.MULTILINE)
    
    # Text splitter for oversized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,  # ~450 tokens (safe margin)
        chunk_overlap=200,
        separators=[
            "\n\n",          # Paragraph breaks (DON'T split at Article)
            "\n",            # Line breaks
            ". ",            # Sentence breaks
            " ",             # Word breaks
            ""
        ],
        keep_separator=True,
        length_function=len,
    )
    
    def estimate_tokens(text):
        """Rough estimation: 1 token ≈ 4 characters"""
        return len(text) / 4
    
    for i, chunk in enumerate(chunks):
        estimated_tokens = estimate_tokens(chunk.page_content)
        
        if estimated_tokens <= max_tokens:
            # Chunk is fine as-is
            valid_chunks.append(chunk)
        else:
            # Chunk is too large, split it
            oversized_count += 1
            original_article = chunk.metadata.get('article_number', 'unknown')
            print(f"  Splitting chunk {i+1} (Article {original_article}, {estimated_tokens:.0f} tokens)")
            
            # Split the oversized chunk
            sub_chunks = text_splitter.split_documents([chunk])
            
            # Process each sub-chunk
            for j, sub_chunk in enumerate(sub_chunks):
                # Start with original metadata
                sub_chunk.metadata = chunk.metadata.copy()
                sub_chunk.metadata['sub_chunk'] = j + 1
                sub_chunk.metadata['total_sub_chunks'] = len(sub_chunks)
                sub_chunk.metadata['original_chunk_tokens'] = int(estimated_tokens)
                sub_chunk.metadata['original_article_number'] = original_article
                
                # ✅ RE-DETECT article number in this specific sub-chunk
                first_300_chars = sub_chunk.page_content[:300].strip()
                article_match = article_pattern.search(first_300_chars)
                
                if article_match:
                    # Found an article in this sub-chunk - update metadata
                    article_type = article_match.group(1)
                    article_num = article_match.group(2)
                    detected_article_number = int(article_num)
                    
                    # Only update if different from original
                    if detected_article_number != original_article:
                        sub_chunk.metadata['article_no'] = f"{article_type} {article_num}"
                        sub_chunk.metadata['article_number'] = detected_article_number
                        sub_chunk.metadata['language'] = "hr" if article_type == "Članak" else "en"
                        sub_chunk.metadata['article_redetected'] = True
                        redetected_count += 1
                        print(f"    ✅ Re-detected {article_type} {article_num} in sub-chunk {j+1} (was {original_article})")
                    else:
                        sub_chunk.metadata['article_redetected'] = False
                else:
                    # No article detected - this is a continuation chunk
                    sub_chunk.metadata['article_redetected'] = False
                    sub_chunk.metadata['is_continuation'] = True
                
                # Verify sub-chunk size
                sub_tokens = estimate_tokens(sub_chunk.page_content)
                if sub_tokens <= max_tokens:
                    valid_chunks.append(sub_chunk)
                else:
                    print(f"    ⚠️  Warning: Sub-chunk still too large ({sub_tokens:.0f} tokens)")
                    # Could recursively split again here if needed
    
    print(f"\n✅ Processed {len(chunks)} chunks:")
    print(f"   - {len(chunks) - oversized_count} chunks were within limit")
    print(f"   - {oversized_count} chunks were split into sub-chunks")
    print(f"   - {redetected_count} sub-chunks had articles re-detected")
    print(f"   - Final total: {len(valid_chunks)} chunks")
    
    # Verify article coverage
    article_numbers = sorted(set(c.metadata.get('article_number') for c in valid_chunks if c.metadata.get('article_number')))
    if article_numbers:
        print(f"\n📊 Article coverage:")
        print(f"   - Range: Article {min(article_numbers)} - {max(article_numbers)}")
        print(f"   - Total unique articles: {len(article_numbers)}")
        
        # Check for specific articles
        if 32 in article_numbers and 33 in article_numbers:
            print(f"   - ✅ Articles 32 & 33 are present!")
        else:
            if 32 not in article_numbers:
                print(f"   - ❌ Article 32 is MISSING!")
            if 33 not in article_numbers:
                print(f"   - ❌ Article 33 is MISSING!")
    
    return valid_chunks


# Usage in your notebook:
# Replace this line:
#   valid_chunks = split_large_chunks(chunks, max_tokens=512)
# With this:
#   valid_chunks = split_large_chunks_with_redetection(chunks, max_tokens=512)
```

## Alternative: Article-Aware Splitter

If you want to preserve entire articles without splitting them:

```python
def split_respecting_articles(chunks, max_tokens=512):
    """
    Split chunks but try to keep articles together.
    If an article is too long, mark it clearly.
    """
    import re
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    valid_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=50,  # Minimal overlap to avoid crossing articles
        separators=[
            "\n\n\n",        # Triple newlines (strong article separation)
            "\n\n",          # Paragraph breaks
            "\n",
        ],
    )
    
    def estimate_tokens(text):
        return len(text) / 4
    
    for chunk in chunks:
        tokens = estimate_tokens(chunk.page_content)
        article_num = chunk.metadata.get('article_number')
        
        if tokens <= max_tokens:
            valid_chunks.append(chunk)
        else:
            # Article is too long - split but preserve article metadata
            print(f"  ⚠️  Article {article_num} is very long ({tokens:.0f} tokens)")
            print(f"     Splitting into parts while maintaining article identity")
            
            sub_chunks = text_splitter.split_documents([chunk])
            
            for j, sub_chunk in enumerate(sub_chunks, 1):
                # Keep same article number for all parts
                sub_chunk.metadata = chunk.metadata.copy()
                sub_chunk.metadata['article_part'] = j
                sub_chunk.metadata['total_parts'] = len(sub_chunks)
                sub_chunk.metadata['is_long_article'] = True
                
                valid_chunks.append(sub_chunk)
                print(f"     - Part {j}/{len(sub_chunks)}: {estimate_tokens(sub_chunk.page_content):.0f} tokens")
    
    return valid_chunks
```

## Validation Function

Add this to verify your chunks:

```python
def validate_article_coverage(chunks):
    """Validate that all expected articles are present"""
    import re
    
    article_numbers = []
    chunks_per_article = {}
    
    for chunk in chunks:
        article_num = chunk.metadata.get('article_number')
        if article_num:
            article_numbers.append(article_num)
            chunks_per_article[article_num] = chunks_per_article.get(article_num, 0) + 1
    
    article_numbers = sorted(set(article_numbers))
    
    print("📊 Article Coverage Validation")
    print("="*60)
    print(f"Total unique articles: {len(article_numbers)}")
    print(f"Article range: {min(article_numbers)} - {max(article_numbers)}")
    
    # Check for gaps
    expected_range = set(range(min(article_numbers), max(article_numbers) + 1))
    missing = sorted(expected_range - set(article_numbers))
    
    if missing:
        print(f"\n⚠️  Missing articles: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if 32 in missing:
            print(f"   🔴 Article 32 is MISSING!")
        if 33 in missing:
            print(f"   🔴 Article 33 is MISSING!")
    else:
        print(f"\n✅ No gaps in article coverage!")
    
    # Show distribution
    print(f"\n📈 Chunks per article statistics:")
    chunk_counts = list(chunks_per_article.values())
    print(f"   - Average: {sum(chunk_counts) / len(chunk_counts):.1f} chunks/article")
    print(f"   - Min: {min(chunk_counts)} chunks")
    print(f"   - Max: {max(chunk_counts)} chunks")
    
    # Articles with many chunks (might be problematic)
    large_articles = [(num, count) for num, count in chunks_per_article.items() if count > 5]
    if large_articles:
        print(f"\n📦 Articles with >5 chunks:")
        for num, count in sorted(large_articles, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - Article {num}: {count} chunks")
    
    return article_numbers, missing

# Run validation
article_numbers, missing_articles = validate_article_coverage(valid_chunks)
```
