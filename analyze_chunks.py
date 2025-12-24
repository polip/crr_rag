#!/usr/bin/env python3
"""
Analyze pickle files to debug article recognition issues
"""

import pickle
import os
from pathlib import Path
import re

def load_pickle_file(filepath):
    """Load a pickle file and return its contents"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def analyze_article_coverage(chunks):
    """Analyze which articles are present in the chunks"""
    article_pattern = re.compile(r'(Article|Članak)\s+(\d+)', re.IGNORECASE)
    
    articles_found = {}
    chunks_per_article = {}
    
    print("\n" + "="*80)
    print("📊 ARTICLE COVERAGE ANALYSIS")
    print("="*80)
    
    for i, chunk in enumerate(chunks):
        # Try to get text from different possible structures
        if hasattr(chunk, 'page_content'):
            text = chunk.page_content
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        elif hasattr(chunk, 'text'):
            text = chunk.text
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        elif isinstance(chunk, dict):
            text = chunk.get('content', chunk.get('text', ''))
            metadata = chunk.get('metadata', {})
        else:
            text = str(chunk)
            metadata = {}
        
        # Check for article mentions
        matches = article_pattern.findall(text)
        
        if matches:
            for article_type, article_num in matches:
                article_num = int(article_num)
                article_key = f"{article_type} {article_num}"
                
                if article_key not in articles_found:
                    articles_found[article_key] = []
                    chunks_per_article[article_num] = 0
                
                articles_found[article_key].append({
                    'chunk_index': i,
                    'metadata': metadata,
                    'text_preview': text[:200]
                })
                chunks_per_article[article_num] = chunks_per_article.get(article_num, 0) + 1
    
    return articles_found, chunks_per_article

def search_for_specific_articles(chunks, target_articles=[32, 33]):
    """Search specifically for articles 32 and 33"""
    print("\n" + "="*80)
    print(f"🔍 SEARCHING FOR ARTICLES: {target_articles}")
    print("="*80)
    
    article_pattern = re.compile(r'(Article|Članak)\s+(\d+)', re.IGNORECASE)
    
    for target in target_articles:
        print(f"\n📌 Searching for Article {target}...")
        found_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get text
            if hasattr(chunk, 'page_content'):
                text = chunk.page_content
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            elif hasattr(chunk, 'text'):
                text = chunk.text
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            elif isinstance(chunk, dict):
                text = chunk.get('content', chunk.get('text', ''))
                metadata = chunk.get('metadata', {})
            else:
                text = str(chunk)
                metadata = {}
            
            # Search for the article
            if f"Article {target}" in text or f"Članak {target}" in text:
                found_chunks.append({
                    'index': i,
                    'text': text,
                    'metadata': metadata
                })
        
        if found_chunks:
            print(f"   ✅ Found {len(found_chunks)} chunk(s) with Article {target}")
            for chunk_info in found_chunks:
                print(f"\n   📄 Chunk #{chunk_info['index']}:")
                print(f"   Metadata: {chunk_info['metadata']}")
                print(f"   Text preview: {chunk_info['text'][:300]}...")
                print(f"   Text length: {len(chunk_info['text'])} chars")
        else:
            print(f"   ❌ Article {target} NOT FOUND in any chunks!")
            
            # Try fuzzy search
            print(f"\n   🔎 Trying fuzzy search for '{target}'...")
            fuzzy_found = False
            for i, chunk in enumerate(chunks[:50]):  # Check first 50 chunks
                if hasattr(chunk, 'page_content'):
                    text = chunk.page_content
                elif hasattr(chunk, 'text'):
                    text = chunk.text
                elif isinstance(chunk, dict):
                    text = chunk.get('content', chunk.get('text', ''))
                else:
                    text = str(chunk)
                
                if str(target) in text[:500]:  # Check if number appears
                    print(f"   ⚠️  Found '{target}' in chunk #{i}: {text[:200]}...")
                    fuzzy_found = True
                    break
            
            if not fuzzy_found:
                print(f"   ❌ Number '{target}' not found in first 50 chunks")

def check_chunk_structure(chunks):
    """Analyze the structure of chunks"""
    print("\n" + "="*80)
    print("🔬 CHUNK STRUCTURE ANALYSIS")
    print("="*80)
    
    if not chunks:
        print("❌ No chunks found!")
        return
    
    print(f"\n📊 Total chunks: {len(chunks)}")
    print(f"📝 First chunk type: {type(chunks[0])}")
    
    # Analyze first chunk in detail
    first_chunk = chunks[0]
    print(f"\n📋 First chunk attributes:")
    if hasattr(first_chunk, '__dict__'):
        for key, value in first_chunk.__dict__.items():
            print(f"   - {key}: {type(value).__name__}")
    elif isinstance(first_chunk, dict):
        for key in first_chunk.keys():
            print(f"   - {key}: {type(first_chunk[key]).__name__}")
    
    # Show metadata structure
    if hasattr(first_chunk, 'metadata'):
        print(f"\n🏷️  Metadata structure:")
        print(f"   {first_chunk.metadata}")
    elif isinstance(first_chunk, dict) and 'metadata' in first_chunk:
        print(f"\n🏷️  Metadata structure:")
        print(f"   {first_chunk['metadata']}")

def generate_article_report(articles_found, chunks_per_article):
    """Generate a detailed report of article coverage"""
    print("\n" + "="*80)
    print("📈 ARTICLE DISTRIBUTION REPORT")
    print("="*80)
    
    # Sort articles by number
    sorted_articles = sorted(chunks_per_article.items())
    
    # Find gaps
    if sorted_articles:
        min_article = sorted_articles[0][0]
        max_article = sorted_articles[-1][0]
        
        print(f"\n📊 Article range: {min_article} - {max_article}")
        print(f"📊 Total unique articles: {len(sorted_articles)}")
        print(f"📊 Total chunks: {sum(chunks_per_article.values())}")
        
        # Find missing articles in range
        all_numbers = set(range(min_article, max_article + 1))
        found_numbers = set(chunks_per_article.keys())
        missing = sorted(all_numbers - found_numbers)
        
        if missing:
            print(f"\n⚠️  Missing articles in range {min_article}-{max_article}:")
            print(f"   {missing[:20]}{'...' if len(missing) > 20 else ''}")
            print(f"   Total missing: {len(missing)}")
            
            # Highlight if 32 or 33 are missing
            if 32 in missing:
                print(f"   🔴 Article 32 is MISSING!")
            if 33 in missing:
                print(f"   🔴 Article 33 is MISSING!")
    
    # Show articles with most chunks
    print(f"\n📊 Top 10 articles by chunk count:")
    top_articles = sorted(chunks_per_article.items(), key=lambda x: x[1], reverse=True)[:10]
    for article_num, count in top_articles:
        print(f"   Article {article_num}: {count} chunks")
    
    # Show articles with fewest chunks
    print(f"\n📊 Bottom 10 articles by chunk count:")
    bottom_articles = sorted(chunks_per_article.items(), key=lambda x: x[1])[:10]
    for article_num, count in bottom_articles:
        print(f"   Article {article_num}: {count} chunks")

def main():
    """Main analysis function"""
    print("🔍 CRR Chunk Analysis Tool")
    print("="*80)
    
    # Find pickle files
    pkl_files = list(Path('.').glob('*.pkl'))
    
    if not pkl_files:
        print("❌ No .pkl files found in current directory")
        print("\n💡 Looking in common locations...")
        
        # Check common locations
        common_paths = [
            Path('.'),
            Path('./data'),
            Path('./chunks'),
            Path('./output'),
        ]
        
        for path in common_paths:
            if path.exists():
                pkl_files.extend(path.glob('*.pkl'))
    
    if not pkl_files:
        print("❌ No pickle files found!")
        print("\n💡 Please specify the path to your pickle file")
        return
    
    print(f"\n📁 Found {len(pkl_files)} pickle file(s):")
    for pf in pkl_files:
        print(f"   - {pf}")
    
    # Analyze each file
    for pkl_file in pkl_files:
        print(f"\n{'='*80}")
        print(f"📂 Analyzing: {pkl_file}")
        print(f"{'='*80}")
        
        # Load pickle file
        data = load_pickle_file(pkl_file)
        
        if data is None:
            continue
        
        # Determine structure
        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict):
            # Try common keys
            chunks = data.get('chunks', data.get('documents', data.get('data', [])))
        else:
            print(f"⚠️  Unknown data structure: {type(data)}")
            chunks = [data]
        
        print(f"📊 Loaded {len(chunks)} chunks")
        
        # Run analyses
        check_chunk_structure(chunks)
        articles_found, chunks_per_article = analyze_article_coverage(chunks)
        generate_article_report(articles_found, chunks_per_article)
        search_for_specific_articles(chunks, target_articles=[32, 33])
        
        # Save detailed report
        report_file = pkl_file.with_suffix('.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Report for {pkl_file}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Unique articles: {len(chunks_per_article)}\n")
            f.write(f"\nArticle distribution:\n")
            for article_num, count in sorted(chunks_per_article.items()):
                f.write(f"  Article {article_num}: {count} chunks\n")
        
        print(f"\n💾 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
