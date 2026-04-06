"""
Multi-document processor for RAG system
Processes PDF regulation documents and stores them in Astra DB with proper metadata
"""

import os
import re
import pickle
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()


class DocumentProcessor:
    """Process legal PDF documents for RAG system"""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        # Use safety threshold - only accept chunks well below the limit
        self.safe_token_limit = int(max_tokens * 0.85)  # 435 tokens - safety margin
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,  # ~400 tokens with conservative estimate
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        self.article_pattern = re.compile(r'^(Članak|Article)\s+(\d+)', re.IGNORECASE | re.MULTILINE)

    @staticmethod
    def estimate_tokens(text: str) -> float:
        """Rough estimation: 1 token ≈ 4 characters"""
        return len(text) / 4

    def extract_articles_from_pdf(self, pdf_path: str, document_id: str, document_name: str) -> List[Document]:
        """Extract articles from PDF using PyMuPDF"""
        chunks = []
        current = None

        print(f"📄 Extracting articles from {document_name}...")

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Split into lines and process
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Check if this is an article heading
                article_match = self.article_pattern.match(line)

                if article_match:
                    article_type = article_match.group(1)  # "Članak" or "Article"
                    article_num = article_match.group(2)   # The number

                    # Save previous chunk before starting a new one
                    if current:
                        chunks.append(Document(**current))

                    # Start new article chunk
                    current = {
                        "page_content": line + "\n",
                        "metadata": {
                            "type": "article",
                            "article_no": f"{article_type} {article_num}",
                            "article_number": int(article_num),
                            "page": page_num + 1,  # Convert to 1-indexed
                            "language": "hr" if article_type == "Članak" else "en",
                            "document_id": document_id,
                            "document_name": document_name,
                        }
                    }
                elif current:
                    # Append content to current article
                    current["page_content"] += line + "\n"

        # Close the document
        doc.close()

        # Don't forget the last chunk
        if current:
            chunks.append(Document(**current))

        print(f"✅ Extracted {len(chunks)} articles")
        return chunks

    def split_large_chunks(self, chunks: List[Document]) -> List[Document]:
        """Split chunks that exceed token limit and re-detect article numbers"""
        print(f"🔧 Checking chunks for {self.max_tokens} token limit (using safety threshold: {self.safe_token_limit})...")

        valid_chunks = []
        oversized_count = 0
        redetected_count = 0

        for i, chunk in enumerate(chunks):
            estimated_tokens = self.estimate_tokens(chunk.page_content)

            # Use safety threshold to ensure chunks are well below limit
            if estimated_tokens <= self.safe_token_limit:
                valid_chunks.append(chunk)
            else:
                oversized_count += 1
                original_article = chunk.metadata.get('article_number', 'unknown')

                # Split the oversized chunk
                sub_chunks = self.text_splitter.split_documents([chunk])

                for j, sub_chunk in enumerate(sub_chunks):
                    # Start with original metadata
                    sub_chunk.metadata = chunk.metadata.copy()
                    sub_chunk.metadata['sub_chunk'] = j + 1
                    sub_chunk.metadata['total_sub_chunks'] = len(sub_chunks)
                    sub_chunk.metadata['original_chunk_tokens'] = int(estimated_tokens)

                    # Re-detect article number in this specific sub-chunk
                    first_300_chars = sub_chunk.page_content[:300].strip()
                    article_match = self.article_pattern.search(first_300_chars)

                    if article_match:
                        article_type = article_match.group(1)
                        article_num = article_match.group(2)
                        detected_article_number = int(article_num)

                        if detected_article_number != original_article:
                            sub_chunk.metadata['article_no'] = f"{article_type} {article_num}"
                            sub_chunk.metadata['article_number'] = detected_article_number
                            sub_chunk.metadata['language'] = "hr" if article_type == "Članak" else "en"
                            sub_chunk.metadata['article_redetected'] = True
                            redetected_count += 1

                    valid_chunks.append(sub_chunk)

        print(f"✅ Processed {len(chunks)} chunks:")
        print(f"   - {len(chunks) - oversized_count} within limit")
        print(f"   - {oversized_count} split into sub-chunks")
        print(f"   - {redetected_count} articles re-detected")
        print(f"   - Final total: {len(valid_chunks)} chunks")

        return valid_chunks

    def process_pdf(self, pdf_path: str, document_id: str, document_name: str) -> List[Document]:
        """Process a PDF document and return chunks"""
        print(f"\n{'='*60}")
        print(f"📄 Processing: {pdf_path}")
        print(f"🆔 Document ID: {document_id}")
        print(f"📝 Document Name: {document_name}")
        print(f"{'='*60}\n")

        # Extract articles directly from PDF
        print("🔄 Extracting text from PDF with PyMuPDF...")
        chunks = self.extract_articles_from_pdf(pdf_path, document_id, document_name)

        # Split large chunks
        valid_chunks = self.split_large_chunks(chunks)

        return valid_chunks

    def save_to_pickle(self, chunks: List[Document], document_id: str):
        """Save chunks to pickle file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_filename = f"pickle_files/chunks_{document_id}_{timestamp}.pkl"

        # Ensure pickle_files directory exists
        os.makedirs("pickle_files", exist_ok=True)

        print(f"\n💾 Saving {len(chunks)} chunks to {pickle_filename}...")
        
        # Create clean Document objects without any non-serializable references
        clean_chunks = [
            Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata.copy()
            )
            for chunk in chunks
        ]
        
        with open(pickle_filename, 'wb') as f:
            pickle.dump(clean_chunks, f)

        print(f"✅ Chunks saved!")
        return pickle_filename


class VectorStoreManager:
    """Manage Astra DB vector store operations"""

    def __init__(self):
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=os.getenv("NVIDIA_API_KEY")
        )

        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=os.getenv("ASTRA_DB_COLLECTION_NAME"),
            token=os.getenv("ASTRA_DB_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        print("✅ Connected to Astra DB")

    def add_chunks(self, chunks: List[Document], batch_size: int = 25):
        """Add chunks to vector store in batches"""
        print(f"\n📚 Adding {len(chunks)} chunks to Astra DB...")

        total_batches = (len(chunks) + batch_size - 1) // batch_size
        successfully_added = 0
        failed_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                self.vectorstore.add_documents(batch)
                successfully_added += len(batch)
                print(f"  ✅ Batch {batch_num} added successfully")
            except Exception as e:
                print(f"  ❌ Batch {batch_num} failed: {str(e)[:100]}")
                failed_chunks.extend(batch)

        print(f"\n📊 Results:")
        print(f"✅ Successfully added: {successfully_added} chunks")
        print(f"❌ Failed: {len(failed_chunks)} chunks")

        return successfully_added, failed_chunks


def main():
    """Main function to process a new document"""
    import sys

    # Check for --pickle-only flag
    pickle_only = "--pickle-only" in sys.argv
    if pickle_only:
        sys.argv.remove("--pickle-only")

    if len(sys.argv) < 4:
        print("Usage: python process_document.py <pdf_path> <document_id> <document_name> [--pickle-only]")
        print("\nOptions:")
        print("  --pickle-only    Process document and save to pickle file only (skip vector DB upload)")
        print("\nExamples:")
        print("  python process_document.py new_regulation.pdf MiFID_II 'MiFID II Regulation'")
        print("  python process_document.py new_regulation.pdf MiFID_II 'MiFID II Regulation' --pickle-only")
        sys.exit(1)

    pdf_path = sys.argv[1]
    document_id = sys.argv[2]
    document_name = sys.argv[3]

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)

    # Process document
    processor = DocumentProcessor()
    chunks = processor.process_pdf(pdf_path, document_id, document_name)

    # Save to pickle
    pickle_file = processor.save_to_pickle(chunks, document_id)

    # Add to vector store (unless --pickle-only flag is used)
    if pickle_only:
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE (PICKLE ONLY)")
        print(f"{'='*60}")
        print(f"📄 Document: {document_name}")
        print(f"🆔 ID: {document_id}")
        print(f"📦 Chunks created: {len(chunks)}")
        print(f"💾 Backup saved to: {pickle_file}")
        print(f"{'='*60}")
        print(f"\n💡 To upload to vector database later, use upload_from_pickle.py")
        print(f"{'='*60}\n")
    else:
        store_manager = VectorStoreManager()
        successfully_added, failed_chunks = store_manager.add_chunks(chunks)

        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📄 Document: {document_name}")
        print(f"🆔 ID: {document_id}")
        print(f"📦 Chunks created: {len(chunks)}")
        print(f"✅ Successfully stored: {successfully_added}")
        print(f"💾 Backup saved to: {pickle_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
