
import os
import sys
import pickle
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


class PickleUploader:
    """Upload chunks from pickle files to vector database"""

    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens
        self.safe_token_limit = int(max_tokens * 0.85)  # 870 tokens - safety margin
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2800,  # Slightly smaller for re-splitting
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_EMBEDD_KEY")
        )
        
        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=os.getenv("ASTRA_DB_COLLECTION_NAME"),
            token=os.getenv("ASTRA_DB_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        print("✅ Connected to Astra DB")

    @staticmethod
    def estimate_tokens(text: str) -> float:
        """Rough estimation: 1 token ≈ 4 characters"""
        return len(text) / 4

    def load_from_pickle(self, pickle_path: str) -> List[Document]:
        """Load chunks from pickle file"""
        print(f"\n📂 Loading chunks from {pickle_path}...")
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"✅ Loaded {len(chunks)} chunks from pickle file")
        return chunks

    def validate_and_split_chunks(self, chunks: List[Document]) -> List[Document]:
        """Validate chunks and split any that exceed token limit"""
        print(f"\n🔧 Validating chunks against {self.max_tokens} token limit (safety threshold: {self.safe_token_limit})...")
        
        valid_chunks = []
        oversized_count = 0
        
        for chunk in chunks:
            estimated_tokens = self.estimate_tokens(chunk.page_content)
            
            if estimated_tokens <= self.safe_token_limit:
                valid_chunks.append(chunk)
            else:
                oversized_count += 1
                print(f"  ⚠️  Splitting chunk with ~{int(estimated_tokens)} tokens...")
                
                # Split the oversized chunk
                sub_chunks = self.text_splitter.split_documents([chunk])
                
                for j, sub_chunk in enumerate(sub_chunks):
                    # Preserve original metadata
                    sub_chunk.metadata = chunk.metadata.copy()
                    sub_chunk.metadata['sub_chunk'] = j + 1
                    sub_chunk.metadata['total_sub_chunks'] = len(sub_chunks)
                    sub_chunk.metadata['original_chunk_tokens'] = int(estimated_tokens)
                    sub_chunk.metadata['auto_split'] = True
                    
                    valid_chunks.append(sub_chunk)
        
        print(f"✅ Validation complete:")
        print(f"   - {len(chunks) - oversized_count} chunks within limit")
        print(f"   - {oversized_count} chunks split")
        print(f"   - Final total: {len(valid_chunks)} chunks")
        
        return valid_chunks

    def upload_chunks(self, chunks: List[Document], batch_size: int = 25):
        """Upload chunks to vector store in batches"""
        print(f"\n📚 Uploading {len(chunks)} chunks to Astra DB...")
        
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
                print(f"  ✅ Batch {batch_num} uploaded successfully")
            except Exception as e:
                print(f"  ❌ Batch {batch_num} failed: {str(e)[:100]}")
                failed_chunks.extend(batch)
        
        print(f"\n📊 Upload Results:")
        print(f"✅ Successfully uploaded: {successfully_added} chunks")
        print(f"❌ Failed: {len(failed_chunks)} chunks")
        
        return successfully_added, failed_chunks


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python upload_from_pickle.py <pickle_file_path> [--skip-validation]")
        print("\nOptions:")
        print("  --skip-validation    Skip token validation and upload chunks as-is")
        print("\nExamples:")
        print("  python upload_from_pickle.py pickle_files/chunks_CRR_20260306_143022.pkl")
        print("  python upload_from_pickle.py pickle_files/chunks_CRR_20260306_143022.pkl --skip-validation")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    skip_validation = "--skip-validation" in sys.argv
    
    print(f"\n{'='*60}")
    print(f"📦 UPLOAD FROM PICKLE")
    print(f"{'='*60}")
    print(f"📂 Pickle file: {pickle_path}")
    print(f"🔧 Validation: {'DISABLED' if skip_validation else 'ENABLED'}")
    print(f"{'='*60}\n")
    
    try:
        uploader = PickleUploader()
        
        # Load chunks
        chunks = uploader.load_from_pickle(pickle_path)
        
        # Validate and split if needed (unless skipped)
        if skip_validation:
            print("\n⚠️  Skipping validation - uploading chunks as-is")
            validated_chunks = chunks
        else:
            validated_chunks = uploader.validate_and_split_chunks(chunks)
        
        # Upload to vector store
        successfully_added, failed_chunks = uploader.upload_chunks(validated_chunks)
        
        print(f"\n{'='*60}")
        print(f"✅ UPLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"📂 Source: {pickle_path}")
        print(f"📦 Total chunks processed: {len(validated_chunks)}")
        print(f"✅ Successfully uploaded: {successfully_added}")
        print(f"❌ Failed: {len(failed_chunks)}")
        print(f"{'='*60}\n")
        
        if failed_chunks:
            print(f"⚠️  Some chunks failed to upload. Check error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
