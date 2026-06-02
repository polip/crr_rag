"""
Intelligent Document Router for Multi-Document RAG System
Routes questions to appropriate documents and combines knowledge from multiple sources
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from astrapy import DataAPIClient

load_dotenv()


class DocumentRouter:
    """Routes queries to appropriate documents and combines results"""

    ARTICLE_PATTERN = re.compile(r'(?:Članak|Article)\s+(\d+)', re.IGNORECASE)

    def __init__(self):
        # Verify required environment variables
        required_vars = ["OPENAI_EMBEDD_KEY", "ASTRA_DB_TOKEN", "ASTRA_DB_API_ENDPOINT", 
                        "ASTRA_DB_COLLECTION_NAME", "OPENAI_CHAT_KEY"]
        missing_vars = [var for var in required_vars if not (os.getenv(var) or "").strip()]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_EMBEDD_KEY", "").strip()
        )

        # Connect to Astra DB with timeout
        try:
            client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))
            database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
            self.collection = database.get_collection(os.getenv("ASTRA_DB_COLLECTION_NAME"))
            print("✅ Connected to AstraDB")
        except Exception as e:
            print(f"❌ Failed to connect to AstraDB: {e}")
            raise

        # Initialize LLM for routing decisions
        self.llm = ChatOpenAI(  
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=0.1,
            api_key=os.getenv("OPENAI_CHAT_KEY", "").strip()
        )

        # Cache available documents (with timeout protection)
        print("🔄 Fetching available documents...")
        self.available_documents = self._get_available_documents()
        print(f"📚 Available documents: {list(self.available_documents.keys())}")

    def _get_available_documents(self) -> Dict[str, str]:
        """Get list of available documents from database (cached)"""
        try:
            # Use aggregation with distinct to get unique document IDs more efficiently
            # This is much faster than fetching 1000 documents
            sample_docs = list(self.collection.find(
                limit=100,  # Reduced from 1000
                projection={"metadata.document_id": 1, "metadata.document_name": 1}
            ))

            documents = {}
            for doc in sample_docs:
                metadata = doc.get("metadata", {})
                doc_id = metadata.get("document_id")
                doc_name = metadata.get("document_name")

                # Only add if both ID and name exist
                if doc_id and doc_name and doc_id not in documents:
                    documents[doc_id] = doc_name
                    
                # Early exit if we've found multiple documents
                if len(documents) >= 10:  # Reasonable upper limit
                    break

            if not documents:
                print("⚠️  No documents found in database!")
                # Return empty dict but don't crash - allows app to start
                
            return documents
        except Exception as e:
            # Log detailed error for debugging
            import traceback
            print(f"⚠️  Could not fetch documents: {e}")
            print(f"Error details: {traceback.format_exc()}")
            # Return empty dict to allow app initialization to continue
            # The app can still function, just without document routing
            return {}

    def route_query(self, question: str) -> List[str]:
        """
        Determine which document(s) should be queried based on the question
        Returns list of document IDs to query
        """
        # Create routing prompt
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document routing assistant. Given a user question and a list of available legal documents, determine which document(s) should be consulted to answer the question.

Available documents:
{document_list}

Rules:
1. Return ONLY the document ID(s), comma-separated
2. Return multiple IDs if the question requires information from multiple documents
3. Return "ALL" if the question is general or could benefit from all documents
4. Be conservative - include a document if it might be relevant

Examples:
Question: "What are capital requirements?" -> CRR
Question: "How do CRR and MiFID requirements interact?" -> CRR,MiFID_II
Question: "Compare requirements across regulations" -> ALL
"""),
            ("user", "Question: {question}\n\nWhich document(s) should be consulted?")
        ])

        # Format document list
        doc_list = "\n".join([f"- {doc_id}: {doc_name}"
                              for doc_id, doc_name in self.available_documents.items()])

        # Get routing decision
        messages = routing_prompt.format_messages(
            document_list=doc_list,
            question=question
        )
        response = self.llm.invoke(messages)
        routing_result = response.content.strip()

        # Parse routing result
        if routing_result == "ALL":
            return list(self.available_documents.keys())
        else:
            # Split by comma and clean
            doc_ids = [doc_id.strip() for doc_id in routing_result.split(",")]
            # Filter to valid document IDs
            return [doc_id for doc_id in doc_ids if doc_id in self.available_documents]

    def detect_article_number(self, query: str) -> Optional[int]:
        """Detect if query references a specific article number"""
        match = self.ARTICLE_PATTERN.search(query)
        if match:
            return int(match.group(1))
        return None

    def retrieve_documents(
        self,
        query: str,
        k: int = 6,
        document_ids: Optional[List[str]] = None,
        article_number: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents using vector search
        Optionally filter by document IDs or specific article number
        """
        import traceback, sys

        print("🔄 Generating query embedding...")
        try:
            query_embedding = self.embeddings.embed_query(query)
            print(f"✅ Embedding generated, dimension={len(query_embedding)}")
        except Exception as e:
            print(f"❌ Embedding failed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            # Try to print the chained exception if it exists
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"  Caused by: {type(e.__cause__).__name__}: {e.__cause__}")
                traceback.print_exception(type(e.__cause__), e.__cause__, e.__cause__.__traceback__, file=sys.stdout)
            raise

        # Build filter
        filter_query = {}
        if document_ids:
            filter_query["metadata.document_id"] = {"$in": document_ids}
        if article_number is not None:
            filter_query["metadata.article_number"] = article_number

        # Execute search
        print("🔄 Executing vector search...")
        try:
            results = self.collection.find(
                filter=filter_query if filter_query else None,
                sort={"$vector": query_embedding},
                limit=k,
                projection={"content": 1, "metadata": 1, "$vector": 1}
            )
        except Exception as e:
            print(f"❌ Vector search failed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            raise

        docs = []
        for doc in results:
            docs.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {})
            })
        print(f"✅ Vector search returned {len(docs)} docs")
        return docs

    def format_docs(self, docs: List[Dict]) -> str:
        """Format documents for prompt with document source information"""
        formatted = []
        for doc in docs:
            metadata = doc['metadata']
            doc_name = metadata.get('document_name', 'Unknown Document')
            article_no = metadata.get('article_no', 'Unknown Article')
            page = metadata.get('page', 'N/A')
            content = doc['content']

            formatted.append(
                f"[{doc_name} - {article_no}, Page {page}]\n{content}"
            )
        return "\n\n---\n\n".join(formatted)

    def answer_with_routing(
        self,
        question: str,
        use_routing: bool = True,
        specific_documents: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Answer question with intelligent document routing

        Args:
            question: User question
            use_routing: Whether to use automatic routing (default: True)
            specific_documents: Specific document IDs to query (overrides routing)

        Returns:
            (answer, retrieved_docs, queried_document_ids)
        """
        import traceback, sys

        # Determine which documents to query
        if specific_documents:
            doc_ids = specific_documents
            print(f"🎯 Querying specific documents: {doc_ids}")
        elif use_routing:
            try:
                doc_ids = self.route_query(question)
                print(f"🎯 Router selected documents: {doc_ids}")
            except Exception as e:
                print(f"❌ Route query failed: {e}")
                traceback.print_exc(file=sys.stdout)
                raise
        else:
            doc_ids = None  # Query all documents
            print(f"🎯 Querying all documents")

        # Detect if query references a specific article
        detected_article = self.detect_article_number(question)
        if detected_article is not None:
            print(f"📄 Detected article number: {detected_article}")

        # Retrieve relevant chunks
        try:
            docs = self.retrieve_documents(
                question,
                k=8,
                document_ids=doc_ids,
                article_number=detected_article
            )
            print(f"📄 Retrieved {len(docs)} document chunks")
        except Exception as e:
            print(f"❌ Document retrieval failed: {e}")
            traceback.print_exc(file=sys.stdout)
            raise

        # Create answer
        context = self.format_docs(docs)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized legal document assistant with expertise in financial regulations.

Your role is to:
1. Provide accurate, precise answers based solely on the provided legal document context
2. Always cite specific documents, articles, sections, or provisions when referencing information
3. When information comes from multiple documents, clearly distinguish the sources
4. Distinguish between mandatory requirements ("shall", "must") and optional provisions ("may", "should")
5. Explain complex legal concepts in clear, professional language
6. When information from multiple documents is relevant, explain how they relate or compare

Important guidelines:
- Only use information from the provided context
- Always reference specific document names and article numbers
- When citing multiple documents, organize information clearly by source
- Maintain professional, formal tone appropriate for legal documentation
- If documents provide conflicting information, note the discrepancy"""),

            ("user", """Based on the following legal document excerpts from potentially multiple sources, please answer the question:

Context: {context}

Question: {question}

Please provide a comprehensive answer with specific references to documents, articles and provisions.""")
        ])

        messages = prompt.format_messages(context=context, question=question)
        try:
            response = self.llm.invoke(messages)
            print("✅ LLM response generated")
        except Exception as e:
            print(f"❌ LLM invoke failed: {e}")
            traceback.print_exc(file=sys.stdout)
            raise

        return response.content, docs, doc_ids or list(self.available_documents.keys())

    def get_document_stats(self, use_estimated_count: bool = True) -> Dict:
        """Get statistics about documents in the database
        
        Args:
            use_estimated_count: If True, uses count_documents which is faster.
                               If False, fetches and counts (slower but accurate for small sets)
        """
        stats = {}

        for doc_id, doc_name in self.available_documents.items():
            try:
                if use_estimated_count:
                    # Try to count with a reasonable upper bound
                    # AstraDB has a limit of 1000 for count operations
                    count = self.collection.count_documents(
                        filter={"metadata.document_id": doc_id},
                        upper_bound=1000
                    )
                else:
                    # Original method: fetch and count (slower)
                    docs = list(self.collection.find(
                        filter={"metadata.document_id": doc_id},
                        projection={"_id": 1},
                        limit=1000
                    ))
                    count = len(docs)
            except Exception as e:
                # If count exceeds limit, just show "1000+"
                if "TooManyDocumentsToCountException" in str(type(e).__name__):
                    count = "1000+"
                else:
                    print(f"⚠️  Error counting docs for {doc_id}: {e}")
                    count = "Unknown"

            stats[doc_id] = {
                "name": doc_name,
                "chunk_count": count
            }

        return stats


def main():
    """Test the document router"""
    router = DocumentRouter()

    # Get document statistics
    print("\n📊 Document Statistics:")
    stats = router.get_document_stats()
    for doc_id, info in stats.items():
        print(f"  {doc_id}: {info['name']} ({info['chunk_count']} chunks)")

    # Test query
    print("\n🔍 Testing query with routing:")
    question = "What are the capital requirements for credit institutions?"
    answer, docs, doc_ids = router.answer_with_routing(question)

    print(f"\n📋 Answer:")
    print(answer)
    print(f"\n📚 Retrieved from documents: {doc_ids}")
    print(f"📄 Number of chunks retrieved: {len(docs)}")


if __name__ == "__main__":
    main()
