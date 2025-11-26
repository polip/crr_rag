import streamlit as st
import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="CRR RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CRR RAG Assistant")
st.subheader("Capital Requirements Regulation Analysis")

@st.cache_resource
def load_rag_system():
    """Load the RAG system components"""
    
    # Initialize embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Connect to vector store
    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="legal_docling_chunks",
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized legal document assistant with expertise in financial regulations, particularly the Capital Requirements Regulation (CRR). 

Your role is to:
1. Provide accurate, precise answers based solely on the provided legal document context
2. Always cite specific articles, sections, or provisions when referencing information
3. Distinguish between mandatory requirements ("shall", "must") and optional provisions ("may", "should")
4. Explain complex legal concepts in clear, professional language
5. When uncertain, clearly state limitations and suggest consulting legal counsel

Important guidelines:
- Only use information from the provided context
- Never speculate or provide general legal advice
- Always reference specific article numbers when applicable
- Maintain professional, formal tone appropriate for legal documentation"""),
    
    ("user", """Based on the following legal document excerpts, please answer the question:

Context: {context}

Question: {question}

Please provide a comprehensive answer with specific references to articles and provisions.""")
])
    
    # Create retriever and format function
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[{doc.metadata.get('article_no', 'Unknown')}, Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Load system
with st.spinner("Loading CRR RAG system..."):
    rag_chain, retriever = load_rag_system()

st.success("✅ CRR RAG system loaded successfully!")

# Sample queries
st.sidebar.header("📋 Sample Queries")
sample_queries = [
    "What are the capital requirements for credit institutions?",
    "What is Common Equity Tier 1 capital?",
    "What are the large exposure requirements?",
    "How is the leverage ratio calculated?",
    "What are the disclosure requirements?"
]

selected_query = st.sidebar.selectbox("Choose a sample query:", [""] + sample_queries)

# Main input
question = st.text_area(
    "Enter your legal question:",
    value=selected_query,
    height=100,
    placeholder="e.g., What are the minimum capital requirements for banks?"
)

# Search button
if st.button("🔍 Search & Answer", type="primary"):
    if question:
        with st.spinner("Searching legal documents..."):
            try:
                # Get answer
                answer = rag_chain.invoke(question)
                
                
                # Display results
                st.subheader("📋 Legal Analysis")
                st.write(answer)
                
                # Show source documents
                with st.expander("📄 Source Documents"):
                    docs = retriever.get_relevant_documents(question)
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**{doc.metadata.get('article_no', 'Unknown Article')} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.write(doc.page_content[:300] + "...")
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Error: {e}")
                
    else:
        st.warning("Please enter a question.")