import streamlit as st
import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy import DataAPIClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="CRR RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CRR RAG Assistant")
st.subheader("Capital Requirements Regulation Analysis")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_rag_system():
    """Load the RAG system components"""
    
    # Initialize embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
     # Connect to Astra DB using DataAPIClient
    client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))
    database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
    collection = database.get_collection("crr_docling_chunks")
    
    
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
    
    def retrieve_documents(query: str, k: int = 6):
        """Retrieve relevant documents using vector search"""
        query_embedding = embeddings.embed_query(query)
        
        results = collection.find(
            sort={"$vector": query_embedding},
            limit=k,
            projection={"content": 1, "metadata": 1, "$vector": 1}
        )
        
        docs = []
        for doc in results:
            docs.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {})
            })
        return docs    
    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[{doc['metadata'].get('article_no', 'Unknown')}, Page {doc['metadata'].get('page', 'N/A')}]\n{doc['content']}"
            for doc in docs
        ])
    
    def rag_chain_invoke(question: str):
        docs = retrieve_documents(question)
        context = format_docs(docs)
        
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        
        return response.content, docs
    
    return rag_chain_invoke

# Load system
with st.spinner("Loading CRR RAG system..."):
    rag_chain = load_rag_system()

st.success("✅ CRR RAG system loaded successfully!")

# Sidebar with sample queries and clear button
st.sidebar.header("📋 Sample Queries")
sample_queries = [
    "What are the capital requirements for credit institutions?",
    "What is Common Equity Tier 1 capital?",
    "What are the large exposure requirements?",
    "How is the leverage ratio calculated?",
    "What are the disclosure requirements?"
]

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("- Ask follow-up questions")
st.sidebar.markdown("- Reference previous answers")
st.sidebar.markdown("- Chat history is maintained")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{source['article']} (Page {source['page']})**")
                    st.text(source['content'][:300] + "...")
                    if i < len(message["sources"]):
                        st.markdown("---")

# Function to process user query
def process_query(user_query):
    """Process a user query and generate response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            try:
                # Get answer from RAG chain
                answer, docs = rag_chain(user_query)
                
                # Display answer
                st.markdown(answer)
                
                # Prepare sources
                sources = [
                    {
                        "article": doc['metadata'].get('article_no', 'Unknown Article'),
                        "page": doc['metadata'].get('page', 'N/A'),
                        "content": doc['content']
                    }
                    for doc in docs
                ]
                
                # Show sources
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{source['article']} (Page {source['page']})**")
                        st.text(source['content'][:300] + "...")
                        if i < len(sources):
                            st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sample queries as buttons in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔍 Quick Questions")
for i, query in enumerate(sample_queries):
    if st.sidebar.button(query, key=f"sample_{i}"):
        process_query(query)
        st.rerun()

# Chat input at the bottom
if prompt := st.chat_input("Ask a question about Capital Requirements Regulation..."):
    process_query(prompt)