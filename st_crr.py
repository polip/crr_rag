import streamlit as st
import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy import DataAPIClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from document_router import DocumentRouter


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("CRR and CRD RAG Assistant")
st.subheader("Financial EU Regulations Analysis with Intelligent Document Routing")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_mode" not in st.session_state:
    st.session_state.document_mode = "auto"  # auto, all, or specific document ID

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []

@st.cache_resource
def load_rag_system():
    """Load the RAG system with multi-document support"""
    try:
        router = DocumentRouter()
        return router
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        st.stop()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_document_statistics(_router):
    """Get document statistics - cached to avoid repeated DB queries"""
    try:
        return _router.get_document_stats(use_estimated_count=True)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch document statistics: {e}")
        return {}

# Load system
with st.spinner("Loading Multi-Document RAG system..."):
    router = load_rag_system()

# Get available documents and stats (cached)
doc_stats = get_document_statistics(router)

if doc_stats:
    st.success(f"✅ RAG system loaded! {len(doc_stats)} document(s) available")
else:
    st.warning("⚠️ RAG system loaded but no documents found in database. Please check your data!")

# Sidebar - Document Selection
st.sidebar.header("📚 Document Selection")

# Display available documents
st.sidebar.markdown("**Available Documents:**")
for doc_id, info in doc_stats.items():
    st.sidebar.markdown(f"- **{info['name']}** ({info['chunk_count']} chunks)")

st.sidebar.markdown("---")

# Document mode selector
document_mode = st.sidebar.radio(
    "Query Mode:",
    ["🤖 Auto-Route (AI selects)", "🌐 All Documents", "🎯 Specific Documents"],
    index=0
)

# Update session state
if document_mode == "🤖 Auto-Route (AI selects)":
    st.session_state.document_mode = "auto"
elif document_mode == "🌐 All Documents":
    st.session_state.document_mode = "all"
else:
    st.session_state.document_mode = "specific"
    # Show document selector
    st.session_state.selected_documents = st.sidebar.multiselect(
        "Select documents to query:",
        options=list(doc_stats.keys()),
        format_func=lambda x: doc_stats[x]['name'],
        default=st.session_state.selected_documents if st.session_state.selected_documents else list(doc_stats.keys())[:1]
    )

st.sidebar.markdown("---")

# Sample queries
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
st.sidebar.markdown("- **Auto-Route**: AI automatically selects relevant documents")
st.sidebar.markdown("- **All Documents**: Search across all documents")
st.sidebar.markdown("- **Specific**: Choose which documents to query")
st.sidebar.markdown("- Ask follow-up questions")
st.sidebar.markdown("- Chat history is maintained")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show document routing info if available
        if "queried_documents" in message:
            doc_names = [doc_stats[doc_id]['name'] for doc_id in message["queried_documents"] if doc_id in doc_stats]
            if doc_names:
                st.caption(f"📚 Queried: {', '.join(doc_names)}")

        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    doc_name = source.get('document', 'Unknown Document')
                    article = source.get('article', 'Unknown Article')
                    page = source.get('page', 'N/A')
                    st.markdown(f"**{doc_name}**")
                    st.markdown(f"*{article} (Page {page})*")
                    st.text(source['content'][:300] + "...")
                    if i < len(message["sources"]):
                        st.markdown("---")

# Function to process user query
def process_query(user_query):
    """Process a user query and generate response with multi-document support"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing EU legal documents..."):
            try:
                # Determine which documents to query based on mode
                if st.session_state.document_mode == "auto":
                    # Use automatic routing
                    answer, docs, queried_doc_ids = router.answer_with_routing(
                        user_query,
                        use_routing=True
                    )
                    routing_info = f"📚 Queried: {', '.join([doc_stats[doc_id]['name'] for doc_id in queried_doc_ids if doc_id in doc_stats])}"
                    st.caption(routing_info)

                elif st.session_state.document_mode == "all":
                    # Query all documents
                    answer, docs, queried_doc_ids = router.answer_with_routing(
                        user_query,
                        use_routing=False
                    )
                    st.caption("📚 Queried: All documents")

                else:
                    # Query specific documents
                    if not st.session_state.selected_documents:
                        st.warning("⚠️ Please select at least one document to query")
                        return

                    answer, docs, queried_doc_ids = router.answer_with_routing(
                        user_query,
                        specific_documents=st.session_state.selected_documents
                    )
                    routing_info = f"📚 Queried: {', '.join([doc_stats[doc_id]['name'] for doc_id in queried_doc_ids if doc_id in doc_stats])}"
                    st.caption(routing_info)

                # Display answer
                st.markdown(answer)

                # Prepare sources
                sources = [
                    {
                        "document": doc['metadata'].get('document_name', 'Unknown Document'),
                        "article": doc['metadata'].get('article_no', 'Unknown Article'),
                        "page": doc['metadata'].get('page', 'N/A'),
                        "content": doc['content']
                    }
                    for doc in docs
                ]

                # Show sources
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{source['document']}**")
                        st.markdown(f"*{source['article']} (Page {source['page']})*")
                        st.text(source['content'][:300] + "...")
                        if i < len(sources):
                            st.markdown("---")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "queried_documents": queried_doc_ids
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