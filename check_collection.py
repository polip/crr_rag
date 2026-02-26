"""
Check what documents are in the AstraDB collection
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

# Connect to Astra DB
client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))
database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
collection = database.get_collection(os.getenv("ASTRA_DB_COLLECTION_NAME"))

print("🔍 Checking AstraDB collection...\n")

# Sample documents to find unique document IDs
sample_docs = list(collection.find( projection={"metadata": 1}))

print(f"📊 Total documents sampled: {len(sample_docs)}\n")

# Analyze documents
documents = {}
for doc in sample_docs:
    metadata = doc.get("metadata", {})
    doc_id = metadata.get("document_id")
    doc_name = metadata.get("document_name")
    
    if doc_id:
        if doc_id not in documents:
            documents[doc_id] = {
                "name": doc_name or "Unknown",
                "count": 0
            }
        documents[doc_id]["count"] += 1

print("📚 Documents found in collection:")
print("=" * 60)
for doc_id, info in documents.items():
    print(f"  Document ID: {doc_id}")
    print(f"  Name: {info['name']}")
    print(f"  Chunks: {info['count']}")
    print("-" * 60)

if not documents:
    print("⚠️  No documents found in collection!")
