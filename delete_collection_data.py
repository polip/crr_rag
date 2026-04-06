"""
Delete documents from AstraDB collection
"""

import os
import sys
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()


def delete_all_documents(collection, confirm: bool = True):
    """Delete all documents from the collection"""
    
    if confirm:
        print("\n⚠️  WARNING: This will delete ALL documents from the collection!")
        response = input("Type 'DELETE ALL' to confirm: ")
        if response != "DELETE ALL":
            print("❌ Operation cancelled.")
            return 0
    
    print("\n🗑️  Deleting all documents...")
    
    try:
        # Delete all documents
        result = collection.delete_many(filter={})
        deleted_count = result.deleted_count
        
        print(f"✅ Successfully deleted {deleted_count} document(s)")
        return deleted_count
        
    except Exception as e:
        print(f"❌ Error during deletion: {str(e)}")
        return 0


def delete_by_document_id(collection, document_id: str, confirm: bool = True):
    """Delete documents with a specific document_id"""
    
    # First, count how many documents will be deleted
    try:
        count_result = collection.count_documents(
            filter={"metadata.document_id": document_id},
            upper_bound=10000
        )
        doc_count = count_result
        
        if doc_count == 0:
            print(f"\n⚠️  No documents found with document_id: {document_id}")
            return 0
        
        print(f"\n📊 Found {doc_count} document(s) with document_id: {document_id}")
        
    except Exception as e:
        print(f"❌ Error counting documents: {str(e)}")
        return 0
    
    if confirm:
        print(f"\n⚠️  WARNING: This will delete {doc_count} document(s)!")
        response = input(f"Type 'DELETE {document_id}' to confirm: ")
        if response != f"DELETE {document_id}":
            print("❌ Operation cancelled.")
            return 0
    
    print(f"\n🗑️  Deleting documents with document_id: {document_id}...")
    
    try:
        # Delete documents with specific document_id
        result = collection.delete_many(
            filter={"metadata.document_id": document_id}
        )
        deleted_count = result.deleted_count
        
        print(f"✅ Successfully deleted {deleted_count} document(s)")
        return deleted_count
        
    except Exception as e:
        print(f"❌ Error during deletion: {str(e)}")
        return 0


def list_document_ids(collection):
    """List all unique document_ids in the collection"""
    
    print("\n📋 Fetching unique document IDs...")
    
    try:
        # Get sample of documents to find unique document_ids
        documents = list(collection.find(
            projection={"metadata.document_id": 1, "metadata.document_name": 1},
            limit=1000
        ))
        
        # Extract unique document_ids
        doc_ids = {}
        for doc in documents:
            if "metadata" in doc and "document_id" in doc["metadata"]:
                doc_id = doc["metadata"]["document_id"]
                doc_name = doc["metadata"].get("document_name", "Unknown")
                if doc_id not in doc_ids:
                    doc_ids[doc_id] = doc_name
        
        if not doc_ids:
            print("⚠️  No documents with document_id metadata found")
            return []
        
        print(f"\n✅ Found {len(doc_ids)} unique document ID(s):\n")
        print("-" * 60)
        for doc_id, doc_name in sorted(doc_ids.items()):
            # Count documents for this ID
            try:
                count = collection.count_documents(
                    filter={"metadata.document_id": doc_id},
                    upper_bound=10000
                )
                print(f"  • {doc_id}")
                print(f"    Name: {doc_name}")
                print(f"    Chunks: {count}")
                print("-" * 60)
            except Exception as e:
                print(f"  • {doc_id} (error counting: {str(e)[:30]})")
                print("-" * 60)
        
        return list(doc_ids.keys())
        
    except Exception as e:
        print(f"❌ Error listing document IDs: {str(e)}")
        return []


def main():
    """Main function"""
    
    print("\n" + "=" * 60)
    print("🗑️  ASTRADB COLLECTION DATA DELETION")
    print("=" * 60 + "\n")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python delete_collection_data.py list")
        print("  python delete_collection_data.py all")
        print("  python delete_collection_data.py <document_id>")
        print("\nOptions:")
        print("  list           - List all document IDs in the collection")
        print("  all            - Delete ALL documents from the collection")
        print("  <document_id>  - Delete documents with specific document_id")
        print("\nExamples:")
        print("  python delete_collection_data.py list")
        print("  python delete_collection_data.py CRR")
        print("  python delete_collection_data.py all")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        # Connect to Astra DB
        print("🔗 Connecting to AstraDB...")
        client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))
        database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
        
        collection_name = os.getenv("ASTRA_DB_COLLECTION_NAME")
        if not collection_name:
            print("❌ Error: ASTRA_DB_COLLECTION_NAME not set in .env")
            sys.exit(1)
        
        print(f"📁 Collection: {collection_name}")
        collection = database.get_collection(collection_name)
        
        # Execute command
        if command == "list":
            list_document_ids(collection)
            
        elif command == "all":
            deleted = delete_all_documents(collection, confirm=True)
            print(f"\n{'=' * 60}")
            print(f"✅ DELETION COMPLETE")
            print(f"{'=' * 60}")
            print(f"🗑️  Deleted: {deleted} document(s)")
            print(f"{'=' * 60}\n")
            
        else:
            # Treat as document_id
            document_id = command
            deleted = delete_by_document_id(collection, document_id, confirm=True)
            print(f"\n{'=' * 60}")
            print(f"✅ DELETION COMPLETE")
            print(f"{'=' * 60}")
            print(f"🆔 Document ID: {document_id}")
            print(f"🗑️  Deleted: {deleted} document(s)")
            print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Check your AstraDB credentials in .env file:")
        print("   - ASTRA_DB_TOKEN")
        print("   - ASTRA_DB_API_ENDPOINT")
        print("   - ASTRA_DB_COLLECTION_NAME")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
