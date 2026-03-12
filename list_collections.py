"""
List all available collections in AstraDB database
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()


def main():
    """List all collections in the AstraDB database"""
    
    print("\n" + "=" * 60)
    print("📚 ASTRADB COLLECTIONS")
    print("=" * 60 + "\n")
    
    try:
        # Connect to Astra DB
        print("🔗 Connecting to AstraDB...")
        client = DataAPIClient(os.getenv("ASTRA_DB_TOKEN"))
        database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
        
        # Get all collections
        print("📋 Fetching collections...\n")
        collections = database.list_collection_names()
        
        if not collections:
            print("⚠️  No collections found in this database.")
            print("\n💡 You may need to create a collection first.")
        else:
            print(f"✅ Found {len(collections)} collection(s):\n")
            print("-" * 60)
            
            for i, collection_name in enumerate(collections, 1):
                print(f"  {i}. {collection_name}")
                
                # Try to get collection info
                try:
                    collection = database.get_collection(collection_name)
                    # Sample a few documents to get count estimate
                    sample = list(collection.find(limit=1, projection={"_id": 1}))
                    
                    if sample:
                        print(f"     └─ Status: Active (contains documents)")
                    else:
                        print(f"     └─ Status: Empty")
                except Exception as e:
                    print(f"     └─ Status: Unknown (error: {str(e)[:50]})")
                
                print("-" * 60)
            
            # Highlight current collection
            current_collection = os.getenv("ASTRA_DB_COLLECTION_NAME")
            if current_collection:
                print(f"\n🎯 Current collection in .env: {current_collection}")
                if current_collection in collections:
                    print(f"   ✅ Collection exists")
                else:
                    print(f"   ⚠️  Collection NOT found in database!")
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Check your AstraDB credentials in .env file:")
        print("   - ASTRA_DB_TOKEN")
        print("   - ASTRA_DB_API_ENDPOINT")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
