#!/usr/bin/env python3
"""
Migrate MedSight Qdrant data from local Docker to Qdrant Cloud.

This script exports your local icmr_guidelines collection and imports it
to Qdrant Cloud. Run this before deploying to Render.

Usage:
    export QDRANT_URL="https://xxx.cloud.qdrant.io:6333"
    export QDRANT_API_KEY="your-api-key"
    python scripts/migrate_to_qdrant_cloud.py
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.services.rag_engine import FASTEMBED_MODEL, EMBEDDING_DIM

# Configuration
LOCAL_QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "icmr_guidelines"
BATCH_SIZE = 100


def get_clients():
    """Initialize local and cloud Qdrant clients."""
    print("🔌 Connecting to Qdrant instances...")
    
    # Local client (no auth needed for local Docker)
    local_client = QdrantClient(url=LOCAL_QDRANT_URL)
    
    # Cloud client (requires env vars)
    cloud_url = os.getenv("QDRANT_URL")
    cloud_api_key = os.getenv("QDRANT_API_KEY")
    
    if not cloud_url:
        print("❌ ERROR: QDRANT_URL environment variable not set")
        print("   Set it to your Qdrant Cloud cluster URL")
        print("   Example: export QDRANT_URL='https://xxx.cloud.qdrant.io:6333'")
        sys.exit(1)
    
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)
    
    print(f"✅ Local Qdrant: {LOCAL_QDRANT_URL}")
    print(f"✅ Cloud Qdrant: {cloud_url[:40]}...")
    
    return local_client, cloud_client


def verify_local_collection(client):
    """Verify local collection exists and has data."""
    print(f"\n📊 Checking local collection '{COLLECTION_NAME}'...")
    
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"❌ ERROR: Collection '{COLLECTION_NAME}' not found in local Qdrant")
            print(f"   Available collections: {collection_names}")
            sys.exit(1)
        
        # Get collection info
        info = client.get_collection(COLLECTION_NAME)
        count = client.count(COLLECTION_NAME).count
        
        print(f"✅ Collection exists")
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            print("   Vectors (Named):")
            for name, params in vectors.items():
                print(f"     - {name}: size={params.size}, distance={params.distance}")
        else:
            print(f"   Vectors: size={vectors.size}, distance={vectors.distance}")
        print(f"   Documents: {count}")
        
        return count
        
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to local Qdrant: {e}")
        print("   Make sure your local Qdrant is running (docker-compose up qdrant)")
        sys.exit(1)


def setup_cloud_collection(cloud_client, local_info):
    """Create or recreate collection in Qdrant Cloud."""
    print(f"\n☁️  Setting up cloud collection '{COLLECTION_NAME}'...")
    
    try:
        # Check if collection exists
        collections = cloud_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"   Collection already exists in cloud")
            response = input("   Delete and recreate? (y/N): ").lower().strip()
            
            if response == 'y':
                print(f"   Deleting existing collection...")
                cloud_client.delete_collection(COLLECTION_NAME)
            else:
                print("   Using existing collection (data may be duplicated)")
                return
        
        # Create new collection with FastEmbed model setup to match local schema
        print(f"   Creating collection configured for FastEmbed model '{FASTEMBED_MODEL}'...")
        cloud_client.set_model(FASTEMBED_MODEL)
        cloud_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=cloud_client.get_fastembed_vector_params()
        )
        print("✅ Collection created in cloud")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to setup cloud collection: {e}")
        sys.exit(1)


def migrate_data(local_client, cloud_client, total_count):
    """Migrate all data from local to cloud."""
    print(f"\n📤 Migrating {total_count} documents...")
    
    try:
        # Scroll through all points in local collection
        offset = None
        migrated = 0
        
        while True:
            # Get batch from local
            results = local_client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=BATCH_SIZE,
                with_vectors=True,
                with_payload=True
            )
            
            points, next_offset = results
            
            if not points:
                break
            
            # Prepare points for cloud upload
            cloud_points = [
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload
                )
                for point in points
            ]
            
            # Upload to cloud
            cloud_client.upsert(
                collection_name=COLLECTION_NAME,
                points=cloud_points
            )
            
            migrated += len(points)
            offset = next_offset
            
            # Progress bar
            progress = (migrated / total_count) * 100
            bar_length = 30
            filled = int(bar_length * migrated / total_count)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r   [{bar}] {migrated}/{total_count} ({progress:.1f}%)", end="", flush=True)
            
            if offset is None:
                break
        
        print(f"\n✅ Migration complete! {migrated} documents uploaded")
        
    except Exception as e:
        print(f"\n❌ ERROR: Migration failed: {e}")
        sys.exit(1)


def verify_migration(cloud_client):
    """Verify data was migrated correctly."""
    print(f"\n🔍 Verifying migration...")
    
    try:
        cloud_count = cloud_client.count(COLLECTION_NAME).count
        
        print(f"   Cloud documents: {cloud_count}")
        
        # Test a simple search using Qdrant's FastEmbed query helper
        print("   Testing vector search...")
        results = cloud_client.query(
            collection_name=COLLECTION_NAME,
            query_text="diabetes treatment guidelines",
            limit=3
        )
        
        if results:
            print(f"✅ Search test passed! Found {len(results)} results")
            # Result objects from .query() have payload attribute directly
            first_payload = results[0].payload if hasattr(results[0], "payload") else getattr(results[0], "payload", {})
            print(f"   Top result: {first_payload.get('drug', 'N/A')[:50]}...")
        else:
            print("⚠️  Warning: Search returned no results (may be normal for small collections)")
        
        print(f"\n{'='*50}")
        print("🎉 Migration successful!")
        print(f"{'='*50}")
        print(f"\nYour data is now in Qdrant Cloud.")
        print("You can deploy to Render with these environment variables:")
        print(f"\n  QDRANT_URL={os.getenv('QDRANT_URL')}")
        print(f"  QDRANT_API_KEY={os.getenv('QDRANT_API_KEY', '[your-api-key]')}")
        
    except Exception as e:
        print(f"❌ ERROR: Verification failed: {e}")
        sys.exit(1)


def main():
    """Main migration workflow."""
    print("="*60)
    print("MedSight Qdrant Cloud Migration Tool")
    print("="*60)
    
    # Get clients
    local_client, cloud_client = get_clients()
    
    # Verify local data
    total_count = verify_local_collection(local_client)
    
    if total_count == 0:
        print("❌ ERROR: Local collection is empty!")
        sys.exit(1)
    
    # Setup cloud collection
    setup_cloud_collection(cloud_client, None)
    
    # Confirm migration
    print(f"\n⚠️  This will migrate {total_count} documents to Qdrant Cloud")
    response = input("Continue? (y/N): ").lower().strip()
    
    if response != 'y':
        print("Migration cancelled")
        sys.exit(0)
    
    # Migrate data
    migrate_data(local_client, cloud_client, total_count)
    
    # Verify
    verify_migration(cloud_client)


if __name__ == "__main__":
    main()