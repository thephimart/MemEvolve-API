#!/usr/bin/env python3
"""
Rebuild unit IDs in the vector store using content-based hashing (SHA256).

Format: unit_<16-char-hex>

This matches the encoder's ID generation, ensuring consistency across all
storage backends. Same content = same ID = automatic deduplication.

IMPORTANT: This rebuilds IDs based on content hash, not timestamps.
FAISS indices must be rebuilt since embeddings are regenerated.
"""

from memevolve.utils.embeddings import create_embedding_function
from memevolve.utils.config import ConfigManager
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_unit_id(unit: dict) -> str:
    """Generate unique unit ID using content hash (SHA256).

    Uses same technique as encoder: SHA256 hash of normalized content.
    This ensures consistent IDs across all storage backends.

    Args:
        unit: The memory unit to generate ID for

    Returns:
        Unit ID string: "unit_<16-char-hex>"
    """
    content_for_hash = {
        "type": unit.get("type", ""),
        "content": unit.get("content", ""),
        "tags": sorted(unit.get("tags", [])),
        "metadata": unit.get("metadata", {})
    }

    content_str = json.dumps(content_for_hash, sort_keys=True, default=str)

    hash_digest = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    return f"unit_{hash_digest}"


def regenerate_vector_store_ids(data_dir: str = "./data/memory"):
    """Regenerate all unit IDs in the vector store."""

    data_file = f"{data_dir}/vector.data"
    index_file = f"{data_dir}/vector.index"

    print("=" * 60)
    print("Vector Store ID Regeneration Tool")
    print("=" * 60)

    # Step 1: Load existing data
    print(f"\n[1] Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and '_next_id' in loaded:
        old_data = loaded['data']
    else:
        old_data = loaded

    old_ids = list(old_data.keys())
    print(f"    Found {len(old_ids)} units")
    print(f"    Sample old IDs: {old_ids[:3]}")

    # Step 2: Load FAISS index to get dimension
    print(f"\n[2] Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    print(f"    FAISS ntotal: {index.ntotal}")
    dimension = index.d
    print(f"    Embedding dimension: {dimension}")

    # Step 3: Initialize embedding function
    print(f"\n[3] Initializing embedding function...")
    config_manager = ConfigManager()
    embedding_config = config_manager.get("embedding")
    provider = getattr(embedding_config, 'model', '') or 'openai'
    base_url = getattr(embedding_config, 'base_url', None)
    dimension = getattr(embedding_config, 'dimension', 768)
    embedding_function = create_embedding_function(
        provider=provider,
        base_url=base_url,
        dimension=dimension,
    )
    print(f"    Using embedding provider: {provider} ({base_url})")

    # Step 4: Generate new IDs and regenerate embeddings
    print(f"\n[4] Generating content-based unit IDs and regenerating embeddings...")
    new_data = {}
    id_mapping = {}
    vectors = []

    for i, old_id in enumerate(old_ids):
        unit = old_data[old_id]

        # Generate new ID based on content hash
        new_id = generate_unit_id(unit)

        id_mapping[old_id] = new_id
        new_data[new_id] = unit
        new_data[new_id]['id'] = new_id

        # Regenerate embedding for this unit
        content = unit.get("content", "")
        if content:
            embedding = embedding_function(content)
            vectors.append(embedding)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(old_ids)} units...")

    print(f"    Generated {len(new_data)} new IDs")
    print(f"    Regenerated {len(vectors)} embeddings")
    print(f"    Sample new IDs: {list(new_data.keys())[:3]}")

    # Step 5: Rebuild FAISS index with new vectors
    print(f"\n[5] Rebuilding FAISS index...")
    new_index = faiss.IndexFlatIP(dimension)

    if vectors:
        vectors_array = np.array(vectors, dtype='float32')
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_array)
        new_index.add(vectors_array)

    print(f"    New FAISS ntotal: {new_index.ntotal}")

    # Step 6: Save new data
    print(f"\n[6] Saving new data to {data_file}...")
    save_data = {
        'data': new_data,
        '_last_retrain_size': len(new_data)
    }
    with open(data_file, 'wb') as f:
        pickle.dump(save_data, f)
    print("    Done!")

    # Step 7: Save new FAISS index
    print(f"\n[7] Saving new FAISS index to {index_file}...")
    faiss.write_index(new_index, index_file)
    print("    Done!")

    # Step 8: Report
    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print(f"Units processed: {len(old_ids)}")
    print(f"Old ID format: timestamp-based (unit_YYYYMMDDHHMMSSmmm)")
    print(f"New ID format: content-hash (unit_<16-char-hex>)")
    print(f"\nID mapping sample:")
    for old, new in list(id_mapping.items())[:3]:
        print(f"    {old} -> {new}")

    print("\n✓ FAISS index rebuilt with regenerated embeddings")
    print("✓ IDs now match encoder's content-based hashing")
    print("✓ Same content = same ID (automatic deduplication)")


if __name__ == "__main__":
    data_dir = os.environ.get('MEMEVOLVE_DATA_DIR', './data')
    memory_dir = f"{data_dir}/memory"

    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python scripts/regenerate_unit_ids.py")
        print("")
        print("Rebuilds unit IDs in the vector store using content-based hashing (SHA256).")
        print("New format: unit_<16-char-hex>")
        print("")
        print("This matches the encoder's ID generation for consistency across all")
        print("storage backends. Same content = same ID = automatic deduplication.")
        print("")
        print("IMPORTANT:")
        print("  - FAISS index is rebuilt from scratch")
        print("  - Embeddings are regenerated using the configured embedding function")
        print("  - IDs are now content-based (not timestamps)")
        sys.exit(0)

    regenerate_vector_store_ids(memory_dir)
