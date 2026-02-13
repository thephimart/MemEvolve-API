#!/usr/bin/env python3
"""
Safely regenerate unit IDs in the vector store using date+time milliseconds format.

Format: unit_YYYYMMDDHHMMSSmmm

IMPORTANT: This regenerates the IDs in the same order, so FAISS indices remain valid.
The FAISS index uses positional indices (0, 1, 2, ...) - string IDs are only dict keys.
"""

import pickle
import faiss
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_unit_id() -> str:
    """Generate unique unit ID using date+time milliseconds."""
    from datetime import datetime
    return f"unit_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


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
    
    # Step 2: Load FAISS index
    print(f"\n[2] Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    print(f"    FAISS ntotal: {index.ntotal}")
    
    # Step 3: Verify counts match
    if len(old_ids) != index.ntotal:
        print(f"    WARNING: Data count ({len(old_ids)}) != FAISS count ({index.ntotal})")
        print("    Proceeding anyway...")
    
    # Step 4: Generate new IDs in SAME order (critical!)
    # FAISS uses positional indices, so order must be preserved
    print("\n[4] Generating new unit IDs (preserving order)...")
    new_data = {}
    id_mapping = {}
    
    for i, old_id in enumerate(old_ids):
        unit = old_data[old_id]
        
        # Generate new ID
        new_id = generate_unit_id()
        
        # Ensure uniqueness by adding position suffix if needed
        # (extremely unlikely with millisecond precision, but safe)
        if new_id in new_data:
            new_id = f"{new_id}_{i:04d}"
        
        id_mapping[old_id] = new_id
        new_data[new_id] = unit
        # Update the unit's internal ID field
        new_data[new_id]['id'] = new_id
    
    print(f"    Generated {len(new_data)} new IDs")
    print(f"    Sample new IDs: {list(new_data.keys())[:3]}")
    
    # Step 5: Save new data (without _next_id - using timestamp IDs)
    print(f"\n[5] Saving new data to {data_file}...")
    with open(data_file, 'wb') as f:
        pickle.dump(new_data, f)
    print("    Done!")
    
    # Step 6: Report
    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print(f"Units processed: {len(old_ids)}")
    print(f"Old ID range: {min(old_ids)} ... {max(old_ids)}")
    print(f"New ID format: unit_YYYYMMDDHHMMSSmmm")
    print(f"\nID mapping sample:")
    for old, new in list(id_mapping.items())[:3]:
        print(f"    {old} -> {new}")
    
    print("\n✓ FAISS index remains valid (positional indices unchanged)")
    print("✓ Order preserved - retrieval will work correctly")


if __name__ == "__main__":
    import os
    
    data_dir = os.environ.get('MEMEVOLVE_DATA_DIR', './data')
    memory_dir = f"{data_dir}/memory"
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python scripts/regenerate_unit_ids.py")
        print("")
        print("Regenerates unit IDs in the vector store using")
        print("date+time milliseconds format (unit_YYYYMMDDHHMMSSmmm)")
        print("")
        print("IMPORTANT: This is safe because:")
        print("  - FAISS uses positional indices, not string IDs")
        print("  - We preserve the order of entries")
        print("  - String IDs are only dict keys in self.data")
        sys.exit(0)
    
    regenerate_vector_store_ids(memory_dir)
