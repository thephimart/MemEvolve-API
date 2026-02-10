#!/usr/bin/env python3
"""
Memory Deduplicate Script

Manually deduplicate memory units based on content similarity.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.config import load_config
from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Memory deduplicate script initialized")
from memevolve.memory_system import MemorySystem


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate memory units")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold (default: 0.9)"
    )
    parser.add_argument(
        "--config",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize memory system
    memory_system = MemorySystem(config)

    print(f"Deduplicating with threshold: {args.threshold}")

    try:
        result = memory_system.manage_memory(
            "deduplicate",
            similarity_threshold=args.threshold
        )
        print(f"Removed {result} duplicate units")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()