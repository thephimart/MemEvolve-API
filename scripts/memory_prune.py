#!/usr/bin/env python3
"""
Memory Prune Script

Manually prune memory units based on criteria.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.config import load_config
from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Memory prune script initialized")
from memevolve.memory_system import MemorySystem


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Prune memory units")
    parser.add_argument(
        "--max-count",
        type=int,
        help="Keep only the most recent N units"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        help="Remove units older than N days"
    )
    parser.add_argument(
        "--type",
        help="Remove units of specific type"
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

    # Determine criteria
    criteria = {}
    if args.max_count:
        criteria["max_count"] = args.max_count
    if args.max_age:
        criteria["max_age"] = args.max_age
    if args.type:
        criteria["type"] = args.type

    if not criteria:
        print("No criteria specified. Use --max-count, --max-age, or --type")
        sys.exit(1)

    print(f"Pruning with criteria: {criteria}")

    try:
        result = memory_system.manage_memory("prune", criteria=criteria)
        print(f"Pruned {result} units")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()