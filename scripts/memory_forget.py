#!/usr/bin/env python3
"""
Memory Forget Script

Manually apply forgetting mechanism to memory units.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.config import load_config
from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Memory forget script initialized")
from memevolve.memory_system import MemorySystem


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Apply forgetting to memory units")
    parser.add_argument(
        "--strategy",
        default="lru",
        choices=["lru", "random"],
        help="Forgetting strategy (default: lru)"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of units to forget (default: 10% of total)"
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

    print(f"Forgetting with strategy: {args.strategy}, count: {args.count or 'auto'}")

    try:
        result = memory_system.manage_memory(
            "forget",
            strategy=args.strategy,
            count=args.count
        )
        print(f"Forgot {result} units")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()