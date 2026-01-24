#!/usr/bin/env python3
"""
Memory Consolidate Script

Manually consolidate memory units.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.config import load_config
from memevolve.memory_system import MemorySystem


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate memory units")
    parser.add_argument(
        "--units",
        nargs="*",
        help="Specific unit IDs to consolidate (default: all)"
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

    units = args.units if args.units else None

    print(f"Consolidating units: {units or 'all'}")

    try:
        result = memory_system.manage_memory("consolidate", units=units)
        print(f"Consolidated to {len(result)} units")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()