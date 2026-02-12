#!/usr/bin/env python3
"""Test script to verify console DEBUG filtering is working."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memevolve.utils.logging_manager import LoggingManager

def test_console_debug_filtering():
    """Test that DEBUG messages are filtered from console but appear in files."""
    
    print("=== Testing Console DEBUG Filtering ===\n")
    
    # Create test logger
    logger = LoggingManager.get_logger(__name__)
    
    print("1. Testing DEBUG message (should NOT appear in console):")
    logger.debug("🔍 DEBUG: This should only appear in file, not console")
    
    print("\n2. Testing INFO message (should appear in both):")
    logger.info("ℹ️ INFO: This should appear in both console and file")
    
    print("\n=== Verification ===")
    print("If you only see the INFO message above, console filtering is working correctly.")
    print("DEBUG messages will be in: ./logs/__main__.log")
    print("Console should show: INFO, WARNING, ERROR (not DEBUG)")

if __name__ == "__main__":
    test_console_debug_filtering()