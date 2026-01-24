#!/usr/bin/env python3
"""
Startup script for MemEvolve API server.
"""

import os
import sys
import argparse
from pathlib import Path

# No longer needed with package structure - memevolve is installed as a package
# project_root = Path(__file__).parent.parent
# src_path = project_root / "src"
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(src_path))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set longer timeout for complex requests to reasoning models
os.environ.setdefault("MEMEVOLVE_UPSTREAM_TIMEOUT", "600")  # 10 minutes for complex requests

def main():
    """Start the MemEvolve API server."""
    parser = argparse.ArgumentParser(description="Start MemEvolve API server")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (disabled by default to reduce watchfile notices)"
    )
    args = parser.parse_args()

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected")
        print("   Consider activating your virtual environment:")
        print("   source .venv/bin/activate")

    # Note: All configuration is now handled via .env file
    # The API server will load configuration from .env or environment variables
    if not os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL"):
        print("❌ ERROR: MEMEVOLVE_UPSTREAM_BASE_URL is not set")
        print("   Please configure MEMEVOLVE_UPSTREAM_BASE_URL in your .env file")
        print("   Example: MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:11434/v1")
        sys.exit(1)

    if not os.getenv("MEMEVOLVE_MEMORY_API_KEY"):
        pass  # API key validation handled by server

    # Import and run the server
    try:
        from memevolve.api.server import app
        import uvicorn

        host = os.getenv("MEMEVOLVE_API_HOST", "127.0.0.1")
        port = int(os.getenv("MEMEVOLVE_API_PORT", "11436"))
        reload_enabled = args.reload

        if reload_enabled:
            print("🔄 Auto-reload enabled (use --reload flag)")
        else:
            print("📋 Auto-reload disabled (use --reload to enable)")



        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload_enabled,
            log_level="info"
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()