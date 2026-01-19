#!/usr/bin/env python3
"""
Startup script for MemEvolve API server.
"""

import os
import sys
from pathlib import Path

# Add the project root and src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def main():
    """Start the MemEvolve API server."""
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected")
        print("   Consider activating your virtual environment:")
        print("   source .venv/bin/activate")
        print()

    # Note: All configuration is now handled via .env file
    # The API server will load configuration from .env or environment variables
    if not os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL"):
        print("ℹ️  Using default upstream URL: http://localhost:8000/v1")
        print("   Configure MEMEVOLVE_UPSTREAM_BASE_URL in .env file for production")

    if not os.getenv("MEMEVOLVE_LLM_API_KEY"):
        print("⚠️  No MEMEVOLVE_LLM_API_KEY set - memory encoding may not work")

    # Import and run the server
    try:
        from src.api.server import app
        import uvicorn

        host = os.getenv("MEMEVOLVE_API_HOST", "127.0.0.1")
        port = int(os.getenv("MEMEVOLVE_API_PORT", "8001"))
        print(f"🚀 Starting MemEvolve API server on {host}:{port}")
        print(f"   API docs available at: http://{host}:{port}/docs")
        print()

        uvicorn.run(
            "src.api.server:app",
            host=host,
            port=port,
            reload=True,
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