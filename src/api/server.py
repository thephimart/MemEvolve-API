"""
MemEvolve API Server - Memory-enhanced proxy for OpenAI-compatible APIs

This server acts as a transparent proxy that integrates MemEvolve memory
functionality with any OpenAI-compatible LLM API endpoint.
"""

import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..memory_system import MemorySystem
from utils.config import load_config, MemEvolveConfig
from .routes import router
from .middleware import MemoryMiddleware
from .evolution_manager import EvolutionManager


class ProxyConfig(BaseModel):
    """Configuration for the API proxy."""
    upstream_base_url: str
    upstream_api_key: Optional[str] = None
    memory_config: Optional[Dict[str, Any]] = None


# Global variables for lifespan management
memory_system: Optional[MemorySystem] = None
proxy_config: Optional[ProxyConfig] = None
http_client: Optional[httpx.AsyncClient] = None
memory_middleware: Optional[MemoryMiddleware] = None
evolution_manager: Optional[EvolutionManager] = None


# Global reference for memory system
_memory_system_instance = None


def get_memory_system():
    """Get the global memory system instance."""
    return _memory_system_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global memory_system, proxy_config, http_client, memory_middleware, evolution_manager, _memory_system_instance

    # Startup
    try:
        # Load configuration
        config = load_config()

        # Validate required configuration
        if not config or not config.api.upstream_base_url:
            raise ValueError(
                "MEMEVOLVE_UPSTREAM_BASE_URL must be configured in .env file")

        # Check if memory integration is enabled
        memory_integration_enabled = config.api.memory_integration if config else True

        # Initialize memory system
        if memory_integration_enabled:
            memory_system = MemorySystem(
                config) if config else MemorySystem(MemEvolveConfig())
        else:
            memory_system = None

        # Store global reference
        _memory_system_instance = memory_system

        # Initialize proxy config
        proxy_config = ProxyConfig(
            upstream_base_url=config.api.upstream_base_url,
            upstream_api_key=config.api.upstream_api_key if config else None,
            memory_config=None  # Not needed since we use the config object directly
        )

        # Initialize memory middleware if enabled
        memory_middleware = (
            MemoryMiddleware(memory_system, evolution_manager)
            if memory_integration_enabled and memory_system else None
        )

        # Initialize evolution manager if enabled
        if config.evolution.enable and memory_system:
            evolution_manager = EvolutionManager(config, memory_system)
            evolution_manager.start_evolution()
        else:
            evolution_manager = None

        # Initialize HTTP client
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True
        )

        print("✅ MemEvolve API server started successfully")
        print(f"   Upstream: {proxy_config.upstream_base_url}")
        memory_status = (
            'Enabled' if memory_system and memory_integration_enabled
            else 'Disabled'
        )
        print(f"   Memory: {memory_status}")
        print(
            f"   Memory Integration: {'Enabled' if memory_middleware else 'Disabled'}")

    except Exception as e:
        print(f"❌ Failed to initialize MemEvolve API server: {e}")
        raise

    yield

    # Shutdown
    if evolution_manager:
        evolution_manager.stop_evolution()

    if http_client:
        await http_client.aclose()

    print("🛑 MemEvolve API server shut down")


# Create FastAPI app
app = FastAPI(
    title="MemEvolve API",
    description="Memory-enhanced proxy for OpenAI-compatible LLM APIs",
    version="0.1.0",
    lifespan=lifespan
)

# Include routes
app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Use global instances or fall back to lifespan variables
    global_memory = get_memory_system()
    evolution_status = evolution_manager.get_status() if evolution_manager else None

    return {
        "status": "healthy",
        "memory_enabled": global_memory is not None or memory_system is not None,
        "memory_integration_enabled": memory_middleware is not None,
        "evolution_enabled": evolution_manager is not None,
        "evolution_status": evolution_status,
        "upstream_url": (
            proxy_config.upstream_base_url if proxy_config else None
        )
    }


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(path: str, request: Request):
    """
    Proxy all requests to the upstream OpenAI-compatible API.

    This endpoint forwards all requests to the configured upstream URL,
    adding memory integration where appropriate.
    """
    if not proxy_config or not http_client:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Get request data
    body = await request.body()
    headers = dict(request.headers)

    # Remove host header (will be set by httpx)
    headers.pop("host", None)

    # Add upstream API key if configured
    if proxy_config.upstream_api_key:
        headers["authorization"] = (
            f"Bearer {proxy_config.upstream_api_key}"
        )

    # Process request through memory middleware
    request_context = {"body": body, "headers": headers}
    if memory_middleware:
        request_context = await memory_middleware.process_request(path, request.method, body, headers)

    # Build upstream URL
    base_url = proxy_config.upstream_base_url.rstrip('/')
    if base_url.endswith('/v1'):
        upstream_url = f"{base_url}/{path}"
    else:
        upstream_url = f"{base_url}/v1/{path}"

    try:
        # Send request to upstream
        response = await http_client.request(
            method=request.method,
            url=upstream_url,
            headers=request_context["headers"],
            content=request_context["body"],
            params=request.query_params
        )

        # Get response content for middleware processing
        response_content = b""
        if request.method == "POST" and path.startswith("chat/completions"):
            response_content = response.content

        # Process response through memory middleware
        if memory_middleware and response_content:
            await memory_middleware.process_response(
                path, request.method, request_context["body"], response_content, request_context
            )

        # Return response
        return StreamingResponse(
            response.aiter_bytes(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, detail=f"Upstream request failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
