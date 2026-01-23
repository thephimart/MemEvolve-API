"""
MemEvolve API Server - Memory-enhanced proxy for OpenAI-compatible APIs

This server acts as a transparent proxy that integrates MemEvolve memory
functionality with any OpenAI-compatible LLM API endpoint.
"""

import os
import logging
import json
import asyncio
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..memory_system import MemorySystem
from utils.config import load_config, MemEvolveConfig
from .routes import router
from .middleware import MemoryMiddleware
from .evolution_manager import EvolutionManager

# Configure logging
api_server_enable = os.getenv('MEMEVOLVE_LOG_API_SERVER_ENABLE', 'false').lower() == 'true'
logs_dir = os.getenv('MEMEVOLVE_LOGS_DIR', './logs')
api_server_dir = os.getenv('MEMEVOLVE_LOG_API_SERVER_DIR', logs_dir)

handlers: list[logging.Handler] = []  # List to collect handlers
handlers.append(logging.StreamHandler())  # Always log to console

if api_server_enable:
    os.makedirs(api_server_dir, exist_ok=True)
    handlers.insert(0, logging.FileHandler(os.path.join(api_server_dir, 'api-server.log')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)


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

        # Initialize evolution manager first (needed by middleware)
        evolution_manager = None
        if config.evolution.enable and memory_system:
            evolution_manager = EvolutionManager(config, memory_system)
            evolution_manager.start_evolution()

        # Initialize memory middleware if enabled
        memory_middleware = (
            MemoryMiddleware(memory_system, evolution_manager, config)
            if memory_integration_enabled and memory_system else None
        )

        # Initialize HTTP client
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(config.upstream.timeout), connect=10.0),
            follow_redirects=True
        )

        # Enhanced startup information
        print()
        print("✅ MemEvolve API server started successfully")
        print()
        upstream_llm_status = "Enabled" if config.upstream.base_url else "Disabled"
        if upstream_llm_status == "Disabled":
            raise ValueError("Upstream LLM configuration is required")

        print(f"   Upstream LLM: {upstream_llm_status}")
        if config.upstream.base_url:
            print(f"     Base URL: {config.upstream.base_url}")
            if config.upstream.model:
                print(f"     Model: {config.upstream.model}")

        # Memory LLM (for encoding experiences)
        memory_llm_status = "Enabled" if config.memory.base_url else "Disabled"
        print(f"   Memory LLM: {memory_llm_status}")
        if config.memory.base_url:
            print(f"     Base URL: {config.memory.base_url}")
            if config.memory.model:
                print(f"     Model: {config.memory.model}")

        # Embedding LLM (for semantic search)
        embedding_status = "Using Embedding LLM" if config.embedding.base_url else "Using Upstream LLM"
        print(f"   Embedding LLM: {embedding_status}")
        if config.embedding.base_url:
            print(f"     Base URL: {config.embedding.base_url}")
            if config.embedding.model:
                print(f"     Model: {config.embedding.model}")

            # Show max_tokens and dimension
            if config.embedding.max_tokens is not None:
                tokens_source = os.getenv("MEMEVOLVE_EMBEDDING_MAX_TOKENS")
                if tokens_source and tokens_source.strip():
                    tokens_source = ".env"
                else:
                    tokens_source = "auto-detected"
                print(f"     Max Tokens: {config.embedding.max_tokens} ({tokens_source})")
            if config.embedding.dimension is not None:
                dim_source = os.getenv("MEMEVOLVE_EMBEDDING_DIMENSION")
                if dim_source and dim_source.strip():
                    dim_source = ".env"
                else:
                    dim_source = "auto-detected"
                print(f"     Dimension: {config.embedding.dimension} ({dim_source})")

        # Memory system status
        memory_status = (
            'Enabled' if memory_system and memory_integration_enabled
            else 'Disabled'
        )
        print(f"   Memory System: {memory_status}")
        print(f"   Memory Integration: {'Enabled' if memory_middleware else 'Disabled'}")
        if memory_system and memory_integration_enabled:
            # Show current memory stats
            health = memory_system.get_health_metrics()
            if health:
                print(f"     Memories: {health.total_units}")
                print(f"     Storage: {config.storage.backend_type}")
                print(f"     Retrieval: {config.retrieval.strategy_type}")

            # Show current architecture if available
            current_genotype = evolution_manager.current_genotype if evolution_manager else None
            if current_genotype:
                print(f"     Architecture: {current_genotype.get_genome_id()[:8]}")

        # Evolution status
        evolution_status = (
            'Enabled' if evolution_manager and config.evolution.enable
            else 'Disabled'
        )
        print(f"   Evolution: {evolution_status}")
        if evolution_manager and config.evolution.enable:
            print(f"     Generations: {config.evolution.generations}")
            print(f"     Population Size: {config.evolution.population_size}")

        # API Endpoints
        print()
        print(f"   API Endpoints: See http://{config.api.host}:{config.api.port}/docs for full API documentation")
        print()
        
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
        "middleware_debug": {
            "middleware_exists": memory_middleware is not None,
            "memory_system_in_middleware": memory_middleware.memory_system is not None if memory_middleware else False,
            "evolution_manager_in_middleware": memory_middleware.evolution_manager is not None if memory_middleware else False,
            "process_request_calls": memory_middleware.process_request_count if memory_middleware else 0,
            "process_response_calls": memory_middleware.process_response_count if memory_middleware else 0
        },
        "evolution_enabled": evolution_manager is not None,
        "evolution_status": evolution_status,
        "upstream_url": (
            proxy_config.upstream_base_url if proxy_config else None
        )
    }


async def _async_encode_experience(memory_middleware, evolution_manager, request_body: bytes, response_data: bytes):
    """Asynchronously encode experience from streaming response data."""
    try:
        logger.info("Async experience encoding started")
        logger.info(f"Request body length: {len(request_body)}")
        logger.info(f"Response data length: {len(response_data)}")
        logger.info(f"Response data preview: {response_data[:200].decode('utf-8', errors='ignore')}")

        await memory_middleware.process_response(
            "chat/completions", "POST", request_body, response_data, {}
        )
        logger.info("Async experience encoding completed")
    except Exception as e:
        logger.error(f"Async experience encoding failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

# Import the shared streaming utility
from ..utils import extract_final_from_stream


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(path: str, request: Request):
    logger.info(f"Inbound request: {request.method} /v1/{path}")
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
    # Remove content-length header (will be set automatically by httpx)
    headers.pop("content-length", None)

    # Add upstream API key if configured
    if proxy_config.upstream_api_key:
        headers["authorization"] = (
            f"Bearer {proxy_config.upstream_api_key}"
        )

    request_context = {"body": body, "headers": headers}

    # Check if this is a streaming request
    is_streaming_request = False
    if request.method == "POST" and path == "chat/completions":
        try:
            request_data = json.loads(body)
            is_streaming_request = request_data.get("stream", False)
        except json.JSONDecodeError:
            pass

    # Phase 1: Always inject memories for chat completions (both streaming and non-streaming)
    if memory_middleware and request.method == "POST" and path == "chat/completions":
        logger.info("Injecting memories into request for enhanced responses")
        original_body = body
        request_context = await memory_middleware.process_request(path, request.method, body, headers)
        if request_context["body"] != original_body:
            logger.info("Memories successfully injected - request enhanced")
        else:
            logger.info("No memories found to inject - using original request")

    # Build upstream URL
    base_url = proxy_config.upstream_base_url.rstrip('/')
    if base_url.endswith('/v1'):
        upstream_url = f"{base_url}/{path}"
    else:
        upstream_url = f"{base_url}/v1/{path}"

    try:
        # Start timing for API request
        request_start_time = time.time()

        # Send request to upstream
        response = await http_client.request(
            method=request.method,
            url=upstream_url,
            headers=request_context["headers"],
            content=request_context["body"],
            params=request.query_params
        )

        # Calculate response time
        response_time = time.time() - request_start_time

        # Handle streaming requests with async experience encoding
        if request.method == "POST" and path == "chat/completions" and is_streaming_request:
            logger.info("Streaming request detected, injecting memories and enabling async experience encoding")
            logger.info(f"Upstream response status: {response.status_code}")

            # Phase 2: For streaming requests, collect response data for async experience encoding
            if memory_middleware and response.status_code == 200:
                # Collect the streaming response data
                response_chunks = []
                async for chunk in response.aiter_bytes():
                    response_chunks.append(chunk)

                response_data = b''.join(response_chunks)

                # Spawn async task for experience encoding
                logger.info("Spawning async task for experience encoding")
                asyncio.create_task(
                    _async_encode_experience(
                        memory_middleware,
                        evolution_manager,
                        request_context["body"],
                        response_data
                    )
                )

                # Record API request for evolution
                if evolution_manager:
                    success = response.status_code == 200
                    evolution_manager.record_api_request(response_time, success)

                logger.info("Returning streaming response to client")
                return StreamingResponse(
                    iter(response_chunks),  # Use collected chunks
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )
            else:
                # Fallback: pass through without processing
                logger.info("No memory middleware or non-200 response, passing through")
                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )

        # Handle non-streaming chat completions with memory processing
        if request.method == "POST" and path == "chat/completions" and not is_streaming_request:
            # Read response content for middleware processing
            response_content = b""
            try:
                # Read all response data
                chunks = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                response_content = b''.join(chunks)
                logger.info(f"Collected response content, length: {len(response_content)}")
                if len(response_content) > 0:
                    logger.info(f"Response content preview: {response_content[:200].decode('utf-8', errors='ignore')}")

                # Check if this is a streaming response (starts with "data: ")
                response_str = response_content.decode('utf-8', errors='ignore')
                if response_str.strip().startswith('data: '):
                    logger.info("Detected streaming response, extracting final result")
                    extracted = extract_final_from_stream(response_str)
                    if isinstance(extracted, str):
                        response_content = extracted.encode('utf-8')
                    else:
                        response_content = extracted
                    logger.info(f"Extracted final response, length: {len(response_content)}")

            except Exception as e:
                logger.error(f"Error reading response content: {e}")
                response_content = b'{"error": "response_read_failed"}'

            # Process response through memory middleware
            if memory_middleware:
                logger.info("Calling middleware process_response")
                await memory_middleware.process_response(
                    path, request.method, request_context["body"], response_content, request_context
                )
                logger.info("Middleware process_response completed")
            else:
                logger.warning("No memory middleware available")

            # Record API request for evolution
            if evolution_manager:
                success = response.status_code == 200 and len(response_content) > 0
                evolution_manager.record_api_request(response_time, success)

            # Return as Response with collected content
            return Response(
                content=response_content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "application/json")
            )

        # For all other requests, use streaming approach
        # Record API request for evolution
        if evolution_manager and request.method == "POST" and path.startswith("chat/completions"):
            success = response.status_code == 200
            evolution_manager.record_api_request(response_time, success)

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
