"""
Enhanced HTTP Client with Endpoint Metrics Tracking
============================================

This HTTP client wraps httpx.AsyncClient to add comprehensive metrics tracking
for all API calls to upstream, memory, and embedding endpoints.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, Union

try:
    import httpx
except ImportError as e:
    raise RuntimeError(f"Missing required dependency for async HTTP client: {e}")

from ..utils.endpoint_metrics_collector import get_endpoint_metrics_collector
from ..utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)


class EnhancedHTTPClient:
    """Enhanced HTTP client with automatic endpoint metrics tracking."""

    def __init__(
        self,
        base_url: str = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        config: Optional[Any] = None,
        **kwargs
    ):
        self.base_url = base_url
        self.default_headers = headers or {}
        self.timeout = timeout
        self.kwargs = kwargs

        # Initialize underlying httpx client
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            **kwargs
        )

        # Get metrics collector with config
        self.metrics_collector = get_endpoint_metrics_collector(config)

        logger.debug(f"EnhancedHTTPClient initialized for {base_url}")

    async def post(
        self,
        url: str,
        content: Optional[bytes] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Enhanced POST request with automatic metrics tracking."""
        # Determine endpoint type from URL
        endpoint_type = self._determine_endpoint_type(url)

        # Count input tokens
        input_tokens = self._count_request_tokens(json or data or content)

        # Start metrics tracking
        call_id = self.metrics_collector.start_endpoint_call(
            request_id=self._generate_request_id(),
            endpoint_type=endpoint_type,
            input_tokens=input_tokens,
            request_type='api_call'
        )

        try:
            # Make the actual request
            start_time = time.time()

            # Enhanced logging for debugging
            logger.debug(f"Making POST request to {url}")
            logger.debug(f"Request headers: {headers}")
            if json:
                logger.debug(f"Request JSON keys: {list(json.keys())}")
            logger.debug(f"Request size: {len(content or b'')} bytes")

            response = await self.client.post(
                url=url,
                content=content,
                data=data,
                json=json,
                headers=headers,
                **kwargs
            )

            request_time = (time.time() - start_time) * 1000

            # Count response tokens
            output_tokens = self._count_response_tokens(response)

            # End metrics tracking
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                output_tokens=output_tokens,
                success=response.is_success,
                error_code=str(response.status_code) if not response.is_success else None,
                error_message=response.reason_phrase if not response.is_success else None
            )

            logger.debug(
                f"HTTP POST {endpoint_type} completed: {url} -> {
                    response.status_code} in {
                    request_time:.1f}ms, {output_tokens} tokens")
            return response

        except Exception as e:
            # End tracking with error
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                success=False,
                error_code=type(e).__name__,
                error_message=str(e)
            )

            # Handle event loop closure gracefully
            if "Event loop is closed" in str(e):
                logger.warning(
                    f"HTTP {endpoint_type} call interrupted by event loop closure - this is expected during cleanup")
                raise RuntimeError(f"HTTP call interrupted: {e}") from None
            else:
                logger.error(f"Enhanced HTTP {endpoint_type} call failed: {e}")
                raise

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        content: Optional[bytes] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Enhanced generic HTTP request with automatic metrics tracking."""
        endpoint_type = self._determine_endpoint_type(url)

        # Count input tokens
        input_tokens = self._count_request_tokens(json or content)

        # Start metrics tracking
        call_id = self.metrics_collector.start_endpoint_call(
            request_id=self._generate_request_id(),
            endpoint_type=endpoint_type,
            input_tokens=input_tokens,
            request_type='api_call'
        )

        try:
            # Make the actual request
            start_time = time.time()

            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                content=content,
                params=params,
                json=json,
                **kwargs
            )

            request_time = (time.time() - start_time) * 1000

            # Count response tokens
            output_tokens = self._count_response_tokens(response)

            # End metrics tracking
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                output_tokens=output_tokens,
                success=response.is_success,
                error_code=str(response.status_code) if not response.is_success else None,
                error_message=response.reason_phrase if not response.is_success else None
            )

            logger.debug(
                f"Enhanced HTTP {endpoint_type} {method} call completed: {
                    request_time:.1f}ms, {output_tokens} tokens")
            return response

        except Exception as e:
            # End tracking with error
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                success=False,
                error_code=type(e).__name__,
                error_message=str(e)
            )
            logger.error(f"Enhanced HTTP {endpoint_type} {method} call failed: {e}")
            raise

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Enhanced GET request with automatic metrics tracking."""
        endpoint_type = self._determine_endpoint_type(url)

        # Start metrics tracking
        call_id = self.metrics_collector.start_endpoint_call(
            request_id=self._generate_request_id(),
            endpoint_type=endpoint_type,
            request_type='api_call'
        )

        try:
            logger.debug(f"Making GET request to {url}")
            response = await self.client.get(url, headers=headers, **kwargs)
            logger.debug(f"GET response: {response.status_code} from {url}")

            # Count response tokens for GET requests (usually minimal)
            output_tokens = self._count_response_tokens(response)

            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                output_tokens=output_tokens,
                success=response.is_success,
                error_code=str(response.status_code) if not response.is_success else None
            )

            logger.debug(
                f"GET request completed: {url} -> {response.status_code} in {output_tokens} tokens")
            return response

        except Exception as e:
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                success=False,
                error_code=type(e).__name__,
                error_message=str(e)
            )
            logger.error(f"Enhanced HTTP {endpoint_type} call failed: {e}")
            raise

    def _determine_endpoint_type(self, url: str) -> str:
        """Determine the type of endpoint from URL."""
        if not self.base_url:
            return "unknown"

        if url.startswith(self.base_url) or "/chat/completions" in url:
            return "upstream"
        elif "memory" in self.base_url.lower() or "11433" in self.base_url:
            return "memory"
        elif "embedding" in self.base_url.lower() or "11435" in self.base_url:
            return "embedding"
        else:
            return "upstream"

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def _count_request_tokens(self, data: Any) -> int:
        """Count tokens in request data."""
        if not data:
            return 0

        try:
            if isinstance(data, dict):
                # Extract messages or prompt text for token counting
                if "messages" in data:
                    text = " ".join(
                        msg.get("content", "") for msg in data["messages"]
                    )
                elif "prompt" in data:
                    text = str(data["prompt"])
                elif "input" in data:
                    text = str(data["input"])
                else:
                    # Fallback: stringify the entire data
                    text = json.dumps(data)

                # Use tiktoken for accurate token counting
                try:
                    import tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except ImportError:
                    # Fallback to rough estimation
                    return len(text.split()) * 1.3
            elif isinstance(data, str):
                # String data
                try:
                    import tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(data))
                except ImportError:
                    return len(data.split()) * 1.3
            elif isinstance(data, bytes):
                # Byte data - try to decode and count
                try:
                    text = data.decode('utf-8')
                    try:
                        import tiktoken
                        encoding = tiktoken.get_encoding("cl100k_base")
                        return len(encoding.encode(text))
                    except ImportError:
                        return len(text.split()) * 1.3
                except BaseException:
                    return 0
            else:
                return 0
        except Exception:
            return 0

    def _count_response_tokens(self, response) -> int:
        """Count tokens in HTTP response."""
        if not response.content:
            return 0

        try:
            # Parse JSON response
            data = response.json()

            # Try to extract text content for token counting
            text = ""
            if isinstance(data, dict):
                if "choices" in data:
                    # Chat completion response
                    for choice in data["choices"]:
                        if "message" in choice:
                            text += choice["message"].get("content", "")
                        elif "text" in choice:
                            text += choice["text"]
                elif "data" in data and data["data"]:
                    # Embedding response or list response
                    for item in data["data"]:
                        if isinstance(item, dict) and "embedding" in item:
                            continue  # Don't count embedding vectors
                        text += json.dumps(item)
                else:
                    # Fallback
                    text = json.dumps(data)
            else:
                text = json.dumps(data)

            # Count tokens using tiktoken
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except ImportError:
                return len(text.split()) * 1.3
        except Exception:
            # Fallback: estimate from content length
            return len(response.content) // 4  # Rough estimate

    async def close(self):
        """Close the underlying client."""
        await self.client.aclose()

    async def aclose(self):
        """Alias for close() - compatibility with server."""
        await self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class IsolatedHTTPClient:
    """Isolated HTTP client with dedicated event loop management.

    This client solves the event loop closure issues by creating isolated
    sessions for each request, preventing race conditions and resource conflicts.
    """

    def __init__(
        self,
        base_url: str = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        config: Optional[Any] = None,
        **kwargs
    ):
        self.base_url = base_url
        self.default_headers = headers or {}
        self.timeout = timeout
        self.kwargs = kwargs

        # Session management for isolation
        self._session_lock = asyncio.Lock()
        self._active_sessions = set()
        self._session_counter = 0

        # Get metrics collector with config
        self.metrics_collector = get_endpoint_metrics_collector(config)

        logger.debug(f"IsolatedHTTPClient initialized for {base_url}")

    async def _create_isolated_session(self) -> httpx.AsyncClient:
        """Create truly isolated HTTP session."""
        async with self._session_lock:
            self._session_counter += 1
            session_id = self._session_counter

            session = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers,
                timeout=self.timeout,
                **self.kwargs
            )

            self._active_sessions.add(session_id)
            logger.debug(
                f"[ISOLATED_HTTP] Created session {session_id}, active sessions: {len(self._active_sessions)}")
            return session

    async def _cleanup_session(self, session: httpx.AsyncClient, session_id: int) -> None:
        """Explicit session cleanup with proper resource management."""
        try:
            await session.aclose()
            async with self._session_lock:
                self._active_sessions.discard(session_id)
            logger.debug(
                f"[ISOLATED_HTTP] Cleaned up session {session_id}, active sessions: {len(self._active_sessions)}")
        except Exception as e:
            logger.warning(f"[ISOLATED_HTTP] Error cleaning up session {session_id}: {e}")

    async def make_request(
        self,
        method: str,
        url: str,
        content: Optional[bytes] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """Thread-safe isolated request handling."""
        endpoint_type = self._determine_endpoint_type(url)
        session_id = None
        session = None

        try:
            # Create isolated session
            session = await self._create_isolated_session()
            session_id = self._session_counter

            # Count input tokens
            input_tokens = self._count_request_tokens(json or data or content)

            # Start metrics tracking
            call_id = self.metrics_collector.start_endpoint_call(
                request_id=self._generate_request_id(),
                endpoint_type=endpoint_type,
                input_tokens=input_tokens,
                request_type='api_call'
            )

            # Make the request
            start_time = time.time()
            logger.debug(f"[ISOLATED_HTTP] Making {method} request with session {session_id}")

            response = await session.request(
                method=method,
                url=url,
                content=content,
                data=data,
                json=json,
                headers=headers,
                **kwargs
            )

            request_time = (time.time() - start_time) * 1000

            # Count response tokens
            output_tokens = self._count_response_tokens(response)

            # End metrics tracking
            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                output_tokens=output_tokens,
                success=response.is_success,
                error_code=str(response.status_code) if not response.is_success else None,
                error_message=response.reason_phrase if not response.is_success else None
            )

            logger.debug(
                f"[ISOLATED_HTTP] {method} {endpoint_type} completed: {url} -> {response.status_code} "
                f"in {request_time:.1f}ms, {output_tokens} tokens (session {session_id})")

            return response

        except Exception as e:
            # End tracking with error
            if 'call_id' in locals():
                self.metrics_collector.end_endpoint_call(
                    call_id=call_id,
                    success=False,
                    error_code=type(e).__name__,
                    error_message=str(e)
                )

            logger.error(
                f"[ISOLATED_HTTP] {method} {endpoint_type} failed (session {session_id}): {e}")
            raise

        finally:
            # Always cleanup session
            if session and session_id:
                await self._cleanup_session(session, session_id)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Isolated POST request."""
        return await self.make_request("POST", url, **kwargs)

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Isolated GET request."""
        return await self.make_request("GET", url, **kwargs)

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Isolated generic request."""
        return await self.make_request(method, url, **kwargs)

    async def close(self):
        """Close all active sessions."""
        async with self._session_lock:
            logger.debug(
                f"[ISOLATED_HTTP] Closing client, cleaning up {len(self._active_sessions)} sessions")
            # Note: Sessions are cleaned up per-request, but ensure any remaining are closed
            self._active_sessions.clear()

    async def aclose(self):
        """Alias for close()."""
        await self.close()

    def _determine_endpoint_type(self, url: str) -> str:
        """Determine the type of endpoint from URL."""
        if not self.base_url:
            return "unknown"

        if url.startswith(self.base_url) or "/chat/completions" in url:
            return "upstream"
        elif "memory" in self.base_url.lower() or "11433" in self.base_url:
            return "memory"
        elif "embedding" in self.base_url.lower() or "11435" in self.base_url:
            return "embedding"
        else:
            return "upstream"

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def _count_request_tokens(self, data: Any) -> int:
        """Count tokens in request data."""
        if not data:
            return 0

        try:
            if isinstance(data, dict):
                # Extract messages or prompt text for token counting
                if "messages" in data:
                    text = " ".join(
                        msg.get("content", "") for msg in data["messages"]
                    )
                elif "prompt" in data:
                    text = str(data["prompt"])
                elif "input" in data:
                    text = str(data["input"])
                else:
                    # Fallback: stringify the entire data
                    text = json.dumps(data)

                # Use tiktoken for accurate token counting
                try:
                    import tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except ImportError:
                    # Fallback to rough estimation
                    return len(text.split()) * 1.3
            elif isinstance(data, str):
                # String data
                try:
                    import tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(data))
                except ImportError:
                    return len(data.split()) * 1.3
            elif isinstance(data, bytes):
                # Byte data - try to decode and count
                try:
                    text = data.decode('utf-8')
                    try:
                        import tiktoken
                        encoding = tiktoken.get_encoding("cl100k_base")
                        return len(encoding.encode(text))
                    except ImportError:
                        return len(text.split()) * 1.3
                except BaseException:
                    return 0
            else:
                return 0
        except Exception:
            return 0

    def _count_response_tokens(self, response) -> int:
        """Count tokens in HTTP response."""
        if not response.content:
            return 0

        try:
            # Parse JSON response
            data = response.json()

            # Try to extract text content for token counting
            text = ""
            if isinstance(data, dict):
                if "choices" in data:
                    # Chat completion response
                    for choice in data["choices"]:
                        if "message" in choice:
                            text += choice["message"].get("content", "")
                        elif "text" in choice:
                            text += choice["text"]
                elif "data" in data and data["data"]:
                    # Embedding response or list response
                    for item in data["data"]:
                        if isinstance(item, dict) and "embedding" in item:
                            continue  # Don't count embedding vectors
                        text += json.dumps(item)
                else:
                    # Fallback
                    text = json.dumps(data)
            else:
                text = json.dumps(data)

            # Count tokens using tiktoken
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except ImportError:
                return len(text.split()) * 1.3
        except Exception:
            # Fallback: estimate from content length
            return len(response.content) // 4  # Rough estimate

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class IsolatedOpenAICompatibleClient:
    """OpenAI API compatibility wrapper for IsolatedHTTPClient.

    This client uses isolated sessions to prevent event loop conflicts
    and provides a clean interface for memory encoding operations.
    """

    def __init__(self, base_url: str = None, headers: Optional[Dict[str, str]] = None,
                 timeout: Optional[Union[float, httpx.Timeout]] = None,
                 config: Optional[Any] = None, **kwargs):
        # Create isolated HTTP client
        self.isolated_client = IsolatedHTTPClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            config=config,
            **kwargs
        )
        self.chat = _IsolatedChatCompletionsWrapper(self.isolated_client)
        self.embeddings = _IsolatedEmbeddingsWrapper(self.isolated_client)

    async def close(self):
        """Close the underlying isolated client."""
        await self.isolated_client.close()

    async def aclose(self):
        """Alias for close()."""
        await self.close()


class OpenAICompatibleClient:
    """OpenAI API compatibility wrapper for EnhancedHTTPClient."""

    def __init__(self, http_client: EnhancedHTTPClient):
        self.http_client = http_client
        self.chat = _ChatCompletionsWrapper(http_client)
        self.embeddings = _EmbeddingsWrapper(http_client)


class _IsolatedChatCompletionsWrapper:
    """Isolated chat completions API wrapper."""

    def __init__(self, isolated_client: IsolatedHTTPClient):
        self.isolated_client = isolated_client
        self.completions = _IsolatedCompletionsWrapper(isolated_client)


class _IsolatedCompletionsWrapper:
    """Isolated completions API wrapper."""

    def __init__(self, isolated_client: IsolatedHTTPClient):
        self.isolated_client = isolated_client

    def create(self, **kwargs):
        """Create chat completion using isolated HTTP client."""
        import asyncio

        if "messages" in kwargs:
            data = {
                "messages": kwargs["messages"],
                "max_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
            }

            if "model" in kwargs:
                data["model"] = kwargs["model"]

            # Use dedicated event loop for memory operations
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, run in thread with new isolated loop
                    import concurrent.futures

                    def run_in_isolated_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self._create_async_isolated(data))
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_isolated_thread)
                        response = future.result()
                else:
                    # Use existing loop directly
                    response = loop.run_until_complete(self._create_async_isolated(data))
            except RuntimeError:
                # No event loop, create new one
                response = asyncio.run(self._create_async_isolated(data))

            # Convert HTTP response to OpenAI-compatible format
            return _OpenAIResponse(response.json())

    async def _create_async_isolated(self, data: dict):
        """Async chat completion creation with isolated client."""
        response = await self.isolated_client.post(
            url="/chat/completions",
            json=data
        )
        return response


class _IsolatedEmbeddingsWrapper:
    """Isolated embeddings API wrapper."""

    def __init__(self, isolated_client: IsolatedHTTPClient):
        self.isolated_client = isolated_client

    def create(self, **kwargs):
        """Create embedding using isolated HTTP client."""
        import asyncio
        data = {}

        if "input" in kwargs:
            data["input"] = kwargs["input"]
        if "model" in kwargs:
            data["model"] = kwargs["model"]

        # Use dedicated event loop for memory operations
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, run in thread with new isolated loop
                import concurrent.futures

                def run_in_isolated_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._create_async_isolated(data))
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_isolated_thread)
                    response = future.result()
            else:
                # Use existing loop directly
                response = loop.run_until_complete(self._create_async_isolated(data))
        except RuntimeError:
            # No event loop, create new one
            response = asyncio.run(self._create_async_isolated(data))

        # Return response data directly for embeddings
        return _EmbeddingResponse(response.json())

    async def _create_async_isolated(self, data: dict):
        """Async embedding creation with isolated client."""
        response = await self.isolated_client.post(
            url="/embeddings",
            json=data
        )
        return response


class _ChatCompletionsWrapper:
    """Chat completions API wrapper."""

    def __init__(self, http_client: EnhancedHTTPClient):
        self.http_client = http_client
        self.completions = _CompletionsWrapper(http_client)


class _CompletionsWrapper:
    """Completions API wrapper."""

    def __init__(self, http_client: EnhancedHTTPClient):
        self.http_client = http_client

    def create(self, **kwargs):
        """Create chat completion using enhanced HTTP client."""
        import asyncio
        if "messages" in kwargs:
            data = {
                "messages": kwargs["messages"],
                "max_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
            }

            if "model" in kwargs:
                data["model"] = kwargs["model"]

            # Run async HTTP client in sync context
            loop = None
            try:
                loop = asyncio.get_event_loop()
                logger.debug(
                    f"[EVENT_LOOP_DEBUG] Got existing event loop: {
                        id(loop)}, running: {
                        loop.is_running()}")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.debug(
                    f"[EVENT_LOOP_DEBUG] Created new event loop: {
                        id(loop)}, running: {
                        loop.is_running()}")

            if loop.is_running():
                # If event loop is already running, we need to run in a thread
                logger.warning(
                    f"[EVENT_LOOP_DEBUG] Event loop {
                        id(loop)} is running - using thread executor (POTENTIAL RACE CONDITION)")
                import concurrent.futures

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    logger.debug(f"[EVENT_LOOP_DEBUG] Thread created new loop: {id(new_loop)}")
                    try:
                        return new_loop.run_until_complete(self._create_async(data))
                    finally:
                        new_loop.close()
                        logger.debug(f"[EVENT_LOOP_DEBUG] Thread closed loop: {id(new_loop)}")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    response = future.result()
                    logger.debug(f"[EVENT_LOOP_DEBUG] Thread executor completed for completion")
            else:
                logger.debug(f"[EVENT_LOOP_DEBUG] Using existing loop {id(loop)} directly")
                response = loop.run_until_complete(self._create_async(data))
                logger.debug(f"[EVENT_LOOP_DEBUG] Direct loop execution completed")

            # Convert HTTP response to OpenAI-compatible format
            return _OpenAIResponse(response.json())

    async def _create_async(self, data: dict):
        """Async chat completion creation."""
        response = await self.http_client.post(
            url="/chat/completions",
            json=data
        )
        return response


class _EmbeddingsWrapper:
    """Embeddings API wrapper."""

    def __init__(self, http_client: EnhancedHTTPClient):
        self.http_client = http_client

    def create(self, **kwargs):
        """Create embedding using enhanced HTTP client."""
        import asyncio
        data = {}

        if "input" in kwargs:
            data["input"] = kwargs["input"]
        if "model" in kwargs:
            data["model"] = kwargs["model"]

        # Run async HTTP client in sync context
        loop = None
        try:
            loop = asyncio.get_event_loop()
            logger.debug(
                f"[EVENT_LOOP_DEBUG] Embeddings - Got existing loop: {id(loop)}, running: {loop.is_running()}")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug(
                f"[EVENT_LOOP_DEBUG] Embeddings - Created new loop: {id(loop)}, running: {loop.is_running()}")

        if loop.is_running():
            # If event loop is already running, we need to run in a thread
            logger.warning(
                f"[EVENT_LOOP_DEBUG] Embeddings - Loop {id(loop)} running - using thread executor (POTENTIAL RACE CONDITION)")
            import concurrent.futures

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                logger.debug(f"[EVENT_LOOP_DEBUG] Embeddings thread created loop: {id(new_loop)}")
                try:
                    return new_loop.run_until_complete(self._create_async(data))
                finally:
                    new_loop.close()
                    logger.debug(
                        f"[EVENT_LOOP_DEBUG] Embeddings thread closed loop: {
                            id(new_loop)}")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                response = future.result()
                logger.debug(f"[EVENT_LOOP_DEBUG] Embeddings thread executor completed")
        else:
            logger.debug(f"[EVENT_LOOP_DEBUG] Embeddings - Using existing loop {id(loop)} directly")
            response = loop.run_until_complete(self._create_async(data))
            logger.debug(f"[EVENT_LOOP_DEBUG] Embeddings direct loop execution completed")

        # Return response data directly for embeddings
        return _EmbeddingResponse(response.json())

    async def _create_async(self, data: dict):
        """Async embedding creation."""
        response = await self.http_client.post(
            url="/embeddings",
            json=data
        )
        return response


class _OpenAIResponse:
    """OpenAI response format wrapper."""

    def __init__(self, data: dict):
        self._data = data
        self.choices = [_Choice(choice) for choice in data.get("choices", [])]


class _Choice:
    """OpenAI choice wrapper."""

    def __init__(self, choice_data: dict):
        self._data = choice_data
        self.message = _Message(choice_data.get("message", {}))


class _Message:
    """OpenAI message wrapper."""

    def __init__(self, message_data: dict):
        self._data = message_data
        self.content = message_data.get("content")


class _EmbeddingResponse:
    """Embedding response wrapper that behaves like OpenAI response."""

    def __init__(self, data: dict):
        self._data = data
        self.data = [_EmbeddingData(item) for item in data.get("data", [])]


class _EmbeddingData:
    """Embedding data wrapper."""

    def __init__(self, item_data: dict):
        self._data = item_data
        self.embedding = item_data.get("embedding", [])


def create_enhanced_http_client(*args, **kwargs):
    """Factory function to create enhanced HTTP client."""
    return EnhancedHTTPClient(*args, **kwargs)
