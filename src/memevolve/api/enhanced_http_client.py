"""
Enhanced HTTP Client with Endpoint Metrics Tracking
============================================

This HTTP client wraps httpx.AsyncClient to add comprehensive metrics tracking
for all API calls to upstream, memory, and embedding endpoints.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union
import httpx

# Import metrics collector
from ..utils.endpoint_metrics_collector import get_endpoint_metrics_collector

logger = logging.getLogger(__name__)


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

        logger.info(f"EnhancedHTTPClient initialized for {base_url}")

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

            logger.info(
                f"Enhanced HTTP {endpoint_type} call completed: {
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

            logger.info(
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
            response = await self.client.get(url, headers=headers, **kwargs)

            # Count response tokens for GET requests (usually minimal)
            output_tokens = self._count_response_tokens(response)

            self.metrics_collector.end_endpoint_call(
                call_id=call_id,
                output_tokens=output_tokens,
                success=response.is_success,
                error_code=str(response.status_code) if not response.is_success else None
            )

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
        import uuid
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
