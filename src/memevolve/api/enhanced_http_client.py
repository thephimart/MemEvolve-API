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
            start_time = time.time()
            response = await self.client.get(url, headers=headers, **kwargs)
            request_time = (time.time() - start_time) * 1000

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
            raise

    async def put(
        self,
        url: str,
        content: Optional[bytes] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Enhanced PUT request with automatic metrics tracking."""
        endpoint_type = self._determine_endpoint_type(url)
        input_tokens = self._count_request_tokens(json or data or content)

        call_id = self.metrics_collector.start_endpoint_call(
            request_id=self._generate_request_id(),
            endpoint_type=endpoint_type,
            input_tokens=input_tokens,
            request_type='api_call'
        )

        try:
            start_time = time.time()
            response = await self.client.put(url, content, data, json, headers, **kwargs)
            request_time = (time.time() - start_time) * 1000
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
            raise

    def _determine_endpoint_type(self, url: str) -> str:
        """Determine endpoint type from URL."""
        url_lower = url.lower()

        if 'chat/completions' in url_lower or 'completions' in url_lower:
            return 'upstream'
        elif 'memory' in url_lower or 'query' in url_lower or 'search' in url_lower:
            return 'memory'
        elif 'embed' in url_lower:
            return 'embedding'
        else:
            return 'unknown'

    def _count_request_tokens(self, data: Any) -> int:
        """Count tokens in request data."""
        if not data:
            return 0

        try:
            if isinstance(data, dict):
                # Convert to JSON and estimate tokens
                json_str = json.dumps(data)
                return self._estimate_tokens_from_text(json_str)
            elif isinstance(data, str):
                return self._estimate_tokens_from_text(data)
            elif isinstance(data, bytes):
                # Try to decode as text first
                try:
                    text_str = data.decode('utf-8')
                    return self._estimate_tokens_from_text(text_str)
                except BaseException:
                    # Fallback: rough estimation
                    return len(data) // 4
            else:
                # For other types, use rough estimation
                return len(str(data)) // 4
        except Exception as e:
            logger.warning(f"Error counting request tokens: {e}")
            return 0

    def _count_response_tokens(self, response: httpx.Response) -> int:
        """Count tokens in response."""
        try:
            # Try to get response content
            content_length = len(response.content) if response.content else 0

            # For JSON responses, we can be more accurate
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    response_data = response.json()
                    response_str = json.dumps(response_data)
                    return self._estimate_tokens_from_text(response_str)
                except BaseException:
                    pass

            # Fallback: rough estimation from content length
            return content_length // 4  # Rough: 1 token ≈ 4 characters

        except Exception as e:
            logger.warning(f"Error counting response tokens: {e}")
            return 0

    def _estimate_tokens_from_text(self, text: str) -> int:
        """Estimate token count from text."""
        if not text:
            return 0

        # Simple estimation: 1 token ≈ 4 characters for English text
        # This is a rough approximation - for production, use proper tokenizer
        return len(text) // 4

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]  # Short ID for tracking

    async def __aenter__(self):
        """Async context manager entry."""
        return await self.client.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return await self.client.__aexit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        """Delegate other methods to underlying client."""
        return getattr(self.client, name)


class EndpointAwareClientFactory:
    """Factory for creating endpoint-aware HTTP clients."""

    @staticmethod
    def create_upstream_client(base_url: str, api_key: str = None, config: Optional[Any] = None, **kwargs):
        """Create client for upstream API endpoint."""
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        return EnhancedHTTPClient(
            base_url=base_url,
            headers=headers,
            config=config,
            **kwargs
        )

    @staticmethod
    def create_memory_client(base_url: str, api_key: str = None, config: Optional[Any] = None, **kwargs):
        """Create client for memory API endpoint."""
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        return EnhancedHTTPClient(
            base_url=base_url,
            headers=headers,
            config=config,
            **kwargs
        )

    @staticmethod
    def create_embedding_client(base_url: str, api_key: str = None, config: Optional[Any] = None, **kwargs):
        """Create client for embedding API endpoint."""
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        return EnhancedHTTPClient(
            base_url=base_url,
            headers=headers,
            config=config,
            **kwargs
        )
