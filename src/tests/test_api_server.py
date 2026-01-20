"""
Tests for MemEvolve API server.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import json
from fastapi.testclient import TestClient

from src.api.server import app, get_memory_system
from src.memory_system import MemorySystem, MemorySystemConfig


@pytest.fixture
def test_memory_config():
    """Create a test memory configuration using real environment variables."""
    from dotenv import load_dotenv
    import os
    load_dotenv()  # Load .env file

    # Use real environment variables with proper fallback hierarchy for memory calls
    # MEMEVOLVE_LLM_BASE_URL if defined, else MEMEVOLVE_UPSTREAM_BASE_URL
    llm_base_url = os.getenv("MEMEVOLVE_LLM_BASE_URL") or os.getenv(
        "MEMEVOLVE_UPSTREAM_BASE_URL")
    if not llm_base_url:
        raise ValueError(
            "LLM base URL must be configured in .env (MEMEVOLVE_LLM_BASE_URL or MEMEVOLVE_UPSTREAM_BASE_URL)")

    return MemorySystemConfig(
        llm_base_url=llm_base_url,
        llm_api_key=os.getenv("MEMEVOLVE_LLM_API_KEY", ""),
        storage_backend=None,  # Use default
        default_retrieval_top_k=3
    )


@pytest.fixture
def mock_memory_system(test_memory_config):
    """Create a mock memory system."""
    memory_system = Mock(spec=MemorySystem)
    memory_system.config = test_memory_config

    # Mock health metrics
    health_mock = Mock()
    health_mock.total_units = 5
    health_mock.newest_unit_timestamp = "2024-01-01T00:00:00Z"
    memory_system.get_health_metrics.return_value = health_mock

    # Mock operation log
    memory_system.get_operation_log.return_value = [
        {"op": "encode"}, {"op": "retrieve"}]

    # Mock query_memory
    memory_system.query_memory.return_value = [
        {"content": "Test memory 1", "score": 0.9},
        {"content": "Test memory 2", "score": 0.8}
    ]

    # Mock clear_operation_log
    memory_system.clear_operation_log = Mock()

    return memory_system


@pytest.fixture
def test_client(mock_memory_system):
    """Create a test client with mocked memory system."""
    from unittest.mock import patch

    # Set global variables for health check
    import src.api.server
    original_memory = getattr(src.api.server, '_memory_system_instance', None)

    src.api.server._memory_system_instance = mock_memory_system

    # Mock the get_memory_system function where it's used
    with patch('src.api.server.get_memory_system', return_value=mock_memory_system):
        client = TestClient(app)
        yield client

    # Restore
    src.api.server._memory_system_instance = original_memory


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["memory_enabled"] is True
        assert "memory_integration_enabled" in data
        assert "upstream_url" in data

    def test_memory_stats_endpoint(self, test_client, mock_memory_system):
        """Test memory stats endpoint."""
        response = test_client.get("/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_experiences" in data
        assert "retrieval_count" in data
        assert data["total_experiences"] == 5
        assert data["retrieval_count"] == 2

    def test_memory_search_endpoint(self, test_client, mock_memory_system):
        """Test memory search endpoint."""
        request_data = {
            "query": "test query",
            "limit": 5,
            "include_metadata": False
        }

        response = test_client.post("/memory/search", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert "content" in data[0]
        assert "score" in data[0]

    def test_memory_clear_endpoint(self, test_client, mock_memory_system):
        """Test memory clear endpoint."""
        response = test_client.post("/memory/clear")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        mock_memory_system.clear_operation_log.assert_called_once()

    def test_memory_config_endpoint(self, test_client, mock_memory_system):
        """Test memory config endpoint."""
        response = test_client.get("/memory/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm_base_url" in data
        # Should match the configured LLM base URL from environment
        import os
        expected_url = os.getenv("MEMEVOLVE_LLM_BASE_URL") or os.getenv(
            "MEMEVOLVE_UPSTREAM_BASE_URL")
        assert data["llm_base_url"] == expected_url

    def test_proxy_request_without_memory(self, test_client, monkeypatch):
        """Test proxy request when memory is disabled."""
        # Temporarily disable memory
        import src.api.server
        original_memory = src.api.server._memory_system_instance
        src.api.server._memory_system_instance = None

        try:
            # Mock httpx client
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aiter_bytes = AsyncMock(
                return_value=[b'{"result": "test"}'])

            with monkeypatch.context() as m:
                mock_client = AsyncMock()
                mock_client.request = AsyncMock(return_value=mock_response)

                # Replace the http_client in the lifespan context
                m.setattr("src.api.server.http_client", mock_client)

                # This would need more complex mocking for the full proxy test
                # For now, just ensure the endpoint exists
                response = test_client.get("/v1/models")
                # Should return 503 since http_client is not initialized in test
                assert response.status_code == 503

        finally:
            src.api.server._memory_system_instance = original_memory


class TestMemoryIntegration:
    """Test memory integration functionality."""

    def test_memory_disabled_endpoints(self):
        """Test endpoints when memory is disabled."""
        from unittest.mock import patch
        import src.api.server

        # Create a test client with no memory system
        original_memory = getattr(
            src.api.server, '_memory_system_instance', None)
        src.api.server._memory_system_instance = None

        with patch('src.api.server.get_memory_system', return_value=None):
            client = TestClient(app)

            # Test memory endpoints return 503
            response = client.get("/memory/stats")
            assert response.status_code == 503

            response = client.post("/memory/search", json={"query": "test"})
            assert response.status_code == 503

            response = client.post("/memory/clear")
            assert response.status_code == 503

            response = client.get("/memory/config")
            assert response.status_code == 503

        # Restore
        src.api.server._memory_system_instance = original_memory


class TestMiddleware:
    """Test memory middleware functionality."""

    @pytest.mark.asyncio
    async def test_middleware_process_request(self):
        """Test request processing with memory context."""
        from src.api.middleware import MemoryMiddleware

        mock_memory = Mock()
        mock_memory.query_memory.return_value = [
            {"content": "Relevant memory", "score": 0.9}
        ]

        middleware = MemoryMiddleware(mock_memory)

        # Test chat completion request
        body = json.dumps({
            "messages": [
                {"role": "user", "content": "How do I use Python?"}
            ]
        }).encode()

        headers = {"content-type": "application/json"}

        result = await middleware.process_request(
            "/v1/chat/completions", "POST", body, headers
        )

        assert "body" in result
        assert "headers" in result
        assert "original_query" in result

        # Verify memory was queried
        mock_memory.query_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_process_response(self):
        """Test response processing for memory encoding."""
        from src.api.middleware import MemoryMiddleware

        mock_memory = Mock()
        mock_memory.add_experience = Mock()

        middleware = MemoryMiddleware(mock_memory)

        # Test response processing
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": "What is AI?"}
            ]
        }).encode()

        response_body = json.dumps({
            "choices": [
                {"message": {"content": "AI stands for Artificial Intelligence"}}
            ]
        }).encode()

        context = {"original_query": "What is AI?"}

        await middleware.process_response(
            "/v1/chat/completions", "POST", request_body, response_body, context
        )

        # Verify experience was encoded
        mock_memory.add_experience.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
