"""
Integration tests for full MemEvolve pipeline validation.

These tests ensure end-to-end functionality works correctly:
1. API request -> memory retrieval -> quality scoring -> response -> storage
2. Memory scores flow through entire pipeline
3. Quality scoring integrates properly with memory system
4. No regressions in core functionality
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from api.server import app
from api.middleware import MemoryMiddleware
from memory_system import MemorySystem, MemorySystemConfig
from components.retrieve.base import RetrievalResult


class TestFullPipelineIntegration:
    """Test complete integration of MemEvolve pipeline components."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_memory_config(self):
        """Create mock configuration for testing."""
        return MemorySystemConfig(
            storage_backend=None,  # Mocked
            default_retrieval_top_k=3
        )
    
    @pytest.fixture
    def sample_memory_data(self):
        """Sample memory data for comprehensive testing."""
        return [
            {
                "id": "python_optimization",
                "type": "lesson",
                "content": "Python optimization techniques include using built-in functions, avoiding unnecessary loops, and leveraging list comprehensions for better performance.",
                "tags": ["python", "optimization", "performance"],
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "id": "database_indexing",
                "type": "skill", 
                "content": "Database indexing improves query performance by creating efficient lookup structures. B-tree indexes work well for range queries, while hash indexes excel at exact matches.",
                "tags": ["database", "indexing", "performance"],
                "timestamp": "2024-01-01T11:00:00Z"
            },
            {
                "id": "caching_strategies",
                "type": "tool",
                "content": "Implement Redis caching with TTL keys to reduce database load. Cache frequently accessed data with appropriate expiration times.",
                "tags": ["redis", "caching", "performance"],
                "timestamp": "2024-01-01T12:00:00Z"
            }
        ]
    
    @pytest.fixture
    def sample_retrieval_results(self, sample_memory_data):
        """Sample RetrievalResult objects with realistic scores."""
        return [
            RetrievalResult(
                unit_id="python_optimization",
                unit=sample_memory_data[0],
                score=0.923,
                metadata={
                    "semantic_score": 0.923,
                    "keyword_score": 0.0,
                    "semantic_rank": 0,
                    "keyword_rank": None
                }
            ),
            RetrievalResult(
                unit_id="database_indexing", 
                unit=sample_memory_data[1],
                score=0.856,
                metadata={
                    "semantic_score": 0.856,
                    "keyword_score": 0.1,
                    "semantic_rank": 1,
                    "keyword_rank": 2
                }
            ),
            RetrievalResult(
                unit_id="caching_strategies",
                unit=sample_memory_data[2],
                score=0.734,
                metadata={
                    "semantic_score": 0.734,
                    "keyword_score": 0.2,
                    "semantic_rank": 2,
                    "keyword_rank": 1
                }
            )
        ]
    
    # ========== End-to-End API Pipeline Tests ==========
    
    @patch('api.server.get_memory_system')
    @patch('memory_system.MemorySystem._initialize_storage')
    @patch('memory_system.MemorySystem._initialize_encoding')
    @patch('memory_system.MemorySystem._initialize_retrieval')
    @patch('memory_system.MemorySystem._initialize_management')
    def test_complete_api_pipeline_with_memory_scoring(self, mock_init_management, mock_init_retrieval,
                                                   mock_init_encoding, mock_init_storage,
                                                   mock_get_memory_system, mock_memory_config,
                                                   sample_retrieval_results):
        """Test complete API pipeline from request to response."""
        # Create and mock memory system
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock retrieval context
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = sample_retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage and encoding
        mock_storage = Mock()
        mock_storage.store.return_value = "new_unit_id"
        memory_system.storage = mock_storage
        
        mock_encoder = Mock()
        mock_encoder.encode_experience.return_value = {
            "type": "lesson",
            "content": "New experience from API interaction",
            "tags": ["api", "test"]
        }
        memory_system.encoder = mock_encoder
        
        # Mock get_memory_system to return our configured system
        mock_get_memory_system.return_value = memory_system
        
        # Mock upstream API response
        mock_upstream_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Based on the retrieved context, Python optimization involves using efficient data structures, built-in functions, and avoiding unnecessary computations. Database indexing and caching are complementary strategies for improving overall application performance."
                }
            }],
            "usage": {"completion_tokens": 50, "prompt_tokens": 100}
        }
        
        client = TestClient(app)
        
        # Make API request
        with patch('httpx.Client.post') as mock_post:
            mock_post.return_value.json.return_value = mock_upstream_response
            mock_post.return_value.status_code = 200
            
            response = client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "How can I optimize my Python application performance?"}]
            })
        
        # Verify successful response
        assert response.status_code == 200
        
        response_data = response.json()
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        
        # Verify memory system was queried
        mock_retrieval_context.retrieve.assert_called_once()
        
        # Verify experience was encoded and stored
        mock_encoder.encode_experience.assert_called_once()
        mock_storage.store.assert_called_once()
    
    def test_memory_scores_propagate_through_pipeline(self, mock_get_memory_system, mock_memory_config, sample_retrieval_results):
        """Test that memory scores propagate correctly through entire pipeline."""
        # Set up memory system with scored results
        memory_system = MemorySystem(mock_memory_config)
        
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = sample_retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        mock_storage.store.return_value = "new_unit_id"
        memory_system.storage = mock_storage
        
        mock_encoder = Mock()
        mock_encoder.encode_experience.return_value = {
            "type": "lesson",
            "content": "Pipeline test experience",
            "tags": ["test"]
        }
        memory_system.encoder = mock_encoder
        
        mock_get_memory_system.return_value = memory_system
        
        # Create middleware instance
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        middleware.memory_system = memory_system
        
        # Test request processing
        request_context = {
            "original_query": "test query for score propagation",
            "messages": [{"role": "user", "content": "test query for score propagation"}]
        }
        
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": "Response with relevant context."
                }
            }]
        }
        
        # Process the complete pipeline
        asyncio.run(middleware.process_response(request_context, response_data))
        
        # Verify memory system returned scores
        retrieved_memories = memory_system.query_memory("test query for score propagation")
        
        # Check that scores are included in results
        for i, memory in enumerate(retrieved_memories):
            assert "score" in memory, f"Score missing from memory {i}"
            assert memory["score"] == sample_retrieval_results[i].score
            assert "retrieval_metadata" in memory
            assert memory["retrieval_metadata"] == sample_retrieval_results[i].metadata
    
    # ========== Quality Scoring Integration Tests ==========
    
    @patch('api.server.get_memory_system')
    def test_quality_scoring_integration_with_memory_scores(self, mock_get_memory_system, mock_memory_config):
        """Test integration between quality scoring and memory scores."""
        # Create memory system
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock high-quality memory retrieval
        high_quality_memories = [
            RetrievalResult(
                unit_id="quality_memory_1",
                unit={
                    "id": "quality_memory_1",
                    "content": "Highly relevant, accurate information",
                    "type": "lesson",
                    "tags": ["quality", "relevant"]
                },
                score=0.934,
                metadata={"semantic_score": 0.934}
            )
        ]
        
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = high_quality_memories
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        mock_storage.store.return_value = "quality_unit"
        memory_system.storage = mock_storage
        
        mock_encoder = Mock()
        memory_system.encoder = mock_encoder
        
        mock_get_memory_system.return_value = memory_system
        
        # Mock quality scorer to return high quality
        with patch('utils.quality_scorer.ResponseQualityScorer') as mock_scorer_class:
            mock_scorer = Mock()
            mock_scorer.calculate_response_quality.return_value = 0.876
            mock_scorer_instance = mock_scorer.return_value
            mock_scorer_instance.calculate_response_quality.return_value = 0.876
            
            # Create middleware with mocked quality scorer
            middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
            middleware.quality_scorer = mock_scorer_instance
            middleware.memory_system = memory_system
            
            # Test request processing
            request_context = {
                "original_query": "How do I implement effective caching?",
                "messages": [{"role": "user", "content": "How do I implement effective caching?"}]
            }
            
            response_data = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Implement effective caching using Redis with appropriate TTL strategies, cache invalidation patterns, and hierarchical caching for different data types.",
                        "reasoning_content": "Let me think step by step: 1. Consider cache types, 2. Evaluate cache strategies, 3. Plan implementation..."
                    }
                }]
            }
            
            # Process response
            asyncio.run(middleware.process_response(request_context, response_data))
            
            # Verify quality scorer was called with correct parameters
            mock_scorer_instance.calculate_response_quality.assert_called_once()
            call_args = mock_scorer_instance.calculate_response_quality.call_args
            assert "response" in call_args[1]
            assert "context" in call_args[1]
            assert "query" in call_args[1]
            
            # Should get high quality score due to relevant memories and good response
            assert call_args[1]["query"] == "How do I implement effective caching?"
    
    # ========== Regression Tests ==========
    
    def test_no_score_na_in_memory_results(self, mock_get_memory_system, mock_memory_config):
        """Regression test: Ensure no 'score: N/A' in memory results."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Test with various RetrievalResult configurations
        test_cases = [
            {
                "name": "high_score_memory",
                "results": [RetrievalResult(
                    unit_id="test_high",
                    unit={"id": "test_high", "content": "High quality content"},
                    score=0.956,
                    metadata={"semantic_score": 0.956}
                )]
            },
            {
                "name": "medium_score_memory", 
                "results": [RetrievalResult(
                    unit_id="test_medium",
                    unit={"id": "test_medium", "content": "Medium quality content"},
                    score=0.543,
                    metadata={"semantic_score": 0.543}
                )]
            },
            {
                "name": "low_score_memory",
                "results": [RetrievalResult(
                    unit_id="test_low",
                    unit={"id": "test_low", "content": "Low quality content"},
                    score=0.123,
                    metadata={"semantic_score": 0.123}
                )]
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case["name"]):
                # Mock retrieval context
                mock_retrieval_context = Mock()
                mock_retrieval_context.retrieve.return_value = case["results"]
                memory_system.retrieval_context = mock_retrieval_context
                
                # Mock storage
                mock_storage = Mock()
                memory_system.storage = mock_storage
                
                # Query memory
                results = memory_system.query_memory("test query")
                
                # Verify no N/A scores
                for result in results:
                    assert "score" in result, f"Missing score in result: {result}"
                    assert result["score"] is not None, f"Score is None in result: {result}"
                    assert result["score"] != "N/A", f"Score is 'N/A' in result: {result}"
                    
                    # Verify actual score value
                    expected_result = case["results"][0]
                    assert result["score"] == expected_result.score, \
                        f"Expected score {expected_result.score}, got {result['score']}"
    
    # ========== Performance Impact Tests ==========
    
    def test_pipeline_performance_with_new_features(self, mock_get_memory_system, mock_memory_config, sample_retrieval_results):
        """Test that new features don't significantly impact performance."""
        import time
        
        memory_system = MemorySystem(mock_memory_config)
        
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = sample_retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        mock_storage.store.return_value = "perf_test_unit"
        memory_system.storage = mock_storage
        
        mock_encoder = Mock()
        mock_encoder.encode_experience.return_value = {
            "type": "lesson",
            "content": "Performance test experience",
            "tags": ["performance"]
        }
        memory_system.encoder = mock_encoder
        
        mock_get_memory_system.return_value = memory_system
        
        # Mock quality scorer for realistic performance
        with patch('utils.quality_scorer.ResponseQualityScorer') as mock_scorer_class:
            mock_scorer = Mock()
            mock_scorer_instance = mock_scorer.return_value
            mock_scorer_instance.calculate_response_quality.return_value = 0.743
            
            middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
            middleware.quality_scorer = mock_scorer_instance
            middleware.memory_system = memory_system
            
            # Measure performance of multiple request cycles
            start_time = time.time()
            
            for i in range(50):
                request_context = {
                    "original_query": f"Performance test query {i}",
                    "messages": [{"role": "user", "content": f"Performance test query {i}"}]
                }
                
                response_data = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"Performance test response {i}",
                            "reasoning_content": f"Performance test reasoning {i}"
                        }
                    }]
                }
                
                # Process complete pipeline
                asyncio.run(middleware.process_response(request_context, response_data))
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should be reasonably fast (less than 2 seconds for 50 requests)
            assert total_time < 2.0, f"Pipeline too slow: {total_time:.3f}s for 50 requests"
            
            # Average time per request should be minimal
            avg_time = total_time / 50
            assert avg_time < 0.04, f"Average too slow: {avg_time:.6f}s per request"
    
    # ========== Error Handling and Resilience Tests ==========
    
    def test_pipeline_handles_missing_scores_gracefully(self, mock_get_memory_system, mock_memory_config):
        """Test pipeline handles edge cases with missing/invalid scores."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Test edge case with missing score
        edge_case_results = [
            RetrievalResult(
                unit_id="edge_case",
                unit={"id": "edge_case", "content": "Edge case content"},
                score=None,  # Missing score
                metadata={}
            )
        ]
        
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = edge_case_results
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        mock_get_memory_system.return_value = memory_system
        
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        middleware.memory_system = memory_system
        
        request_context = {
            "original_query": "Edge case test query",
            "messages": [{"role": "user", "content": "Edge case test query"}]
        }
        
        # Should handle missing scores gracefully
        results = memory_system.query_memory("Edge case test query")
        assert len(results) == 1
        # Should handle None score gracefully
        assert "score" in results[0]
    
    def test_pipeline_quality_scoring_fallback_behavior(self, mock_get_memory_system, mock_memory_config):
        """Test quality scoring fallback behavior."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock empty memory retrieval
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = []
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        mock_encoder = Mock()
        mock_encoder.encode_experience.return_value = {
            "type": "lesson",
            "content": "Fallback test experience",
            "tags": ["fallback", "test"]
        }
        memory_system.encoder = mock_encoder
        
        mock_get_memory_system.return_value = memory_system
        
        # Mock quality scorer with fallback behavior
        with patch('utils.quality_scorer.ResponseQualityScorer') as mock_scorer_class:
            mock_scorer = Mock()
            mock_scorer_instance = mock_scorer.return_value
            mock_scorer_instance.calculate_response_quality.return_value = 0.3  # Low quality fallback
            
            middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
            middleware.quality_scorer = mock_scorer_instance
            middleware.memory_system = memory_system
            
            request_context = {
                "original_query": "Fallback test query",
                "messages": [{"role": "user", "content": "Fallback test query"}]
            }
            
            response_data = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Fallback response with no relevant memories."
                    }
                }]
            }
            
            # Process response
            asyncio.run(middleware.process_response(request_context, response_data))
            
            # Quality scorer should still work with no memories
            mock_scorer_instance.calculate_response_quality.assert_called_once()
            
            # Should get fallback quality score
            call_args = mock_scorer_instance.calculate_response_quality.call_args
            assert call_args[1]["response"]["content"] == "Fallback response with no relevant memories."


class TestPipelineConsistency:
    """Test pipeline consistency and data integrity."""
    
    def test_memory_score_consistency_across_calls(self, mock_memory_config):
        """Test memory scores are consistent across multiple calls."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Create consistent test data
        test_memories = [
            RetrievalResult(
                unit_id="consistency_test",
                unit={"id": "consistency_test", "content": "Consistency test content"},
                score=0.789,
                metadata={"test": True}
            )
        ]
        
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = test_memories
        memory_system.retrieval_context = mock_retrieval_context
        
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        # Make multiple identical queries
        results_1 = memory_system.query_memory("consistency test")
        results_2 = memory_system.query_memory("consistency test") 
        results_3 = memory_system.query_memory("consistency test")
        
        # Results should be identical
        assert results_1 == results_2 == results_3
        
        # Scores should be consistent
        for result_set in [results_1, results_2, results_3]:
            assert len(result_set) == 1
            assert result_set[0]["score"] == 0.789
            assert result_set[0]["retrieval_metadata"]["test"] is True


if __name__ == "__main__":
    pytest.main([__file__])