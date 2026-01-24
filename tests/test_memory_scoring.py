"""
Tests for memory scoring functionality and integration.

This test suite ensures that:
1. Memory scores are properly displayed (not "N/A")
2. RetrievalResult scores propagate through the system
3. Memory system includes scores in returned dictionaries
4. Integration between memory scoring and API layer works correctly
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from memevolve.memory_system import MemorySystem, MemorySystemConfig
from memevolve.components.retrieve.base import RetrievalResult
from memevolve.api.middleware import MemoryMiddleware


class TestMemoryScoring:
    """Test memory scoring functionality across the pipeline."""
    
    @pytest.fixture
    def mock_memory_config(self):
        """Create a mock memory configuration for testing."""
        return MemorySystemConfig(
            storage_backend=None,  # Mocked
            default_retrieval_top_k=5
        )
    
    @pytest.fixture
    def sample_memory_units(self):
        """Sample memory units with various content."""
        return [
            {
                "id": "unit_1",
                "type": "lesson",
                "content": "Python list comprehensions provide a concise way to create lists based on existing iterables.",
                "tags": ["python", "performance"],
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "id": "unit_2", 
                "type": "skill",
                "content": "Use vector databases for efficient similarity search in large-scale applications.",
                "tags": ["database", "vector", "search"],
                "timestamp": "2024-01-01T11:00:00Z"
            },
            {
                "id": "unit_3",
                "type": "tool",
                "content": "Redis caching improves API response times by storing frequently accessed data in memory.",
                "tags": ["redis", "caching", "performance"],
                "timestamp": "2024-01-01T12:00:00Z"
            }
        ]
    
    @pytest.fixture
    def sample_retrieval_results(self, sample_memory_units):
        """Sample RetrievalResult objects with scores."""
        return [
            RetrievalResult(
                unit_id="unit_1",
                unit=sample_memory_units[0],
                score=0.892,
                metadata={"semantic_score": 0.892, "keyword_score": 0.0}
            ),
            RetrievalResult(
                unit_id="unit_2",
                unit=sample_memory_units[1],
                score=0.756,
                metadata={"semantic_score": 0.756, "keyword_score": 0.1}
            ),
            RetrievalResult(
                unit_id="unit_3",
                unit=sample_memory_units[2],
                score=0.634,
                metadata={"semantic_score": 0.634, "keyword_score": 0.2}
            )
        ]
    
    # ========== Memory System Score Integration Tests ==========
    
    @patch('memory_system.MemorySystem._initialize_storage')
    @patch('memory_system.MemorySystem._initialize_encoding')
    @patch('memory_system.MemorySystem._initialize_retrieval')
    @patch('memory_system.MemorySystem._initialize_management')
    def test_memory_system_includes_scores_in_results(self, mock_init_management, mock_init_retrieval, 
                                                 mock_init_encoding, mock_init_storage, 
                                                 mock_memory_config, sample_retrieval_results):
        """Test that MemorySystem includes scores in query results."""
        # Create memory system with mocked components
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock retrieval context to return our test results
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = sample_retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage for the query
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        # Perform query
        results = memory_system.query_memory("python optimization", top_k=5)
        
        # Verify structure of returned results
        assert len(results) == 3
        for i, result in enumerate(results):
            # Should be a dictionary (not RetrievalResult object)
            assert isinstance(result, dict)
            
            # Should include score from RetrievalResult
            assert "score" in result
            assert result["score"] == sample_retrieval_results[i].score
            
            # Should include original memory unit data
            assert "id" in result
            assert "content" in result
            assert "type" in result
            assert result["id"] == sample_retrieval_results[i].unit["id"]
            
            # Should include retrieval metadata
            assert "retrieval_metadata" in result
            assert result["retrieval_metadata"] == sample_retrieval_results[i].metadata
    
    def test_memory_system_score_propagation_with_retrieve_by_ids(self, mock_memory_config, sample_memory_units):
        """Test score propagation in retrieve_by_ids method."""
        # Create memory system with mocked components
        memory_system = MemorySystem(mock_memory_config)
        
        # Create sample RetrievalResults
        retrieval_results = [
            RetrievalResult(
                unit_id="unit_1",
                unit=sample_memory_units[0],
                score=0.945,
                metadata={"semantic_score": 0.945}
            )
        ]
        
        # Mock retrieval context
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve_by_ids.return_value = retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        # Test retrieve by IDs
        results = memory_system.retrieve_by_ids(["unit_1"])
        
        # Verify score inclusion
        assert len(results) == 1
        assert "score" in results[0]
        assert results[0]["score"] == 0.945
        assert "retrieval_metadata" in results[0]
    
    # ========== Middleware Memory Score Tests ==========
    
    @patch('api.middleware.logger')
    async def test_middleware_processes_memory_scores_correctly(self, mock_logger, mock_memory_config, sample_retrieval_results):
        """Test that middleware processes memory scores correctly."""
        # Create middleware with mocked memory system
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        
        # Mock memory system to return scored results
        mock_memory_system = Mock()
        mock_memory_system.query_memory.return_value = [
            {
                "id": "unit_1",
                "content": "Python list comprehensions are efficient...",
                "score": 0.876,
                "retrieval_metadata": {"semantic_score": 0.876}
            },
            {
                "id": "unit_2",
                "content": "Vector databases enable fast search...",
                "score": 0.743,
                "retrieval_metadata": {"semantic_score": 0.743}
            }
        ]
        middleware.memory_system = mock_memory_system
        
        # Test memory logging in middleware
        request_context = {
            "original_query": "How do I optimize Python code?",
            "messages": [{"role": "user", "content": "How do I optimize Python code?"}]
        }
        
        # Inject memories
        await middleware._inject_relevant_memories(request_context)
        
        # Verify memory system was called correctly
        mock_memory_system.query_memory.assert_called_once_with(
            query="How do I optimize Python code?",
            top_k=mock_memory_config.default_retrieval_top_k
        )
        
        # Should not log "score: N/A" anymore
        # Instead, should log actual scores
        mock_logger.info.assert_any_call(
            "Retrieved memories for API request:"
        )
        
        # Check that score logging would show actual scores
        log_calls = mock_logger.info.call_args_list
        score_logs = [call for call in log_calls if "score:" in str(call)]
        assert len(score_logs) > 0
    
    async def test_middleware_uses_actual_scores_not_na(self, mock_memory_config):
        """Test that middleware uses actual scores instead of 'N/A'."""
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        
        # Mock memory system with scored results
        mock_memory_system = Mock()
        mock_memory_system.query_memory.return_value = [
            {
                "id": "unit_123",
                "content": "Test memory content",
                "score": 0.567,  # Actual score, not N/A
                "retrieval_metadata": {"semantic_score": 0.567}
            }
        ]
        middleware.memory_system = mock_memory_system
        
        request_context = {
            "original_query": "test query",
            "messages": [{"role": "user", "content": "test query"}]
        }
        
        with patch('api.middleware.logger') as mock_logger:
            await middleware._inject_relevant_memories(request_context)
            
            # Verify that scores are logged correctly
            # Should NOT contain "score: N/A"
            log_output = str(mock_logger.info.call_args_list)
            assert "score: N/A" not in log_output
            assert "score: 0.567" in log_output
    
    # ========== Score Format and Content Tests ==========
    
    def test_memory_score_ranges_are_valid(self, mock_memory_config, sample_memory_units):
        """Test that memory scores are in valid ranges."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Create results with various score ranges
        test_cases = [
            {"score": 0.0, "description": "Minimum score"},
            {"score": 0.5, "description": "Middle score"},
            {"score": 0.999, "description": "High score"},
            {"score": 1.0, "description": "Perfect score"}
        ]
        
        for case in test_cases:
            retrieval_results = [
                RetrievalResult(
                    unit_id="test_unit",
                    unit=sample_memory_units[0],
                    score=case["score"],
                    metadata={}
                )
            ]
            
            # Mock retrieval context
            mock_retrieval_context = Mock()
            mock_retrieval_context.retrieve.return_value = retrieval_results
            memory_system.retrieval_context = mock_retrieval_context
            
            # Mock storage
            mock_storage = Mock()
            memory_system.storage = mock_storage
            
            results = memory_system.query_memory("test query")
            
            # Verify score is preserved correctly
            assert results[0]["score"] == case["score"]
            assert 0.0 <= results[0]["score"] <= 1.0
    
    def test_memory_metadata_preservation(self, mock_memory_config, sample_memory_units):
        """Test that retrieval metadata is preserved correctly."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Create results with detailed metadata
        detailed_metadata = {
            "semantic_score": 0.876,
            "keyword_score": 0.2,
            "semantic_rank": 0,
            "keyword_rank": 3,
            "retrieval_time_ms": 45,
            "index_used": "ivf"
        }
        
        retrieval_results = [
            RetrievalResult(
                unit_id="unit_1",
                unit=sample_memory_units[0],
                score=0.854,
                metadata=detailed_metadata
            )
        ]
        
        # Mock retrieval context
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        results = memory_system.query_memory("test query")
        
        # Verify metadata is preserved
        assert "retrieval_metadata" in results[0]
        assert results[0]["retrieval_metadata"] == detailed_metadata
    
    # ========== Integration with Quality Scoring Tests ==========
    
    async def test_memory_scores_and_quality_scoring_integration(self, mock_memory_config):
        """Test integration between memory scores and quality scoring."""
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        
        # Mock memory system with high-score memories
        high_score_memories = [
            {
                "id": "unit_high",
                "content": "High-quality, relevant information",
                "score": 0.923,  # Very relevant memory
                "retrieval_metadata": {"semantic_score": 0.923}
            }
        ]
        
        # Mock quality scorer to return high quality
        mock_quality_scorer = Mock()
        mock_quality_scorer.calculate_response_quality.return_value = 0.856
        
        middleware.quality_scorer = mock_quality_scorer
        middleware.memory_system = Mock()
        middleware.memory_system.query_memory.return_value = high_score_memories
        
        request_context = {
            "original_query": "Complex technical question",
            "messages": [{"role": "user", "content": "Complex technical question"}]
        }
        
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Comprehensive technical answer with examples."
                }
            }]
        }
        
        # Process response
        with patch('api.middleware.logger') as mock_logger:
            quality_score = await middleware._calculate_response_quality_score(
                request_context, response_data, response_data["choices"][0]["message"]
            )
        
        # Verify quality scorer was called
        mock_quality_scorer.calculate_response_quality.assert_called_once()
        
        # Should get high quality score due to relevant memory
        assert quality_score == 0.856
    
    # ========== Error Handling and Edge Cases ==========
    
    def test_memory_scoring_with_empty_results(self, mock_memory_config):
        """Test memory scoring with no results."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock empty retrieval
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = []
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        results = memory_system.query_memory("query with no results")
        
        # Should handle empty results gracefully
        assert results == []
    
    def test_memory_scoring_with_missing_scores(self, mock_memory_config, sample_memory_units):
        """Test handling of RetrievalResults with missing scores."""
        memory_system = MemorySystem(mock_memory_config)
        
        # Create results with missing score (edge case)
        retrieval_results = [
            RetrievalResult(
                unit_id="unit_1",
                unit=sample_memory_units[0],
                score=None,  # Missing score
                metadata={}
            )
        ]
        
        # Mock retrieval context
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        # Should handle missing scores gracefully
        results = memory_system.query_memory("test query")
        assert len(results) == 1
        # Score should be None or handled appropriately
        assert "score" in results[0]
    
    # ========== Performance Tests ==========
    
    def test_memory_scoring_performance_impact(self, mock_memory_config, sample_retrieval_results):
        """Test that memory scoring doesn't significantly impact performance."""
        import time
        
        memory_system = MemorySystem(mock_memory_config)
        
        # Mock retrieval context
        mock_retrieval_context = Mock()
        mock_retrieval_context.retrieve.return_value = sample_retrieval_results
        memory_system.retrieval_context = mock_retrieval_context
        
        # Mock storage
        mock_storage = Mock()
        memory_system.storage = mock_storage
        
        # Measure performance of multiple queries
        start_time = time.time()
        
        for _ in range(100):
            results = memory_system.query_memory("test query")
            assert len(results) == 3
            for result in results:
                assert "score" in result
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for 100 queries)
        assert total_time < 1.0, f"Too slow: {total_time:.3f}s for 100 queries"
        
        # Average time per query should be minimal
        avg_time = total_time / 100
        assert avg_time < 0.01, f"Average too slow: {avg_time:.6f}s per query"


class TestMemoryScoreDisplay:
    """Test that memory scores are properly displayed in logs and API responses."""
    
    @patch('api.middleware.logger')
    async def test_memory_scores_displayed_in_middleware_logs(self, mock_logger, mock_memory_config):
        """Test that middleware logs show actual scores instead of N/A."""
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        
        # Mock memory system with scores
        scored_memories = [
            {
                "id": "unit_test_1",
                "content": "Test memory 1",
                "score": 0.789,
                "retrieval_metadata": {"semantic_score": 0.789}
            },
            {
                "id": "unit_test_2", 
                "content": "Test memory 2",
                "score": 0.654,
                "retrieval_metadata": {"semantic_score": 0.654}
            }
        ]
        
        middleware.memory_system = Mock()
        middleware.memory_system.query_memory.return_value = scored_memories
        
        request_context = {
            "original_query": "test query for score display",
            "messages": [{"role": "user", "content": "test query for score display"}]
        }
        
        # Inject memories and check log output
        await middleware._inject_relevant_memories(request_context)
        
        # Verify that scores are logged correctly
        info_calls = mock_logger.info.call_args_list
        score_display_found = False
        
        for call in info_calls:
            call_str = str(call)
            if "Retrieved memories for API request:" in call_str:
                # Should contain actual scores
                assert "score: 0.789" in call_str or "score: 0.654" in call_str
                # Should NOT contain "score: N/A"
                assert "score: N/A" not in call_str
                score_display_found = True
        
        assert score_display_found, "Memory scores not properly displayed in logs"
    
    def test_memory_unit_id_extraction_works_correctly(self, mock_memory_config):
        """Test that middleware extracts unit IDs correctly for logging."""
        middleware = MemoryMiddleware(mock_memory_config, Mock(), Mock())
        
        # Test different ID formats
        test_cases = [
            {
                "memories": [{"id": "unit_123", "score": 0.5}],
                "expected_ids": ["unit_123"]
            },
            {
                "memories": [{"unit_id": "unit_456", "id": "fallback", "score": 0.5}],
                "expected_ids": ["unit_456"]
            },
            {
                "memories": [{"id": None, "unit_id": None, "score": 0.5}],
                "expected_ids": ["memory_0"]  # Fallback format
            }
        ]
        
        for i, case in enumerate(test_cases):
            middleware.memory_system = Mock()
            middleware.memory_system.query_memory.return_value = case["memories"]
            
            request_context = {
                "original_query": f"test query {i}",
                "messages": [{"role": "user", "content": f"test query {i}"}]
            }
            
            with patch('api.middleware.logger') as mock_logger:
                import asyncio
                asyncio.run(middleware._inject_relevant_memories(request_context))
                
                # Check that unit IDs are logged correctly
                log_calls = mock_logger.info.call_args_list
                id_found = False
                
                for call in log_calls:
                    if f"#{i+1}: {case['expected_ids'][0]}" in str(call):
                        id_found = True
                        break
                
                assert id_found, f"Unit ID {case['expected_ids'][0]} not logged correctly for case {i}"


if __name__ == "__main__":
    pytest.main([__file__])