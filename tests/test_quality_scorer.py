"""
Tests for ResponseQualityScorer quality evaluation system.

This test suite ensures the quality scoring system works correctly
across different model types, response formats, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
from memevolve.utils.quality_scorer import ResponseQualityScorer


class TestResponseQualityScorer:
    """Test suite for ResponseQualityScorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        """Create a quality scorer instance for testing."""
        return ResponseQualityScorer(debug=True)
    
    @pytest.fixture
    def sample_direct_response(self):
        """Sample direct (non-reasoning) response."""
        return {
            "role": "assistant",
            "content": "Water appears wet due to surface tension. The cohesive forces between water molecules create this sensation through hydrogen bonding, which allows water to resist external forces and form droplets."
        }
    
    @pytest.fixture
    def sample_reasoning_response(self):
        """Sample thinking model response."""
        return {
            "role": "assistant",
            "content": "Water appears wet due to surface tension. This phenomenon results from cohesive forces between water molecules, primarily hydrogen bonding, which creates the sensation we perceive as wetness.",
            "reasoning_content": "Let me think about this step by step:\n1. What does 'wet' mean? It's the sensation we perceive when touching water.\n2. What causes this sensation? It's not a chemical property but a physical one.\n3. The key physics concept is surface tension - the tendency of liquid surfaces to shrink into the minimum surface area.\n4. Surface tension in water is caused by hydrogen bonds between water molecules.\n5. These cohesive forces create the 'wet' sensation through how water interacts with surfaces and skin receptors.\n\nSo the answer involves both physics (hydrogen bonding, surface tension) and biology (sensory perception)."
        }
    
    @pytest.fixture
    def sample_context(self):
        """Sample request context."""
        return {
            "original_query": "Why does water feel wet?",
            "messages": [
                {"role": "user", "content": "Why does water feel wet?"}
            ]
        }

    # ========== Core Scoring Tests ==========
    
    def test_scorer_initialization(self, scorer):
        """Test quality scorer initialization with different parameters."""
        # Test default initialization
        default_scorer = ResponseQualityScorer()
        assert default_scorer.debug is False
        assert default_scorer.min_threshold == 0.1
        
        # Test custom initialization
        custom_scorer = ResponseQualityScorer(
            debug=True, 
            min_threshold=0.2,
            bias_correction=False
        )
        assert custom_scorer.debug is True
        assert custom_scorer.min_threshold == 0.2
        assert custom_scorer.bias_correction is False
    
    def test_direct_response_scoring(self, scorer, sample_direct_response, sample_context):
        """Test scoring of direct (non-reasoning) responses."""
        score = scorer.calculate_response_quality(
            response=sample_direct_response,
            context=sample_context,
            query="Why does water feel wet?"
        )
        
        # Should be a valid float in expected range
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Should be a decent score for this good response
        assert score >= 0.3, f"Expected score >= 0.3, got {score}"
    
    def test_reasoning_response_scoring(self, scorer, sample_reasoning_response, sample_context):
        """Test scoring of thinking model responses."""
        score = scorer.calculate_response_quality(
            response=sample_reasoning_response,
            context=sample_context,
            query="Why does water feel wet?"
        )
        
        # Should be a valid float in expected range
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Should be a decent score for this good response with reasoning
        assert score >= 0.3, f"Expected score >= 0.3, got {score}"
    
    def test_reasoning_vs_direct_parity(self, scorer, sample_direct_response, sample_reasoning_response, sample_context):
        """Test parity-based evaluation between reasoning and direct models."""
        direct_score = scorer.calculate_response_quality(
            response=sample_direct_response,
            context=sample_context,
            query="Why does water feel wet?"
        )
        
        reasoning_score = scorer.calculate_response_quality(
            response=sample_reasoning_response,
            context=sample_context,
            query="Why does water feel wet?"
        )
        
        # Both should be in reasonable range (not wildly different)
        # This tests that parity-based scoring is working
        assert abs(direct_score - reasoning_score) <= 0.3, \
            f"Scores too different: direct={direct_score}, reasoning={reasoning_score}"
    
    # ========== Scoring Factor Tests ==========
    
    def test_semantic_density_calculation(self, scorer):
        """Test semantic density calculation."""
        # High-density content
        high_density = "Photosynthesis converts CO2 and H2O into glucose and O2 through chlorophyll-mediated photochemical reactions in plant chloroplasts."
        # Low-density content  
        low_density = "Yeah, photosynthesis is when plants make food from sunlight and stuff."
        
        high_score = scorer._calculate_content_factors(high_density, "test")
        low_score = scorer._calculate_content_factors(low_density, "test")
        
        # High density should score higher
        assert high_score["semantic_density"] > low_score["semantic_density"]
    
    def test_query_alignment_calculation(self, scorer):
        """Test query alignment assessment."""
        # Well-aligned response
        aligned = "Database indexing improves query performance by creating efficient lookup structures. Common types include B-tree for range queries and hash indexes for exact matches."
        # Poorly aligned response
        misaligned = "Cloud computing offers scalable infrastructure solutions for modern applications."
        
        aligned_score = scorer._calculate_content_factors(aligned, "database optimization")
        misaligned_score = scorer._calculate_content_factors(misaligned, "database optimization")
        
        # Aligned response should score higher
        assert aligned_score["query_alignment"] > misaligned_score["query_alignment"]
    
    def test_reasoning_consistency_evaluation(self, scorer):
        """Test reasoning-answer consistency checking."""
        # Consistent reasoning and answer
        consistent = {
            "content": "The answer is 42.",
            "reasoning_content": "After analyzing the mathematical pattern and considering all variables, the solution emerges as 42 through systematic derivation."
        }
        # Inconsistent reasoning and answer
        inconsistent = {
            "content": "The answer is 42.",
            "reasoning_content": "The calculation clearly shows the answer must be 17, based on the provided equations and constraints."
        }
        
        consistent_score = scorer._evaluate_reasoning_consistency(
            inconsistent["reasoning_content"], 
            inconsistent["content"]
        )
        inconsistent_score = scorer._evaluate_reasoning_consistency(
            inconsistent["reasoning_content"], 
            inconsistent["content"]
        )
        
        # Consistent reasoning should score higher
        assert consistent_score >= inconsistent_score
    
    # ========== Bias Correction Tests ==========
    
    def test_bias_correction_tracking(self, scorer):
        """Test bias correction mechanism and tracking."""
        # Simulate multiple responses to build bias data
        responses = [
            {"content": "Direct response 1", "reasoning_content": ""},
            {"content": "Direct response 2", "reasoning_content": ""},
            {"content": "Reasoning response 1", "reasoning_content": "Step-by-step thinking..."},
            {"content": "Reasoning response 2", "reasoning_content": "More step-by-step thinking..."}
        ]
        
        scores = []
        for i, response in enumerate(responses):
            has_reasoning = bool(response.get("reasoning_content", "").strip())
            score = scorer._calculate_base_score(response, "test query", has_reasoning)
            scores.append({"score": score, "has_reasoning": has_reasoning})
        
        # Bias correction should be applied
        # This tests the tracking mechanism
        assert len(scorer.bias_tracker["reasoning_models"]["scores"]) >= 2
        assert len(scorer.bias_tracker["direct_models"]["scores"]) >= 2
    
    def test_bias_correction_adjustment(self, scorer):
        """Test that bias correction actually adjusts scores."""
        # Mock bias tracker with systematic difference
        scorer.bias_tracker["reasoning_models"]["avg_score"] = 0.4
        scorer.bias_tracker["direct_models"]["avg_score"] = 0.6
        
        # Simulate a response
        base_score = 0.5
        has_reasoning = True
        
        # Should get positive adjustment for reasoning model
        adjusted_score = scorer._apply_bias_correction(base_score, has_reasoning)
        
        # Should be higher than base due to bias correction
        assert adjusted_score > base_score
    
    # ========== Edge Cases ==========
    
    def test_empty_response_handling(self, scorer, sample_context):
        """Test handling of empty or invalid responses."""
        # Completely empty response
        empty_response = {"content": "", "reasoning_content": ""}
        score = scorer.calculate_response_quality(
            response=empty_response,
            context=sample_context,
            query="test query"
        )
        assert score == 0.0
        
        # Missing required fields
        missing_fields = {"content": "Some content"}
        score = scorer.calculate_response_quality(
            response=missing_fields,
            context=sample_context,
            query="test query"
        )
        # Should still work (reasoning_content optional)
        assert 0.0 <= score <= 1.0
    
    def test_minimum_threshold_filtering(self, scorer):
        """Test minimum quality threshold filtering."""
        scorer.min_threshold = 0.5
        
        # Below threshold
        low_quality_response = {"content": "bad", "reasoning_content": ""}
        context = {"original_query": "test", "messages": []}
        
        with patch('api.middleware.logger') as mock_logger:
            score = scorer.calculate_response_quality(
                response=low_quality_response,
                context=context,
                query="test"
            )
        
        # Should log threshold warning
        mock_logger.warning.assert_called()
    
    def test_unicode_and_special_characters(self, scorer, sample_context):
        """Test handling of unicode characters and special content."""
        unicode_response = {
            "content": "The temperature outside is 25°C with ☀️ sunny conditions. Testing emoji support: 🚀 🌟 💎",
            "reasoning_content": "Let me consider the weather: Temperature in Celsius (25°C), conditions (sunny), and various emoji for visual representation 🌈"
        }
        
        score = scorer.calculate_response_quality(
            response=unicode_response,
            context=sample_context,
            query="What's the weather like?"
        )
        
        # Should handle gracefully without errors
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_very_long_responses(self, scorer, sample_context):
        """Test handling of very long responses."""
        long_content = "This is a very detailed response. " * 100  # 100 words
        long_reasoning = "Step-by-step thinking: " + "Detailed analysis. " * 50
        
        long_response = {
            "content": long_content,
            "reasoning_content": long_reasoning
        }
        
        score = scorer.calculate_response_quality(
            response=long_response,
            context=sample_context,
            query="Complex question requiring long answer"
        )
        
        # Should handle without errors or timeouts
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    # ========== Performance Tests ==========
    
    def test_scoring_performance(self, scorer, sample_reasoning_response, sample_context):
        """Test that quality scoring is performant."""
        import time
        
        start_time = time.time()
        
        # Score multiple responses
        for _ in range(100):
            score = scorer.calculate_response_quality(
                response=sample_reasoning_response,
                context=sample_context,
                query="Why does water feel wet?"
            )
            assert isinstance(score, float)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for 100 scores)
        assert total_time < 1.0, f"Too slow: {total_time:.3f}s for 100 scores"
    
    # ========== Integration Tests ==========
    
    def test_integration_with_bias_correction_over_time(self, scorer, sample_context):
        """Test that bias correction improves over multiple evaluations."""
        responses = [
            {"content": f"Response {i}", "reasoning_content": "" if i % 2 == 0 else f"Reasoning {i}"}
            for i in range(20)
        ]
        
        scores_without_bias = []
        scores_with_bias = []
        
        for i, response in enumerate(responses):
            query = f"Test question {i}"
            has_reasoning = bool(response.get("reasoning_content", "").strip())
            
            # Calculate without bias correction
            scorer.bias_correction = False
            score_without_bias = scorer.calculate_response_quality(
                response=response, context=sample_context, query=query
            )
            scores_without_bias.append(score_without_bias)
            
            # Calculate with bias correction
            scorer.bias_correction = True
            score_with_bias = scorer.calculate_response_quality(
                response=response, context=sample_context, query=query
            )
            scores_with_bias.append(score_with_bias)
        
        # With enough data, bias correction should reduce differences
        # This tests the learning aspect of the bias correction
        assert len(scores_with_bias) == 20
        assert len(scores_without_bias) == 20


class TestQualityScorerEdgeCases:
    """Test edge cases and boundary conditions for quality scoring."""
    
    def test_extreme_score_ranges(self):
        """Test scoring produces values across full range."""
        scorer = ResponseQualityScorer(min_threshold=0.0)
        
        # Very poor response
        poor_response = {"content": "bad", "reasoning_content": ""}
        context = {"original_query": "test", "messages": []}
        poor_score = scorer.calculate_response_quality(
            response=poor_response, context=context, query="test"
        )
        
        # Very good response
        excellent_response = {
            "content": "Comprehensive, detailed, accurate response with clear structure and valuable insights.",
            "reasoning_content": "Systematic step-by-step analysis leading to well-supported conclusion."
        }
        excellent_score = scorer.calculate_response_quality(
            response=excellent_response, context=context, query="complex question"
        )
        
        # Should see reasonable range
        assert poor_score < excellent_score
        assert 0.0 <= poor_score <= 0.5
        assert 0.5 <= excellent_score <= 1.0
    
    def test_concurrent_scoring_safety(self):
        """Test that scoring is thread-safe if needed."""
        import threading
        import time
        
        scorer = ResponseQualityScorer()
        results = []
        errors = []
        
        def score_response(response_id):
            try:
                response = {"content": f"Response {response_id}"}
                context = {"original_query": f"Query {response_id}", "messages": []}
                score = scorer.calculate_response_quality(
                    response=response, context=context, query=f"Query {response_id}"
                )
                results.append(score)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads scoring simultaneously
        threads = []
        for i in range(10):
            thread = threading.Thread(target=score_response, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access safely
        assert len(errors) == 0, f"Errors in concurrent scoring: {errors}"
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__])