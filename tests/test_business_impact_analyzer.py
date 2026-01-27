"""
Tests for Business Impact Analyzer
"""

import json
import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open

# Import the analyzer (adjust path as needed)
import sys
sys.path.append('scripts')
from business_impact_analyzer import BusinessImpactAnalyzer


class TestBusinessImpactAnalyzer:
    """Comprehensive tests for BusinessImpactAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with temp directory."""
        temp_dir = tempfile.mkdtemp()
        return BusinessImpactAnalyzer(data_dir=temp_dir)
    
    @pytest.fixture
    def sample_request_data(self):
        """Sample request metrics data for testing."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00",
                "token_savings": 150,
                "time_impact": -0.2,
                "quality_improvement": 0.1,
                "memory_count": 3
            },
            {
                "timestamp": "2024-01-01T10:01:00", 
                "token_savings": 200,
                "time_impact": -0.1,
                "quality_improvement": 0.15,
                "memory_count": 4
            },
            {
                "timestamp": "2024-01-01T10:02:00",
                "token_savings": 50,
                "time_impact": 0.1,
                "quality_improvement": 0.05,
                "memory_count": 2
            },
            {
                "timestamp": "2024-01-01T10:03:00",
                "token_savings": -50,
                "time_impact": 0.3,
                "quality_improvement": -0.02,
                "memory_count": 1
            },
            {
                "timestamp": "2024-01-01T10:04:00",
                "token_savings": 300,
                "time_impact": -0.3,
                "quality_improvement": 0.2,
                "memory_count": 5
            }
        ]
    
    @pytest.fixture
    def good_performance_data(self):
        """Data showing good performance."""
        return [
            {
                "token_savings": 200,
                "time_impact": -0.2,
                "quality_improvement": 0.12,
                "memory_count": 3
            } for _ in range(50)
        ]
    
    @pytest.fixture
    def poor_performance_data(self):
        """Data showing poor performance."""
        return [
            {
                "token_savings": -100,
                "time_impact": 0.5,
                "quality_improvement": -0.1,
                "memory_count": 1
            } for _ in range(50)
        ]
    
    def setup_request_data(self, analyzer, data):
        """Helper to setup request data file."""
        request_file = analyzer.request_metrics_file
        os.makedirs(request_file.parent, exist_ok=True)
        with open(request_file, 'w') as f:
            json.dump(data, f)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.data_dir.exists()
        assert analyzer.significance_threshold == 0.05
        assert analyzer.min_sample_size == 30
        assert analyzer.business_weights["cost_reduction"] == 0.4
    
    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data."""
        result = analyzer.analyze_upstream_reduction_trends()
        assert "error" in result
        assert "Insufficient data" in result["error"]
        
        result = analyzer.analyze_response_time_impact()
        assert "error" in result
        
        result = analyzer.analyze_quality_enhancement()
        assert "error" in result
    
    def test_upstream_reduction_analysis_basic(self, analyzer, sample_request_data):
        """Test basic upstream reduction analysis."""
        self.setup_request_data(analyzer, sample_request_data)
        
        result = analyzer.analyze_upstream_reduction_trends()
        
        # Check structure
        assert "executive_summary" in result
        assert "quantitative_analysis" in result
        assert "financial_impact" in result
        assert "trend_analysis" in result
        assert "recommendations" in result
        
        # Check executive summary
        exec_summary = result["executive_summary"]
        assert "total_requests_analyzed" in exec_summary
        assert "success_rate" in exec_summary
        assert exec_summary["total_requests_analyzed"] == len(sample_request_data)
    
    def test_time_impact_analysis_basic(self, analyzer, sample_request_data):
        """Test basic time impact analysis."""
        self.setup_request_data(analyzer, sample_request_data)
        
        result = analyzer.analyze_response_time_impact()
        
        # Check structure
        assert "executive_summary" in result
        assert "performance_analysis" in result
        assert "user_experience" in result
        assert "trend_analysis" in result
        assert "recommendations" in result
        
        # Check values
        exec_summary = result["executive_summary"]
        assert "average_time_impact_seconds" in exec_summary
        assert "user_experience_impact" in exec_summary
        assert isinstance(exec_summary["user_experience_impact"], float)
    
    def test_quality_enhancement_analysis_basic(self, analyzer, sample_request_data):
        """Test basic quality enhancement analysis."""
        self.setup_request_data(analyzer, sample_request_data)
        
        result = analyzer.analyze_quality_enhancement()
        
        # Check structure
        assert "executive_summary" in result
        assert "quality_distribution" in result
        assert "memory_effectiveness" in result
        assert "trend_analysis" in result
        assert "recommendations" in result
        
        # Check values
        exec_summary = result["executive_summary"]
        assert "average_quality_improvement" in exec_summary
        assert "memory_injection_correlation" in exec_summary
    
    def test_good_performance_scenario(self, analyzer, good_performance_data):
        """Test with good performance data."""
        self.setup_request_data(analyzer, good_performance_data)
        
        # All analyses should succeed and show good results
        token_result = analyzer.analyze_upstream_reduction_trends()
        time_result = analyzer.analyze_response_time_impact()
        quality_result = analyzer.analyze_quality_enhancement()
        
        # Check no errors
        assert "error" not in token_result
        assert "error" not in time_result
        assert "error" not in quality_result
        
        # Check positive results
        assert token_result["executive_summary"]["success_rate"] > 90
        assert time_result["executive_summary"]["user_experience_impact"] > 0.7
        assert quality_result["executive_summary"]["average_quality_improvement"] > 0.1
    
    def test_poor_performance_scenario(self, analyzer, poor_performance_data):
        """Test with poor performance data."""
        self.setup_request_data(analyzer, poor_performance_data)
        
        token_result = analyzer.analyze_upstream_reduction_trends()
        time_result = analyzer.analyze_response_time_impact()
        quality_result = analyzer.analyze_quality_enhancement()
        
        # Check negative results
        assert token_result["executive_summary"]["success_rate"] < 50
        assert time_result["executive_summary"]["user_experience_impact"] < 0.3
        assert quality_result["executive_summary"]["average_quality_improvement"] < 0
    
    def test_roi_calculation(self, analyzer, good_performance_data):
        """Test ROI calculation."""
        self.setup_request_data(analyzer, good_performance_data)
        
        result = analyzer.calculate_overall_roi()
        
        # Check structure
        assert "executive_summary" in result
        assert "value_breakdown" in result
        assert "roi_metrics" in result
        assert "risk_assessment" in result
        assert "strategic_recommendations" in result
        
        # Check ROI metrics
        roi_metrics = result["roi_metrics"]
        assert "roi_percentage" in roi_metrics
        assert "payback_months" in roi_metrics
        assert "5_year_npv" in roi_metrics
        assert "irr" in roi_metrics
    
    def test_executive_summary_generation(self, analyzer, good_performance_data):
        """Test executive summary generation."""
        self.setup_request_data(analyzer, good_performance_data)
        
        result = analyzer.generate_executive_summary()
        
        # Check structure
        assert "executive_dashboard" in result
        assert "business_insights" in result
        assert "strategic_recommendations" in result
        assert "risk_mitigation" in result
        
        # Check dashboard
        dashboard = result["executive_dashboard"]
        assert "key_metrics" in dashboard
        assert "status_indicators" in dashboard
        
        # Check key metrics
        metrics = dashboard["key_metrics"]
        assert "business_value_created" in metrics
        assert "roi_percentage" in metrics
        assert "annual_cost_savings" in metrics
    
    # Test individual helper methods
    def test_sustainability_score_calculation(self, analyzer):
        """Test sustainability score calculation."""
        # High success rate, consistent savings
        savings = [100, 110, 105, 95, 120]
        no_savings = [0, 0, 0, 0, 0]
        score = analyzer._calculate_sustainability_score(savings, no_savings)
        assert score > 0.8
        
        # Low success rate
        savings = [10, 15, 5]
        no_savings = [100, 95, 120]
        score = analyzer._calculate_sustainability_score(savings, no_savings)
        assert score < 0.5
    
    def test_volatility_index_calculation(self, analyzer):
        """Test volatility index calculation."""
        # Consistent savings
        consistent_savings = [100, 105, 95, 110, 90]
        volatility = analyzer._calculate_volatility_index(consistent_savings)
        assert volatility < 0.3
        
        # Highly variable savings
        variable_savings = [10, 500, 50, 800, 20]
        volatility = analyzer._calculate_volatility_index(variable_savings)
        assert volatility > 0.5
    
    def test_time_stability_calculation(self, analyzer):
        """Test time stability calculation."""
        # Stable time impacts
        stable_impacts = [-0.2, -0.1, -0.15, -0.25, -0.2]
        stability = analyzer._calculate_time_stability(stable_impacts)
        assert stability > 0.5
        
        # Unstable impacts
        unstable_impacts = [-0.1, 1.5, -2.0, 0.8, -1.2]
        stability = analyzer._calculate_time_stability(unstable_impacts)
        assert stability < 0.5
    
    def test_confidence_level_calculation(self, analyzer, sample_request_data):
        """Test confidence level calculation."""
        # With insufficient data
        confidence = analyzer._calculate_confidence_level()
        assert confidence == 0.0
        
        # With good data
        self.setup_request_data(analyzer, sample_request_data * 10)  # 50 requests
        confidence = analyzer._calculate_confidence_level()
        assert confidence > 0.3
    
    def test_risk_assessment(self, analyzer, good_performance_data):
        """Test risk assessment functions."""
        self.setup_request_data(analyzer, good_performance_data)
        
        roi_volatility = analyzer._assess_roi_volatility()
        assert "volatility_level" in roi_volatility
        assert "risk_factors" in roi_volatility
        assert "stability_indicators" in roi_volatility
        
        impl_risk = analyzer._assess_implementation_risk()
        assert "overall_risk_level" in impl_risk
        assert "technical_risks" in impl_risk
        assert "operational_risks" in impl_risk
        assert "business_risks" in impl_risk
    
    def test_recommendations_generation(self, analyzer, sample_request_data):
        """Test various recommendation generators."""
        self.setup_request_data(analyzer, sample_request_data)
        
        # Test recommendation generators
        risk_strategies = analyzer._generate_risk_mitigation_strategies()
        assert len(risk_strategies) > 10
        assert isinstance(risk_strategies, list)
        
        monitoring = analyzer._specify_monitoring_requirements()
        assert "real_time_metrics" in monitoring
        assert "daily_reports" in monitoring
        assert "weekly_analysis" in monitoring
        
        success_criteria = analyzer._define_success_criteria()
        assert "financial_criteria" in success_criteria
        assert "performance_criteria" in success_criteria
        assert "quality_criteria" in success_criteria
    
    def test_break_even_date_estimation(self, analyzer):
        """Test break-even date estimation."""
        # Positive value scenario
        date = analyzer._estimate_break_even_date(10000, 24000)  # 1 year payback
        assert date != "Never (negative cash flow)"
        
        # Negative value scenario
        date = analyzer._estimate_break_even_date(10000, -1000)
        assert date == "Never (negative cash flow)"
    
    def test_edge_cases(self, analyzer):
        """Test edge cases and error handling."""
        # Empty data
        assert analyzer._calculate_sustainability_score([], []) == 0.5
        assert analyzer._calculate_volatility_index([]) == 0.0
        assert analyzer._calculate_time_stability([]) == 0.5
        
        # Single data point
        assert analyzer._calculate_volatility_index([100]) == 0.0
        assert analyzer._calculate_time_stability([0.1]) == 0.5


def test_run_business_impact_analysis():
    """Integration test - run full business impact analysis."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create comprehensive test data
    test_data = []
    for i in range(100):
        test_data.append({
            "timestamp": f"2024-01-{i%30+1:02d}T10:{i%60:02d}:00",
            "token_savings": 150 + (i % 100) - 50,  # Range: 100-200
            "time_impact": -0.1 + (i % 40) * 0.01,  # Range: -0.1 to 0.3
            "quality_improvement": 0.05 + (i % 30) * 0.01,  # Range: 0.05 to 0.35
            "memory_count": 2 + (i % 4)  # Range: 2-5
        })
    
    # Setup analyzer with test data
    analyzer = BusinessImpactAnalyzer(data_dir=temp_dir)
    request_file = analyzer.request_metrics_file
    os.makedirs(request_file.parent, exist_ok=True)
    with open(request_file, 'w') as f:
        json.dump(test_data, f)
    
    # Run all analyses
    token_analysis = analyzer.analyze_upstream_reduction_trends()
    time_analysis = analyzer.analyze_response_time_impact()
    quality_analysis = analyzer.analyze_quality_enhancement()
    roi_analysis = analyzer.calculate_overall_roi()
    executive_summary = analyzer.generate_executive_summary()
    
    # Basic assertions
    assert "error" not in token_analysis
    assert "error" not in time_analysis
    assert "error" not in quality_analysis
    assert "error" not in roi_analysis
    
    # Check that executive summary has all expected sections
    assert "executive_dashboard" in executive_summary
    assert "business_insights" in executive_summary
    assert "strategic_recommendations" in executive_summary
    
    print("✅ Full business impact analysis test passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_run_business_impact_analysis()