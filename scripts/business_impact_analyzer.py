"""
Business Impact Analyzer for MemEvolve-API
=======================================

Advanced analysis engine that answers CRITICAL BUSINESS QUESTIONS:
1. Is memory system ACTUALLY reducing upstream API costs?
2. Is memory system IMPROVING response quality?
3. Is memory system ENHANCING response times?
4. What is the overall ROI of the memory system?

Generates executive-friendly insights and technical recommendations
for optimizing memory system business value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)


class BusinessImpactAnalyzer:
    """
    Executive-level analysis of memory system business impact.
    
    Goes beyond basic metrics to answer: "Is this worth it?"
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.metrics_file = self.data_dir / "comprehensive_metrics.json"
        self.request_metrics_file = self.data_dir / "request_level_metrics.json"
        
        # Analysis configuration
        self.significance_threshold = 0.05  # 5% significance level
        self.min_sample_size = 30  # Minimum requests for statistical significance
        self.break_even_months = 6  # Target break-even timeframe
        
        # Business priorities (configurable)
        self.business_weights = {
            "cost_reduction": 0.4,    # 40% weight on cost savings
            "quality_improvement": 0.3,  # 30% weight on quality
            "time_impact": 0.3          # 30% weight on speed
        }
        
    def analyze_upstream_reduction_trends(self) -> Dict[str, Any]:
        """
        CORE QUESTION: Is memory actually saving tokens/costs?
        
        Analyzes:
        - Token reduction statistical significance
        - Cost savings trends
        - Sustainability of reductions
        - Break-even analysis
        """
        
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No request metrics data available"}
        
        if len(request_data) < self.min_sample_size:
            return {"error": f"Insufficient data: need {self.min_sample_size} requests, have {len(request_data)}"}
        
        # Extract token savings data
        token_savings = [req["token_savings"] for req in request_data if req["token_savings"] > 0]
        no_savings = [req["token_savings"] for req in request_data if req["token_savings"] <= 0]
        
        # Statistical analysis
        avg_savings = statistics.mean(token_savings) if token_savings else 0
        median_savings = statistics.median(token_savings) if token_savings else 0
        savings_std = statistics.stdev(token_savings) if len(token_savings) > 1 else 0
        
        # Significance testing
        positive_percentage = len(token_savings) / len(request_data) * 100
        negative_percentage = len(no_savings) / len(request_data) * 100
        
        # Is reduction statistically significant?
        reduction_significant = self._test_token_reduction_significance(request_data)
        
        # Cost calculations (using GPT-4 pricing as example)
        input_cost_per_1k = 0.001  # $0.001 per 1k input tokens
        total_savings = sum(token_savings)
        monthly_savings = (total_savings * input_cost_per_1k / 1000) * (30 / len(request_data))
        annual_savings = monthly_savings * 12
        
        # Trend analysis
        savings_trend = self._analyze_savings_trend(request_data)
        
        # Break-even analysis
        memory_system_cost = self._estimate_memory_system_cost()
        break_even_months = memory_system_cost / max(monthly_savings, 0.001) if monthly_savings > 0 else float('inf')
        
        return {
            "executive_summary": {
                "total_requests_analyzed": len(request_data),
                "requests_with_savings": len(token_savings),
                "requests_with_no_savings": len(no_savings),
                "success_rate": positive_percentage,
                "statistically_significant": reduction_significant,
                "business_verdict": self._get_cost_reduction_verdict(positive_percentage, reduction_significant, savings_trend)
            },
            
            "quantitative_analysis": {
                "average_tokens_saved": avg_savings,
                "median_tokens_saved": median_savings,
                "savings_consistency": savings_std,
                "max_savings_per_request": max(token_savings) if token_savings else 0,
                "total_tokens_saved": total_savings
            },
            
            "financial_impact": {
                "monthly_cost_savings_usd": monthly_savings,
                "annual_cost_savings_usd": annual_savings,
                "memory_system_estimated_cost": memory_system_cost,
                "break_even_months": break_even_months,
                "break_even_achieved": break_even_months <= self.break_even_months
            },
            
            "trend_analysis": {
                "savings_trend": savings_trend,
                "sustainability_score": self._calculate_sustainability_score(token_savings, no_savings),
                "volatility_index": self._calculate_volatility_index(token_savings)
            },
            
            "recommendations": self._generate_cost_optimization_recommendations(
                positive_percentage, reduction_significant, savings_trend, savings_std
            )
        }
    
    def analyze_response_time_impact(self) -> Dict[str, Any]:
        """
        CORE QUESTION: Is memory improving or degrading response times?
        
        Analyzes:
        - Net time impact of memory system
        - Time saved vs memory overhead
        - User experience impact
        - Performance bottlenecks
        """
        
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No request metrics data available"}
        
        if len(request_data) < self.min_sample_size:
            return {"error": f"Insufficient data: need {self.min_sample_size} requests"}
        
        # Extract time impact data
        time_impacts = [req["time_impact"] for req in request_data if "time_impact" in req]
        
        if not time_impacts:
            return {"error": "No time impact data available"}
        
        # Statistical analysis
        avg_time_impact = statistics.mean(time_impacts)
        median_time_impact = statistics.median(time_impacts)
        time_std = statistics.stdev(time_impacts) if len(time_impacts) > 1 else 0
        
        # Categorize impacts
        faster_responses = len([t for t in time_impacts if t < 0])  # Negative = faster
        slower_responses = len([t for t in time_impacts if t > 0])   # Positive = slower
        neutral_responses = len([t for t in time_impacts if abs(t) < 0.1])  # Within 100ms
        
        # Significance testing
        impact_significant = self._test_time_impact_significance(time_impacts)
        
        # User experience analysis
        user_impact_score = self._calculate_user_experience_impact(time_impacts)
        
        # Time trend analysis
        time_trend = self._analyze_time_trend(request_data)
        
        return {
            "executive_summary": {
                "average_time_impact_seconds": avg_time_impact,
                "user_experience_impact": user_impact_score,
                "faster_responses_percentage": (faster_responses / len(time_impacts)) * 100,
                "slower_responses_percentage": (slower_responses / len(time_impacts)) * 100,
                "statistically_significant": impact_significant,
                "business_verdict": self._get_time_impact_verdict(avg_time_impact, user_impact_score, time_trend)
            },
            
            "performance_analysis": {
                "median_time_impact": median_time_impact,
                "time_consistency": time_std,
                "worst_case_impact": max(time_impacts),
                "best_case_impact": min(time_impacts),
                "response_distribution": {
                    "significantly_faster": len([t for t in time_impacts if t < -0.5]),
                    "moderately_faster": len([t for t in time_impacts if -0.5 <= t < -0.1]),
                    "neutral": len([t for t in time_impacts if -0.1 <= t <= 0.1]),
                    "moderately_slower": len([t for t in time_impacts if 0.1 < t <= 0.5]),
                    "significantly_slower": len([t for t in time_impacts if t > 0.5])
                }
            },
            
            "user_experience": {
                "experience_score": user_impact_score,
                "perceptible_impact_percentage": len([t for t in time_impacts if abs(t) > 0.2]) / len(time_impacts) * 100,
                "frustration_risk": len([t for t in time_impacts if t > 1.0]) / len(time_impacts) * 100,
                "delight_potential": len([t for t in time_impacts if t < -0.5]) / len(time_impacts) * 100
            },
            
            "trend_analysis": {
                "time_trend": time_trend,
                "stability_score": self._calculate_time_stability(time_impacts),
                "optimization_opportunities": self._identify_time_optimization_opportunities(time_impacts, request_data)
            },
            
            "recommendations": self._generate_time_optimization_recommendations(
                avg_time_impact, user_impact_score, faster_responses, slower_responses
            )
        }
    
    def analyze_quality_enhancement(self) -> Dict[str, Any]:
        """
        CORE QUESTION: Is memory actually improving response quality?
        
        Analyzes:
        - Quality improvement statistical significance  
        - Quality vs memory injection correlation
        - Quality sustainability over time
        - ROI of quality improvements
        """
        
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No request metrics data available"}
        
        if len(request_data) < self.min_sample_size:
            return {"error": f"Insufficient data: need {self.min_sample_size} requests"}
        
        # Extract quality improvement data
        quality_improvements = [req["quality_improvement"] for req in request_data if "quality_improvement" in req]
        
        if not quality_improvements:
            return {"error": "No quality improvement data available"}
        
        # Statistical analysis
        avg_improvement = statistics.mean(quality_improvements)
        median_improvement = statistics.median(quality_improvements)
        improvement_std = statistics.stdev(quality_improvements) if len(quality_improvements) > 1 else 0
        
        # Categorize improvements
        significant_improvements = len([q for q in quality_improvements if q > 0.1])  # >10% improvement
        degradations = len([q for q in quality_improvements if q < -0.05])   # >5% degradation
        neutral_changes = len([q for q in quality_improvements if -0.05 <= q <= 0.1])
        
        # Significance testing
        quality_significant = self._test_quality_improvement_significance(quality_improvements)
        
        # Memory injection correlation
        injection_correlation = self._analyze_memory_quality_correlation(request_data)
        
        # Quality trend analysis
        quality_trend = self._analyze_quality_trend(request_data)
        
        return {
            "executive_summary": {
                "average_quality_improvement": avg_improvement,
                "significant_improvements_percentage": (significant_improvements / len(quality_improvements)) * 100,
                "quality_degradations_percentage": (degradations / len(quality_improvements)) * 100,
                "statistically_significant": quality_significant,
                "memory_injection_correlation": injection_correlation,
                "business_verdict": self._get_quality_verdict(avg_improvement, quality_significant, quality_trend)
            },
            
            "quality_distribution": {
                "median_improvement": median_improvement,
                "improvement_consistency": improvement_std,
                "best_improvement": max(quality_improvements),
                "worst_degradation": min(quality_improvements),
                "improvement_range": max(quality_improvements) - min(quality_improvements)
            },
            
            "memory_effectiveness": {
                "injection_quality_correlation": injection_correlation["correlation_coefficient"],
                "optimal_memory_count": injection_correlation["optimal_memory_count"],
                "diminishing_returns_point": injection_correlation["diminishing_returns_point"],
                "quality_saturation_threshold": injection_correlation["saturation_threshold"]
            },
            
            "trend_analysis": {
                "quality_trend": quality_trend,
                "sustainability_score": self._calculate_quality_sustainability(quality_improvements),
                "learning_rate": self._calculate_quality_learning_rate(request_data)
            },
            
            "recommendations": self._generate_quality_optimization_recommendations(
                avg_improvement, quality_significant, injection_correlation, quality_trend
            )
        }
    
    def calculate_overall_roi(self) -> Dict[str, Any]:
        """
        CORE QUESTION: Overall ROI of the memory system?
        
        Combines:
        - Token cost savings
        - Quality improvements value
        - Time impact value
        - System implementation costs
        """
        
        # Get individual analyses
        token_analysis = self.analyze_upstream_reduction_trends()
        time_analysis = self.analyze_response_time_impact()  
        quality_analysis = self.analyze_quality_enhancement()
        
        if "error" in token_analysis or "error" in time_analysis or "error" in quality_analysis:
            return {"error": "Insufficient data for ROI analysis"}
        
        # Extract key metrics
        annual_cost_savings = token_analysis["financial_impact"]["annual_cost_savings_usd"]
        avg_time_impact = time_analysis["executive_summary"]["average_time_impact_seconds"]
        avg_quality_improvement = quality_analysis["executive_summary"]["average_quality_improvement"]
        
        # Calculate weighted business value
        cost_value = annual_cost_savings
        
        # Time value (user time is valuable - estimate $50/hour)
        time_value_per_request = -avg_time_impact * (50/3600)  # Negative impact is positive value
        annual_time_value = time_value_per_request * token_analysis["executive_summary"]["total_requests_analyzed"] * (365/30)
        
        # Quality value (harder to quantify, use business weights)
        quality_value_score = max(0, avg_quality_improvement) * 100  # Convert to 0-100 scale
        annual_quality_value = quality_value_score * token_analysis["executive_summary"]["total_requests_analyzed"] * (365/30)
        
        # Total business value
        total_annual_value = (
            self.business_weights["cost_reduction"] * cost_value +
            self.business_weights["time_impact"] * annual_time_value +
            self.business_weights["quality_improvement"] * annual_quality_value
        )
        
        # System costs
        implementation_cost = self._estimate_memory_system_cost()
        annual_maintenance_cost = implementation_cost * 0.2  # 20% annual maintenance
        
        # ROI calculations
        net_annual_value = total_annual_value - annual_maintenance_cost
        roi_percentage = (net_annual_value / implementation_cost) * 100 if implementation_cost > 0 else 0
        payback_months = implementation_cost / (net_annual_value / 12) if net_annual_value > 0 else float('inf')
        
        return {
            "executive_summary": {
                "total_annular_business_value": total_annual_value,
                "net_annular_value": net_annual_value,
                "roi_percentage": roi_percentage,
                "payback_period_months": payback_months,
                "investment_worthwhile": roi_percentage > 20 and payback_months < 12,  # 20% ROI, <12 month payback
                "business_verdict": self._get_overall_roi_verdict(roi_percentage, payback_months)
            },
            
            "value_breakdown": {
                "cost_savings_value": cost_value,
                "time_savings_value": annual_time_value,
                "quality_improvement_value": annual_quality_value,
                "total_system_cost": implementation_cost,
                "annual_maintenance_cost": annual_maintenance_cost,
                "net_value_after_costs": net_annual_value
            },
            
            "roi_metrics": {
                "roi_percentage": roi_percentage,
                "payback_months": payback_months,
                "break_even_month": self._estimate_break_even_date(implementation_cost, net_annual_value),
                "5_year_npv": self._calculate_npv(net_annual_value, implementation_cost, 5),
                "irr": self._calculate_irr(net_annual_value, implementation_cost)
            },
            
            "risk_assessment": {
                "confidence_level": self._calculate_confidence_level(),
                "volatility_risk": self._assess_roi_volatility(),
                "implementation_risk": self._assess_implementation_risk(),
                "mitigation_strategies": self._generate_risk_mitigation_strategies()
            },
            
            "strategic_recommendations": self._generate_strategic_roi_recommendations(
                roi_percentage, payback_months, cost_value, annual_time_value, annual_quality_value
            )
        }
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate C-suite friendly summary of memory system business impact."""
        
        token_analysis = self.analyze_upstream_reduction_trends()
        time_analysis = self.analyze_response_time_impact()
        quality_analysis = self.analyze_quality_enhancement()
        roi_analysis = self.calculate_overall_roi()
        
        # Extract key executive metrics
        key_metrics = {
            "business_value_created": roi_analysis.get("executive_summary", {}).get("total_annular_business_value", 0),
            "roi_percentage": roi_analysis.get("executive_summary", {}).get("roi_percentage", 0),
            "payback_months": roi_analysis.get("executive_summary", {}).get("payback_period_months", float('inf')),
            "annual_cost_savings": token_analysis.get("financial_impact", {}).get("annual_cost_savings_usd", 0),
            "quality_improvement": quality_analysis.get("executive_summary", {}).get("average_quality_improvement", 0),
            "user_experience_impact": time_analysis.get("executive_summary", {}).get("user_experience_impact", 0),
            "investment_worthwhile": roi_analysis.get("executive_summary", {}).get("investment_worthwhile", False)
        }
        
        # Generate actionable insights
        insights = self._generate_executive_insights(key_metrics, token_analysis, time_analysis, quality_analysis, roi_analysis)
        
        return {
            "executive_dashboard": {
                "key_metrics": key_metrics,
                "status_indicators": {
                    "cost_savings": self._get_status_indicator(token_analysis.get("executive_summary", {}), "success_rate", 70),
                    "quality_improvement": self._get_status_indicator(quality_analysis.get("executive_summary", {}), "significant_improvements_percentage", 30),
                    "user_experience": self._get_status_indicator(time_analysis.get("executive_summary", {}), "faster_responses_percentage", 50),
                    "roi_health": self._get_status_indicator(roi_analysis.get("executive_summary", {}), "roi_percentage", 20)
                }
            },
            
            "business_insights": insights,
            
            "strategic_recommendations": {
                "immediate_actions": self._prioritize_immediate_actions(token_analysis, time_analysis, quality_analysis),
                "investment_decisions": self._recommend_investment_decisions(roi_analysis),
                "optimization_opportunities": self._identify_optimization_opportunities(token_analysis, time_analysis, quality_analysis),
                "scaling_considerations": self._assess_scaling_readiness(token_analysis, time_analysis, quality_analysis)
            },
            
            "risk_mitigation": {
                "primary_risks": self._identify_primary_business_risks(token_analysis, time_analysis, quality_analysis, roi_analysis),
                "mitigation_strategies": self._generate_risk_mitigation_strategies(),
                "monitoring_requirements": self._specify_monitoring_requirements(),
                "success_criteria": self._define_success_criteria()
            }
        }
    
    # Helper methods for statistical analysis and insights
    def _test_token_reduction_significance(self, request_data: List[Dict]) -> bool:
        """Test if token reduction is statistically significant."""
        # Simple t-test implementation
        savings = [req["token_savings"] for req in request_data if "token_savings" in req]
        if len(savings) < 20:
            return False
        
        mean_savings = statistics.mean(savings)
        if len(savings) > 1:
            std_savings = statistics.stdev(savings)
        else:
            std_savings = 0
        
        # Handle division by zero
        if std_savings == 0:
            return mean_savings > 0
        
        # Test if mean is significantly greater than 0
        t_statistic = mean_savings / (std_savings / (len(savings) ** 0.5))
        return abs(t_statistic) > 1.96  # 95% confidence
    
    def _analyze_savings_trend(self, request_data: List[Dict]) -> str:
        """Analyze if savings are improving, stable, or declining over time."""
        if len(request_data) < 20:
            return "insufficient_data"
        
        # Split data into halves
        mid_point = len(request_data) // 2
        first_half = request_data[:mid_point]
        second_half = request_data[mid_point:]
        
        first_avg = statistics.mean([req["token_savings"] for req in first_half if "token_savings" in req])
        second_avg = statistics.mean([req["token_savings"] for req in second_half if "token_savings" in req])
        
        improvement_rate = (second_avg - first_avg) / max(abs(first_avg), 1)
        
        if improvement_rate > 0.1:
            return "improving"
        elif improvement_rate < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_cost_reduction_verdict(self, success_rate: float, significant: bool, trend: str) -> str:
        """Generate business verdict for cost reduction."""
        if success_rate > 70 and significant and trend in ["improving", "stable"]:
            return "EXCELLENT: Significant cost reduction with improving trends"
        elif success_rate > 50 and significant:
            return "GOOD: Meaningful cost reduction achieved"
        elif success_rate > 30:
            return "MODERATE: Some cost reduction but needs optimization"
        else:
            return "POOR: Cost reduction not achieved - reconsider approach"
    
    def _estimate_memory_system_cost(self) -> float:
        """Estimate total cost of memory system implementation."""
        # Placeholder - should be based on actual costs
        return 10000  # $10,000 implementation cost
    
    def _generate_cost_optimization_recommendations(self, success_rate: float, significant: bool, trend: str, std_dev: float) -> List[str]:
        """Generate specific recommendations for cost optimization."""
        recommendations = []
        
        if success_rate < 50:
            recommendations.append("URGENT: Memory system is increasing costs - review relevance scoring")
        elif success_rate < 70:
            recommendations.append("Improve memory retrieval precision to increase cost savings rate")
        
        if not significant:
            recommendations.append("Increase memory injection to achieve statistically significant savings")
        
        if trend == "declining":
            recommendations.append("CRITICAL: Cost savings declining - investigate memory quality degradation")
        elif trend == "stable":
            recommendations.append("Optimize memory strategies to move from stable to improving trend")
        
        if std_dev > 50:
            recommendations.append("High volatility in cost savings - implement consistency controls")
        
        return recommendations
    
    def _calculate_sustainability_score(self, token_savings: List[float], no_savings: List[float]) -> float:
        """Calculate sustainability score of token savings (0-1 scale)."""
        if not token_savings and not no_savings:
            return 0.5  # Neutral if no data
        
        total_requests = len(token_savings) + len(no_savings)
        if total_requests == 0:
            return 0.5
        
        # Score based on success rate and consistency
        success_rate = len(token_savings) / total_requests
        
        # Calculate consistency (lower std dev = more sustainable)
        if token_savings and len(token_savings) > 1:
            mean_saving = statistics.mean(token_savings)
            if mean_saving > 0:
                std_saving = statistics.stdev(token_savings)
                consistency_score = 1 - min(std_saving / mean_saving, 1)
            else:
                consistency_score = 0
        else:
            consistency_score = 0.5
        
        # Combine success rate and consistency
        sustainability = (success_rate * 0.7) + (consistency_score * 0.3)
        return max(0.0, min(1.0, sustainability))
    
    def _calculate_volatility_index(self, token_savings: List[float]) -> float:
        """Calculate volatility index of token savings (0-1 scale, higher = more volatile)."""
        if not token_savings or len(token_savings) < 2:
            return 0.0  # No volatility with insufficient data
        
        mean_saving = statistics.mean(token_savings)
        if mean_saving == 0:
            return 1.0  # Maximum volatility if mean is zero but there are variations
        
        # Coefficient of variation as volatility measure
        std_saving = statistics.stdev(token_savings)
        volatility = std_saving / abs(mean_saving)
        
        # Scale to 0-1 range (typical CV range 0-2 for this use case)
        return min(volatility / 2.0, 1.0)
    
    def _calculate_time_stability(self, time_impacts: List[float]) -> float:
        """Calculate time impact stability score (0-1 scale, higher = more stable)."""
        if not time_impacts or len(time_impacts) < 2:
            return 0.5  # Neutral stability with insufficient data
        
        # Lower standard deviation = higher stability
        mean_impact = statistics.mean(time_impacts)
        std_impact = statistics.stdev(time_impacts)
        
        # Calculate coefficient of variation
        if mean_impact == 0:
            cv = float('inf') if std_impact > 0 else 0
        else:
            cv = abs(std_impact / mean_impact)
        
        # Convert CV to stability score (inverse relationship)
        # CV of 0 = perfect stability (score 1), CV > 1 = poor stability (score < 0.5)
        if cv == 0:
            return 1.0
        elif cv <= 0.5:
            return 1.0 - cv
        else:
            return max(0.0, 0.5 - (cv - 0.5) * 0.5)
    
    def _identify_time_optimization_opportunities(self, time_impacts: List[float], request_data: List[Dict]) -> List[str]:
        """Identify specific opportunities for time optimization."""
        opportunities = []
        
        if not time_impacts:
            return ["No time impact data available for analysis"]
        
        # Analyze time impact patterns
        avg_impact = statistics.mean(time_impacts)
        worst_cases = [t for t in time_impacts if t > 1.0]  # >1 second degradation
        
        if avg_impact > 0.5:
            opportunities.append("CRITICAL: Average response time increased by >500ms - optimize memory retrieval")
        
        if worst_cases:
            opportunities.append(f"Address {len(worst_cases)} requests with >1s slowdown - implement caching")
        
        # Look for patterns in request data
        memory_heavy_requests = [req for req in request_data if req.get("memory_count", 0) > 5]
        if memory_heavy_requests:
            opportunities.append("High memory count requests causing delays - optimize relevance scoring")
        
        # Check for time trends
        if len(time_impacts) >= 10:
            recent_impacts = time_impacts[-10:]
            older_impacts = time_impacts[:10]
            recent_avg = statistics.mean(recent_impacts)
            older_avg = statistics.mean(older_impacts)
            
            if recent_avg > older_avg + 0.2:
                opportunities.append("Performance degrading over time - investigate memory bloat")
        
        if not opportunities:
            opportunities.append("Time performance is optimized - maintain current configuration")
        
        return opportunities
    
    def _generate_time_optimization_recommendations(self, avg_impact: float, ux_score: float, faster_count: int, slower_count: int) -> List[str]:
        """Generate specific recommendations for time optimization."""
        recommendations = []
        
        total_requests = faster_count + slower_count
        if total_requests == 0:
            return ["Insufficient data for time optimization recommendations"]
        
        faster_percentage = (faster_count / total_requests) * 100
        slower_percentage = (slower_count / total_requests) * 100
        
        # Overall performance assessment
        if avg_impact > 0.5:
            recommendations.append("URGENT: Memory system significantly slowing responses - review retrieval strategy")
        elif avg_impact > 0.2:
            recommendations.append("Memory system adding noticeable latency - optimize indexing and caching")
        elif avg_impact < -0.2:
            recommendations.append("Excellent time performance - consider scaling memory injection")
        
        # User experience focus
        if ux_score < 0.4:
            recommendations.append("CRITICAL: User experience degraded - implement performance monitoring")
        elif ux_score < 0.6:
            recommendations.append("User experience needs improvement - optimize memory relevance")
        
        # Balance recommendations
        if slower_percentage > 60:
            recommendations.append("Too many slow responses - reduce memory injection volume")
        elif faster_percentage > 80:
            recommendations.append("Strong time performance - opportunity for quality enhancement")
        
        # Technical recommendations
        recommendations.append("Monitor memory retrieval latency trends")
        recommendations.append("Implement adaptive memory injection based on query complexity")
        
        return recommendations
    
    def _calculate_quality_sustainability(self, quality_improvements: List[float]) -> float:
        """Calculate sustainability score of quality improvements (0-1 scale)."""
        if not quality_improvements or len(quality_improvements) < 2:
            return 0.5  # Neutral with insufficient data
        
        # Calculate consistency and trend
        mean_improvement = statistics.mean(quality_improvements)
        
        # Positive mean improvements are more sustainable
        base_sustainability = max(0, min(1, (mean_improvement + 0.1) * 5))  # Scale to 0-1
        
        # Calculate consistency
        if len(quality_improvements) > 1:
            std_improvement = statistics.stdev(quality_improvements)
            if mean_improvement > 0:
                consistency = 1 - min(std_improvement / abs(mean_improvement), 1)
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        
        # Check trend over time
        if len(quality_improvements) >= 10:
            first_half = quality_improvements[:len(quality_improvements)//2]
            second_half = quality_improvements[len(quality_improvements)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            trend_factor = 0.0
            if second_avg > first_avg:
                trend_factor = 0.2  # Improving trend
            elif second_avg < first_avg - 0.05:
                trend_factor = -0.2  # Declining trend
        else:
            trend_factor = 0.0
        
        # Combine factors
        sustainability = (base_sustainability * 0.6) + (consistency * 0.3) + (trend_factor + 0.2) * 0.2
        return max(0.0, min(1.0, sustainability))
    
    def _calculate_quality_learning_rate(self, request_data: List[Dict]) -> float:
        """Calculate learning rate of quality improvements over time."""
        if not request_data or len(request_data) < 10:
            return 0.0  # Insufficient data for learning rate
        
        # Extract quality improvements with timestamps if available
        quality_data = []
        for req in request_data:
            if "quality_improvement" in req:
                quality_data.append(req["quality_improvement"])
        
        if len(quality_data) < 10:
            return 0.0
        
        # Calculate learning rate as improvement over time
        # Split into quarters to detect trend
        quarter_size = len(quality_data) // 4
        if quarter_size < 3:
            return 0.0
        
        q1_avg = statistics.mean(quality_data[:quarter_size])
        q2_avg = statistics.mean(quality_data[quarter_size:quarter_size*2])
        q3_avg = statistics.mean(quality_data[quarter_size*2:quarter_size*3])
        q4_avg = statistics.mean(quality_data[quarter_size*3:])
        
        # Calculate trend slope using linear regression approximation
        improvements = [q1_avg, q2_avg, q3_avg, q4_avg]
        quarters = [1, 2, 3, 4]
        
        # Simple linear regression
        n = len(improvements)
        sum_x = sum(quarters)
        sum_y = sum(improvements)
        sum_xy = sum(x*y for x, y in zip(quarters, improvements))
        sum_x2 = sum(x*x for x in quarters)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Return positive learning rate (improvement per quarter)
        return max(0.0, slope)
    
    def _generate_quality_optimization_recommendations(self, avg_improvement: float, significant: bool, correlation: Dict, trend: str) -> List[str]:
        """Generate specific recommendations for quality optimization."""
        recommendations = []
        
        # Quality improvement level recommendations
        if avg_improvement < 0:
            recommendations.append("CRITICAL: Quality degrading - review memory relevance scoring immediately")
        elif avg_improvement < 0.02:
            recommendations.append("Minimal quality improvement - enhance memory retrieval precision")
        elif avg_improvement < 0.05:
            recommendations.append("Modest quality gains - optimize memory injection strategy")
        elif avg_improvement > 0.1:
            recommendations.append("Excellent quality improvement - consider expanding memory coverage")
        
        # Statistical significance recommendations
        if not significant:
            recommendations.append("Quality improvements not statistically significant - increase sample size or effectiveness")
        
        # Correlation-based recommendations
        correlation_coef = correlation.get("correlation_coefficient", 0)
        if correlation_coef < 0.2:
            recommendations.append("Weak memory-quality correlation - improve memory relevance algorithm")
        elif correlation_coef > 0.6:
            recommendations.append("Strong memory-quality correlation - optimize injection volume")
        
        # Optimal memory count guidance
        optimal_count = correlation.get("optimal_memory_count", 3)
        recommendations.append(f"Target optimal memory count of {optimal_count} items per request")
        
        # Trend-based recommendations
        if trend == "declining":
            recommendations.append("URGENT: Quality declining - investigate memory saturation")
        elif trend == "stable":
            recommendations.append("Quality stable - implement adaptive memory for improvements")
        elif trend == "improving":
            recommendations.append("Quality improving - maintain current trajectory and monitor")
        
        # Technical recommendations
        recommendations.append("Implement quality scoring for memory relevance feedback")
        recommendations.append("Monitor quality metrics by query type and complexity")
        
        return recommendations
    
    def _estimate_break_even_date(self, implementation_cost: float, net_annual_value: float) -> str:
        """Estimate calendar date when break-even will be achieved."""
        if net_annual_value <= 0:
            return "Never (negative cash flow)"
        
        # Calculate months to break even
        monthly_value = net_annual_value / 12
        months_to_break_even = implementation_cost / monthly_value
        
        # Convert to date
        from datetime import datetime, timedelta
        current_date = datetime.now()
        break_even_date = current_date + timedelta(days=months_to_break_even * 30.44)  # Average month length
        
        return break_even_date.strftime("%B %Y")
    
    def _calculate_confidence_level(self) -> float:
        """Calculate confidence level in ROI analysis (0-1 scale)."""
        # Base confidence on data availability and quality
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.0  # No confidence without data
        
        confidence_factors = []
        
        # Sample size confidence
        sample_size = len(request_data)
        if sample_size >= 100:
            confidence_factors.append(0.9)
        elif sample_size >= 50:
            confidence_factors.append(0.7)
        elif sample_size >= 30:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Data completeness confidence
        complete_requests = 0
        for req in request_data:
            required_fields = ["token_savings", "time_impact", "quality_improvement"]
            if all(field in req for field in required_fields):
                complete_requests += 1
        
        if sample_size > 0:
            completeness_ratio = complete_requests / sample_size
            confidence_factors.append(completeness_ratio)
        
        # Data recency confidence (assuming recent data is better)
        # For now, assume moderate recency confidence
        confidence_factors.append(0.7)
        
        # Statistical significance confidence
        if sample_size >= 30:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Overall confidence is average of factors
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
    
    def _assess_roi_volatility(self) -> Dict[str, Any]:
        """Assess volatility and risk factors in ROI calculations."""
        volatility_assessment = {
            "volatility_level": "medium",
            "risk_factors": [],
            "stability_indicators": []
        }
        
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            volatility_assessment["volatility_level"] = "high"
            volatility_assessment["risk_factors"].append("Insufficient data for volatility assessment")
        
        return volatility_assessment
    
    def _assess_implementation_risk(self) -> Dict[str, Any]:
        """Assess implementation and operational risks."""
        risk_assessment = {
            "overall_risk_level": "medium",
            "technical_risks": [],
            "operational_risks": [],
            "business_risks": []
        }
        
        # Technical risks
        try:
            # Check if core components are functioning
            if not self.metrics_file.exists():
                risk_assessment["technical_risks"].append("Metrics collection not functional")
                risk_assessment["overall_risk_level"] = "high"
            
            if not self.request_metrics_file.exists():
                risk_assessment["technical_risks"].append("Request metrics not being collected")
                risk_assessment["overall_risk_level"] = "high"
        except Exception as e:
            risk_assessment["technical_risks"].append(f"File system access issues: {str(e)}")
            risk_assessment["overall_risk_level"] = "high"
        
        # Operational risks
        try:
            with open(self.request_metrics_file, 'r') as f:
                request_data = json.load(f)
            
            # Check data quality
            incomplete_requests = 0
            for req in request_data:
                required_fields = ["token_savings", "time_impact", "quality_improvement"]
                if not all(field in req for field in required_fields):
                    incomplete_requests += 1
            
            if len(request_data) > 0:
                incomplete_ratio = incomplete_requests / len(request_data)
                if incomplete_ratio > 0.3:
                    risk_assessment["operational_risks"].append("High rate of incomplete request data")
                elif incomplete_ratio > 0.1:
                    risk_assessment["operational_risks"].append("Moderate data quality issues")
        except:
            risk_assessment["operational_risks"].append("Unable to assess operational data quality")
        
        # Business risks
        risk_assessment["business_risks"].append("Memory relevance decay over time")
        risk_assessment["business_risks"].append("Increased operational complexity")
        risk_assessment["business_risks"].append("Potential dependency on memory system")
        
        # Overall risk assessment logic
        total_risks = (len(risk_assessment["technical_risks"]) + 
                      len(risk_assessment["operational_risks"]) + 
                      len(risk_assessment["business_risks"]))
        
        if len(risk_assessment["technical_risks"]) > 0:
            risk_assessment["overall_risk_level"] = "high"
        elif total_risks > 4:
            risk_assessment["overall_risk_level"] = "high"
        elif total_risks > 2:
            risk_assessment["overall_risk_level"] = "medium"
        else:
            risk_assessment["overall_risk_level"] = "low"
        
        return risk_assessment
    
    def _generate_risk_mitigation_strategies(self) -> List[str]:
        """Generate specific risk mitigation strategies."""
        strategies = []
        
        # Technical risk mitigation
        strategies.append("Implement comprehensive monitoring and alerting")
        strategies.append("Establish fallback mechanisms for memory system failures")
        strategies.append("Regular backup and recovery testing")
        strategies.append("Implement version control for memory configurations")
        
        # Operational risk mitigation
        strategies.append("Define clear SOPs for memory system maintenance")
        strategies.append("Implement automated data quality checks")
        strategies.append("Regular performance audits and optimization")
        strategies.append("Staff training on memory system management")
        
        # Business risk mitigation
        strategies.append("Gradual rollout with staged deployment")
        strategies.append("A/B testing to validate memory system impact")
        strategies.append("Regular business impact reviews")
        strategies.append("Establish ROI monitoring and reporting")
        
        # Specific risk mitigations
        strategies.append("Implement memory relevance decay detection")
        strategies.append("Create contingency plans for memory system outages")
        strategies.append("Establish performance degradation thresholds")
        strategies.append("Define clear success and failure criteria")
        
        return strategies
    
    def _generate_strategic_roi_recommendations(self, roi_percentage: float, payback_months: float, cost_value: float, time_value: float, quality_value: float) -> List[str]:
        """Generate strategic recommendations based on ROI analysis."""
        recommendations = []
        
        # ROI level recommendations
        if roi_percentage > 50:
            recommendations.append("OUTSTANDING ROI - aggressively scale memory system usage")
        elif roi_percentage > 25:
            recommendations.append("Strong ROI - expand to additional use cases")
        elif roi_percentage > 15:
            recommendations.append("Good ROI - optimize current implementation")
        elif roi_percentage > 0:
            recommendations.append("Positive ROI - improve efficiency before scaling")
        else:
            recommendations.append("Negative ROI - fundamental redesign required")
        
        # Payback period recommendations
        if payback_months < 6:
            recommendations.append("Rapid payback - prioritize investment acceleration")
        elif payback_months < 12:
            recommendations.append("Reasonable payback - proceed with planned investment")
        elif payback_months < 24:
            recommendations.append("Extended payback - reconsider implementation approach")
        else:
            recommendations.append("Excessive payback - halt investment until improvements made")
        
        # Value component analysis
        total_value = cost_value + time_value + quality_value
        if total_value > 0:
            cost_ratio = cost_value / total_value
            time_ratio = time_value / total_value
            quality_ratio = quality_value / total_value
            
            if cost_ratio > 0.6:
                recommendations.append("Value dominated by cost savings - maintain focus on efficiency")
            elif time_ratio > 0.4:
                recommendations.append("Strong time value - leverage for user experience improvements")
            elif quality_ratio > 0.4:
                recommendations.append("High quality value - invest in premium positioning")
        
        # Strategic positioning
        recommendations.append("Establish quarterly ROI review process")
        recommendations.append("Implement predictive ROI forecasting")
        recommendations.append("Create executive dashboard for value tracking")
        
        return recommendations
    
    def _prioritize_immediate_actions(self, token_analysis: Dict, time_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Prioritize immediate actions based on analysis results."""
        actions = []
        
        # Check for critical issues first
        if token_analysis.get("executive_summary", {}).get("success_rate", 0) < 30:
            actions.append("CRITICAL: Fix cost reduction - memory system increasing expenses")
        
        if time_analysis.get("executive_summary", {}).get("average_time_impact_seconds", 0) > 1.0:
            actions.append("CRITICAL: Address severe response time degradation")
        
        if quality_analysis.get("executive_summary", {}).get("average_quality_improvement", 0) < -0.05:
            actions.append("CRITICAL: Reverse quality degradation - review memory relevance")
        
        # High priority actions
        if token_analysis.get("executive_summary", {}).get("success_rate", 0) < 50:
            actions.append("HIGH: Improve memory retrieval to increase cost savings")
        
        if time_analysis.get("executive_summary", {}).get("user_experience_impact", 0) < 0.4:
            actions.append("HIGH: Optimize memory system for better user experience")
        
        # Medium priority actions
        if not token_analysis.get("executive_summary", {}).get("statistically_significant", False):
            actions.append("MEDIUM: Increase memory injection for statistical significance")
        
        if time_analysis.get("trend_analysis", {}).get("time_trend") == "declining":
            actions.append("MEDIUM: Investigate and reverse performance degradation trend")
        
        # Optimization opportunities
        if token_analysis.get("financial_impact", {}).get("break_even_months", float('inf')) > 12:
            actions.append("OPTIMIZE: Reduce break-even period through efficiency improvements")
        
        if quality_analysis.get("memory_effectiveness", {}).get("injection_quality_correlation", 0) < 0.3:
            actions.append("OPTIMIZE: Strengthen memory-quality correlation")
        
        if not actions:
            actions.append("MAINTAIN: Current performance is acceptable - continue monitoring")
        
        return actions
    
    def _recommend_investment_decisions(self, roi_analysis: Dict) -> List[str]:
        """Recommend investment decisions based on ROI analysis."""
        decisions = []
        
        roi_percentage = roi_analysis.get("executive_summary", {}).get("roi_percentage", 0)
        payback_months = roi_analysis.get("executive_summary", {}).get("payback_period_months", float('inf'))
        investment_worthwhile = roi_analysis.get("executive_summary", {}).get("investment_worthwhile", False)
        
        # Investment scale recommendations
        if roi_percentage > 50 and payback_months < 6:
            decisions.append("AGGRESSIVE INVESTMENT: Outstanding returns - double down on memory system expansion")
        elif roi_percentage > 25 and payback_months < 12:
            decisions.append("MODERATE INVESTMENT: Strong returns - continue planned investment schedule")
        elif roi_percentage > 15 and payback_months < 18:
            decisions.append("CONSERVATIVE INVESTMENT: Acceptable returns - proceed with caution")
        elif roi_percentage > 0:
            decisions.append("MINIMAL INVESTMENT: Marginal returns - optimize before expanding")
        else:
            decisions.append("HALT INVESTMENT: Negative returns - redesign required before continuing")
        
        # Timing recommendations
        if payback_months > 24:
            decisions.append("DELAY: Extended payback period - postpone major investments")
        elif payback_months > 12:
            decisions.append("STAGE: Phase investments with milestone reviews")
        else:
            decisions.append("PROCEED: Favorable payback - maintain investment momentum")
        
        # Risk-adjusted decisions
        risk_level = roi_analysis.get("risk_assessment", {}).get("confidence_level", 0.5)
        if risk_level < 0.3:
            decisions.append("HIGH RISK: Increase validation requirements before investment")
        elif risk_level > 0.7:
            decisions.append("LOW RISK: Accelerate investment decisions")
        
        # Strategic decisions
        if investment_worthwhile:
            decisions.append("STRATEGIC: Memory system aligns with business objectives")
            decisions.append("SCALE: Prepare for broader deployment across use cases")
        else:
            decisions.append("REVIEW: Reassess strategic alignment and value proposition")
        
        return decisions
    
    def _identify_optimization_opportunities(self, token_analysis: Dict, time_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Identify specific optimization opportunities across all dimensions."""
        opportunities = []
        
        # Cost optimization opportunities
        success_rate = token_analysis.get("executive_summary", {}).get("success_rate", 0)
        if success_rate < 70:
            opportunities.append("COST: Improve memory relevance scoring to increase cost savings rate")
        
        if token_analysis.get("trend_analysis", {}).get("savings_trend") == "declining":
            opportunities.append("COST: Reverse declining cost savings trend")
        
        volatility_index = token_analysis.get("trend_analysis", {}).get("volatility_index", 0)
        if volatility_index > 0.7:
            opportunities.append("COST: Reduce cost savings volatility through consistency controls")
        
        # Time optimization opportunities
        avg_time_impact = time_analysis.get("executive_summary", {}).get("average_time_impact_seconds", 0)
        if avg_time_impact > 0.2:
            opportunities.append("TIME: Optimize memory retrieval to reduce latency impact")
        
        ux_score = time_analysis.get("executive_summary", {}).get("user_experience_impact", 0)
        if ux_score < 0.6:
            opportunities.append("TIME: Enhance user experience through performance optimization")
        
        # Quality optimization opportunities
        avg_quality_improvement = quality_analysis.get("executive_summary", {}).get("average_quality_improvement", 0)
        if avg_quality_improvement < 0.05:
            opportunities.append("QUALITY: Strengthen memory-quality correlation")
        
        correlation = quality_analysis.get("memory_effectiveness", {}).get("injection_quality_correlation", 0)
        if correlation < 0.5:
            opportunities.append("QUALITY: Improve memory injection precision and relevance")
        
        # System-level opportunities
        if token_analysis.get("financial_impact", {}).get("break_even_months", float('inf')) > 12:
            opportunities.append("SYSTEM: Accelerate break-even through efficiency improvements")
        
        # Technical optimization opportunities
        opportunities.append("TECHNICAL: Implement adaptive memory injection based on context")
        opportunities.append("TECHNICAL: Optimize memory indexing and retrieval algorithms")
        opportunities.append("TECHNICAL: Implement memory quality feedback loops")
        
        return opportunities
    
    def _assess_scaling_readiness(self, token_analysis: Dict, time_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Assess readiness for scaling the memory system."""
        readiness_factors = []
        
        # Performance readiness
        success_rate = token_analysis.get("executive_summary", {}).get("success_rate", 0)
        if success_rate > 70:
            readiness_factors.append("✅ Cost performance ready for scaling")
        elif success_rate > 50:
            readiness_factors.append("⚠️ Cost performance needs optimization before scaling")
        else:
            readiness_factors.append("❌ Cost performance not ready for scaling")
        
        # User experience readiness
        ux_score = time_analysis.get("executive_summary", {}).get("user_experience_impact", 0)
        if ux_score > 0.7:
            readiness_factors.append("✅ User experience ready for scaling")
        elif ux_score > 0.5:
            readiness_factors.append("⚠️ User experience needs improvement before scaling")
        else:
            readiness_factors.append("❌ User experience not ready for scaling")
        
        # Quality readiness
        avg_quality = quality_analysis.get("executive_summary", {}).get("average_quality_improvement", 0)
        if avg_quality > 0.05:
            readiness_factors.append("✅ Quality improvements ready for scaling")
        elif avg_quality > 0:
            readiness_factors.append("⚠️ Quality improvements need enhancement before scaling")
        else:
            readiness_factors.append("❌ Quality improvements not ready for scaling")
        
        # Stability readiness
        cost_volatility = token_analysis.get("trend_analysis", {}).get("volatility_index", 0)
        if cost_volatility < 0.3:
            readiness_factors.append("✅ Cost stability ready for scaling")
        else:
            readiness_factors.append("⚠️ Reduce cost volatility before scaling")
        
        time_trend = time_analysis.get("trend_analysis", {}).get("time_trend")
        if time_trend in ["stable", "improving"]:
            readiness_factors.append("✅ Time performance stable for scaling")
        else:
            readiness_factors.append("❌ Time performance degrading - fix before scaling")
        
        # Technical readiness
        readiness_factors.append("🔧 Implement automated scaling mechanisms")
        readiness_factors.append("🔧 Establish monitoring and alerting for scaled deployment")
        readiness_factors.append("🔧 Create rollback procedures for scaling issues")
        
        # Overall readiness assessment
        positive_factors = sum(1 for factor in readiness_factors if factor.startswith("✅"))
        caution_factors = sum(1 for factor in readiness_factors if factor.startswith("⚠️"))
        negative_factors = sum(1 for factor in readiness_factors if factor.startswith("❌"))
        
        if negative_factors > 0:
            readiness_factors.insert(0, "🚫 NOT READY: Address critical issues before scaling")
        elif caution_factors > positive_factors:
            readiness_factors.insert(0, "⚠️ PARTIALLY READY: Optimize before scaling")
        else:
            readiness_factors.insert(0, "✅ READY: System prepared for scaling")
        
        return readiness_factors
    
    def _identify_primary_business_risks(self, token_analysis: Dict, time_analysis: Dict, quality_analysis: Dict, roi_analysis: Dict) -> List[str]:
        """Identify primary business risks from analyses."""
        risks = []
        
        # Financial risks
        success_rate = token_analysis.get("executive_summary", {}).get("success_rate", 0)
        if success_rate < 30:
            risks.append("FINANCIAL: Memory system increasing operational costs")
        
        monthly_savings = token_analysis.get("financial_impact", {}).get("monthly_cost_savings_usd", 0)
        if monthly_savings < 100:
            risks.append("FINANCIAL: Insufficient cost savings to justify investment")
        
        break_even_months = token_analysis.get("financial_impact", {}).get("break_even_months", float('inf'))
        if break_even_months > 24:
            risks.append("FINANCIAL: Extended break-even period impacts cash flow")
        
        # Performance risks
        avg_time_impact = time_analysis.get("executive_summary", {}).get("average_time_impact_seconds", 0)
        if avg_time_impact > 0.5:
            risks.append("PERFORMANCE: Response time degradation affects user satisfaction")
        
        ux_score = time_analysis.get("executive_summary", {}).get("user_experience_impact", 0)
        if ux_score < 0.4:
            risks.append("PERFORMANCE: Poor user experience may lead to customer churn")
        
        # Quality risks
        avg_quality = quality_analysis.get("executive_summary", {}).get("average_quality_improvement", 0)
        if avg_quality < 0:
            risks.append("QUALITY: Response quality degradation damages brand reputation")
        
        correlation = quality_analysis.get("memory_effectiveness", {}).get("injection_quality_correlation", 0)
        if correlation < 0.3:
            risks.append("QUALITY: Weak memory-quality correlation reduces effectiveness")
        
        # Operational risks
        roi_percentage = roi_analysis.get("executive_summary", {}).get("roi_percentage", 0)
        if roi_percentage < 15:
            risks.append("OPERATIONAL: Low ROI indicates inefficient resource utilization")
        
        confidence_level = roi_analysis.get("risk_assessment", {}).get("confidence_level", 0.5)
        if confidence_level < 0.5:
            risks.append("OPERATIONAL: Low confidence in metrics impedes decision making")
        
        # Strategic risks
        volatility_assessment = roi_analysis.get("risk_assessment", {})
        if volatility_assessment.get("volatility_level") == "high":
            risks.append("STRATEGIC: High volatility makes business planning difficult")
        
        # Implementation risks
        implementation_risk = roi_analysis.get("risk_assessment", {})
        if implementation_risk.get("overall_risk_level") == "high":
            risks.append("IMPLEMENTATION: Technical issues may disrupt operations")
        
        return risks
    
    def _specify_monitoring_requirements(self) -> Dict[str, List[str]]:
        """Specify detailed monitoring requirements."""
        monitoring = {
            "real_time_metrics": [
                "Memory injection success rate",
                "Memory retrieval latency",
                "Token savings per request",
                "Response time impact",
                "Quality improvement scores"
            ],
            "daily_reports": [
                "Cost savings summary",
                "Performance trend analysis", 
                "Quality improvement metrics",
                "Error rates and exceptions",
                "Memory system health status"
            ],
            "weekly_analysis": [
                "ROI trend calculation",
                "Break-even progress tracking",
                "Volatility assessment",
                "Statistical significance testing",
                "Performance benchmarking"
            ],
            "monthly_reviews": [
                "Comprehensive business impact analysis",
                "Strategic alignment assessment",
                "Scaling readiness evaluation",
                "Risk assessment updates",
                "Investment decision recommendations"
            ],
            "alerts_thresholds": [
                "Cost savings rate < 50%",
                "Response time degradation > 500ms",
                "Quality improvement < 0%",
                "Memory system error rate > 5%",
                "ROI calculation confidence < 60%"
            ],
            "executive_dashboard": [
                "Current ROI percentage",
                "Monthly cost savings",
                "User experience score",
                "Quality improvement rate",
                "Break-even timeline"
            ]
        }
        
        return monitoring
    
    def _define_success_criteria(self) -> Dict[str, Any]:
        """Define clear success criteria for the memory system."""
        criteria = {
            "financial_criteria": {
                "minimum_roi_percentage": 20,
                "maximum_payback_months": 12,
                "minimum_monthly_savings": 500,
                "cost_savings_success_rate": 70
            },
            "performance_criteria": {
                "maximum_response_time_impact": 0.2,
                "minimum_user_experience_score": 0.6,
                "minimum_faster_response_rate": 60,
                "maximum_slow_response_rate": 20
            },
            "quality_criteria": {
                "minimum_quality_improvement": 0.05,
                "minimum_quality_correlation": 0.4,
                "minimum_significant_improvement_rate": 30,
                "maximum_quality_degradation_rate": 10
            },
            "operational_criteria": {
                "minimum_uptime_percentage": 99.5,
                "maximum_error_rate": 2,
                "minimum_data_completeness": 90,
                "maximum_volatility_index": 0.5
            },
            "business_criteria": {
                "minimum_confidence_level": 0.7,
                "positive_trend_requirement": True,
                "statistical_significance_required": True,
                "scaling_readiness_score": 0.8
            },
            "milestone_criteria": {
                "pilot_success": {
                    "duration": "3 months",
                    "success_rate": 60,
                    "roi_threshold": 15
                },
                "production_ready": {
                    "duration": "6 months",
                    "success_rate": 70,
                    "roi_threshold": 20
                },
                "fully_scaled": {
                    "duration": "12 months",
                    "success_rate": 80,
                    "roi_threshold": 25
                }
            }
        }
        
        return criteria
        
        if len(request_data) < 30:
            volatility_assessment["volatility_level"] = "high"
            volatility_assessment["risk_factors"].append("Small sample size increases volatility risk")
        
        # Analyze token savings volatility
        token_savings = [req.get("token_savings", 0) for req in request_data]
        if token_savings:
            savings_cv = self._calculate_volatility_index(token_savings) * 2  # Convert back to CV
            if savings_cv > 1.0:
                volatility_assessment["volatility_level"] = "high"
                volatility_assessment["risk_factors"].append("High token savings volatility")
            elif savings_cv < 0.5:
                volatility_assessment["stability_indicators"].append("Consistent token savings")
        
        # Analyze quality improvement volatility
        quality_improvements = [req.get("quality_improvement", 0) for req in request_data if "quality_improvement" in req]
        if quality_improvements and len(quality_improvements) > 1:
            std_quality = statistics.stdev(quality_improvements)
            mean_quality = statistics.mean(quality_improvements)
            if mean_quality != 0:
                quality_cv = abs(std_quality / mean_quality)
                if quality_cv > 2.0:
                    volatility_assessment["risk_factors"].append("High quality improvement volatility")
        
        # Temporal volatility
        if len(request_data) >= 20:
            # Check for recent trend changes
            recent_data = request_data[-10:]
            older_data = request_data[:10]
            
            recent_savings = statistics.mean([req.get("token_savings", 0) for req in recent_data])
            older_savings = statistics.mean([req.get("token_savings", 0) for req in older_data])
            
            if abs(recent_savings - older_savings) > max(abs(older_savings), 10):
                volatility_assessment["risk_factors"].append("Recent performance trend changes detected")
        
        return volatility_assessment
    
    def _test_time_impact_significance(self, time_impacts: List[float]) -> bool:
        """Test if time impact is statistically significant."""
        if len(time_impacts) < 20:
            return False
        
        mean_impact = statistics.mean(time_impacts)
        std_impact = statistics.stdev(time_impacts)
        
        # Test if mean is significantly different from 0
        t_statistic = mean_impact / (std_impact / (len(time_impacts) ** 0.5))
        return abs(t_statistic) > 1.96
    
    def _calculate_user_experience_impact(self, time_impacts: List[float]) -> float:
        """Calculate user experience impact score (0-1 scale)."""
        # Negative impacts are good, positive are bad
        negative_impacts = [t for t in time_impacts if t < 0]
        positive_impacts = [t for t in time_impacts if t > 0]
        
        # Score based on percentage of noticeable improvements vs degradations
        noticeable_improvement = len([t for t in negative_impacts if abs(t) > 0.2])
        noticeable_degradation = len([t for t in positive_impacts if t > 0.2])
        
        total_requests = len(time_impacts)
        if total_requests == 0:
            return 0.5  # Neutral
        
        # 0.0 = bad, 1.0 = excellent
        ux_score = (noticeable_improvement - noticeable_degradation + total_requests) / (2 * total_requests)
        return max(0.0, min(1.0, ux_score))
    
    def _get_time_impact_verdict(self, avg_impact: float, ux_score: float, trend: str) -> str:
        """Generate business verdict for time impact."""
        if avg_impact < -0.1 and ux_score > 0.6 and trend in ["improving", "stable"]:
            return "EXCELLENT: Memory system significantly improves response speed"
        elif avg_impact < 0 and ux_score > 0.5:
            return "GOOD: Memory system improves response times"
        elif abs(avg_impact) < 0.1:
            return "NEUTRAL: Minimal impact on response times"
        else:
            return "CONCERN: Memory system degrades response speed"
    
    def _test_quality_improvement_significance(self, quality_improvements: List[float]) -> bool:
        """Test if quality improvement is statistically significant."""
        if len(quality_improvements) < 20:
            return False
        
        mean_improvement = statistics.mean(quality_improvements)
        std_improvement = statistics.stdev(quality_improvements)
        
        # Test if mean is significantly greater than 0
        t_statistic = mean_improvement / (std_improvement / (len(quality_improvements) ** 0.5))
        return t_statistic > 1.96
    
    def _analyze_memory_quality_correlation(self, request_data: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between memory injection and quality improvement."""
        # This would analyze memory count vs quality improvement correlation
        # For now, return placeholder
        return {
            "correlation_coefficient": 0.3,
            "optimal_memory_count": 3,
            "diminishing_returns_point": 5,
            "saturation_threshold": 0.2
        }
    
    def _get_quality_verdict(self, avg_improvement: float, significant: bool, trend: str) -> str:
        """Generate business verdict for quality improvement."""
        if avg_improvement > 0.1 and significant and trend in ["improving", "stable"]:
            return "EXCELLENT: Significant quality improvement with positive trends"
        elif avg_improvement > 0.05 and significant:
            return "GOOD: Meaningful quality improvement achieved"
        elif avg_improvement > 0:
            return "MODERATE: Some quality improvement but needs enhancement"
        else:
            return "POOR: No quality improvement detected - review memory relevance"
    
    def _get_overall_roi_verdict(self, roi_percentage: float, payback_months: float) -> str:
        """Generate overall ROI business verdict."""
        if roi_percentage > 50 and payback_months < 6:
            return "OUTSTANDING: Exceptional ROI with rapid payback"
        elif roi_percentage > 25 and payback_months < 12:
            return "EXCELLENT: Strong ROI with reasonable payback"
        elif roi_percentage > 15 and payback_months < 18:
            return "GOOD: Acceptable ROI with moderate payback"
        elif roi_percentage > 0:
            return "MARGINAL: Positive ROI but long payback period"
        else:
            return "POOR: Negative ROI - reconsider investment"
    
    def _generate_executive_insights(self, key_metrics: Dict, token_analysis: Dict, time_analysis: Dict, quality_analysis: Dict, roi_analysis: Dict) -> List[str]:
        """Generate C-suite level insights."""
        insights = []
        
        # Value creation insights
        if key_metrics["investment_worthwhile"]:
            insights.append(f"✅ Memory system generates ${key_metrics['annual_cost_savings']:.0f} annual cost savings with {key_metrics['roi_percentage']:.0f}% ROI")
        else:
            insights.append(f"⚠️ Memory system ROI of {key_metrics['roi_percentage']:.0f}% below target - optimization needed")
        
        # Performance insights
        if key_metrics["quality_improvement"] > 0.05:
            insights.append(f"📈 Response quality improved by {key_metrics['quality_improvement']*100:.1f}%")
        elif key_metrics["quality_improvement"] < 0:
            insights.append(f"📉 Response quality degraded by {abs(key_metrics['quality_improvement'])*100:.1f}% - urgent review needed")
        
        if key_metrics["user_experience_impact"] > 0.6:
            insights.append("⚡ User experience significantly enhanced through memory system")
        elif key_metrics["user_experience_impact"] < 0.4:
            insights.append("🐌 User experience degraded - memory system adding friction")
        
        return insights
    
    def _get_status_indicator(self, analysis_data: Dict, key: str, target: float) -> str:
        """Get status indicator (green/yellow/red) for dashboard."""
        value = analysis_data.get(key, 0)
        if value >= target:
            return "green"
        elif value >= target * 0.7:
            return "yellow"
        else:
            return "red"
    
    # Additional helper methods would be implemented...
    def _analyze_time_trend(self, request_data: List[Dict]) -> str:
        """Analyze time impact trend over time."""
        return "stable"  # Placeholder
    
    def _analyze_quality_trend(self, request_data: List[Dict]) -> str:
        """Analyze quality improvement trend over time."""
        return "stable"  # Placeholder
    
    def _calculate_npv(self, annual_value: float, initial_cost: float, years: int) -> float:
        """Calculate Net Present Value."""
        discount_rate = 0.1  # 10% discount rate
        npv = -initial_cost
        for year in range(1, years + 1):
            npv += annual_value / ((1 + discount_rate) ** year)
        return npv
    
    def _calculate_irr(self, annual_value: float, initial_cost: float) -> float:
        """Calculate Internal Rate of Return."""
        # Simplified IRR calculation
        if initial_cost == 0:
            return 0
        return (annual_value / initial_cost) * 100  # Simple approximation