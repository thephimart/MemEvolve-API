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

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


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
        std_savings = statistics.stdev(savings)
        
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