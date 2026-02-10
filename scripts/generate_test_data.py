#!/usr/bin/env python3
"""
Generate test data for Business Impact Analyzer
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import random
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Generate test data script initialized")


def generate_test_data(output_dir="./data", num_requests=1000, performance_level="good"):
    """
    Generate realistic test data for business impact analyzer testing.
    
    Args:
        output_dir: Directory to save test data
        num_requests: Number of requests to generate
        performance_level: "good", "mixed", or "poor"
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance parameters
    if performance_level == "good":
        token_savings_mean = 200
        token_savings_std = 50
        time_impact_mean = -0.2  # Negative = faster
        time_impact_std = 0.1
        quality_improvement_mean = 0.12
        quality_improvement_std = 0.05
        success_rate = 0.85
    elif performance_level == "poor":
        token_savings_mean = -50
        token_savings_std = 100
        time_impact_mean = 0.3  # Positive = slower
        time_impact_std = 0.2
        quality_improvement_mean = -0.05
        quality_improvement_std = 0.1
        success_rate = 0.25
    else:  # mixed
        token_savings_mean = 50
        token_savings_std = 150
        time_impact_mean = 0.0
        time_impact_std = 0.3
        quality_improvement_mean = 0.02
        quality_improvement_std = 0.1
        success_rate = 0.55
    
    # Generate requests
    requests = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_requests):
        # Generate realistic values with some correlation
        is_successful = random.random() < success_rate
        
        if is_successful:
            # Successful request
            token_savings = max(0, random.gauss(token_savings_mean, token_savings_std))
            time_impact = min(0, random.gauss(time_impact_mean, time_impact_std))
            quality_improvement = max(0, random.gauss(quality_improvement_mean, quality_improvement_std))
            memory_count = random.randint(2, 6)
        else:
            # Failed request
            token_savings = min(0, random.gauss(-50, 100))
            time_impact = max(0.1, random.gauss(abs(time_impact_mean), time_impact_std))
            quality_improvement = min(0, random.gauss(-0.02, quality_improvement_std))
            memory_count = random.randint(0, 2)
        
        # Add some realistic variation
        hour_of_day = (base_time + timedelta(minutes=i*10)).hour
        if 9 <= hour_of_day <= 17:  # Business hours
            token_savings *= 1.2
            quality_improvement *= 1.1
        
        request = {
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "token_savings": round(token_savings, 1),
            "time_impact": round(time_impact, 3),
            "quality_improvement": round(quality_improvement, 4),
            "memory_count": memory_count,
            "request_id": f"req_{i+1:06d}",
            "session_id": f"session_{(i//10)+1:04d}",
            "query_type": random.choice(["general", "technical", "creative", "analytical"]),
            "context_length": random.randint(100, 2000),
            "response_length": random.randint(200, 3000)
        }
        
        requests.append(request)
    
    # Save request metrics
    request_file = Path(output_dir) / "request_level_metrics.json"
    with open(request_file, 'w') as f:
        json.dump(requests, f, indent=2)
    
    # Generate summary metrics
    total_savings = sum(r["token_savings"] for r in requests if r["token_savings"] > 0)
    avg_savings = total_savings / len([r for r in requests if r["token_savings"] > 0]) if requests else 0
    success_count = len([r for r in requests if r["token_savings"] > 0])
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_requests": len(requests),
        "successful_requests": success_count,
        "success_rate": (success_count / len(requests) * 100) if requests else 0,
        "total_tokens_saved": total_savings,
        "average_tokens_saved": avg_savings,
        "performance_level": performance_level,
        "date_range": {
            "start": requests[0]["timestamp"] if requests else None,
            "end": requests[-1]["timestamp"] if requests else None
        }
    }
    
    summary_file = Path(output_dir) / "test_data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Generated {len(requests)} test requests in {output_dir}")
    print(f"📊 Success rate: {summary['success_rate']:.1f}%")
    print(f"💰 Total tokens saved: {total_savings:,.0f}")
    print(f"📈 Average tokens saved per successful request: {avg_savings:.0f}")
    print(f"📁 Data saved to: {request_file}")
    
    return request_file


def run_demo_analysis():
    """Run a demo of the business impact analyzer with generated data."""
    print("🚀 Running Business Impact Analyzer Demo")
    print("=" * 50)
    
    # Import analyzer
    import sys
    sys.path.append('scripts')
    from business_impact_analyzer import BusinessImpactAnalyzer
    
    # Generate test data
    print("\n📝 Generating test data...")
    data_file = generate_test_data(num_requests=500, performance_level="mixed")
    
    # Run analysis
    print("\n🔍 Running business impact analysis...")
    analyzer = BusinessImpactAnalyzer()
    
    try:
        # Run individual analyses
        print("\n💰 Analyzing upstream reduction trends...")
        token_analysis = analyzer.analyze_upstream_reduction_trends()
        
        print("\n⏱️ Analyzing response time impact...")
        time_analysis = analyzer.analyze_response_time_impact()
        
        print("\n📈 Analyzing quality enhancement...")
        quality_analysis = analyzer.analyze_quality_enhancement()
        
        print("\n💼 Calculating overall ROI...")
        roi_analysis = analyzer.calculate_overall_roi()
        
        print("\n📊 Generating executive summary...")
        executive_summary = analyzer.generate_executive_summary()
        
        # Display key results
        print("\n" + "=" * 50)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 50)
        
        if "executive_summary" in token_analysis:
            exec_sum = token_analysis["executive_summary"]
            print(f"💰 Cost Savings Success Rate: {exec_sum.get('success_rate', 0):.1f}%")
            print(f"📊 Statistically Significant: {exec_sum.get('statistically_significant', False)}")
            print(f"🎯 Business Verdict: {exec_sum.get('business_verdict', 'N/A')}")
        
        if "executive_summary" in time_analysis:
            exec_sum = time_analysis["executive_summary"]
            print(f"⚡ Average Time Impact: {exec_sum.get('average_time_impact_seconds', 0):.3f}s")
            print(f"😊 User Experience Score: {exec_sum.get('user_experience_impact', 0):.3f}")
        
        if "executive_summary" in quality_analysis:
            exec_sum = quality_analysis["executive_summary"]
            print(f"📈 Quality Improvement: {exec_sum.get('average_quality_improvement', 0):.1%}")
        
        if "executive_summary" in roi_analysis:
            exec_sum = roi_analysis["executive_summary"]
            print(f"💸 ROI Percentage: {exec_sum.get('roi_percentage', 0):.1f}%")
            print(f"⏰ Payback Period: {exec_sum.get('payback_period_months', 0):.1f} months")
            print(f"✅ Investment Worthwhile: {exec_sum.get('investment_worthwhile', False)}")
        
        # Show insights
        if "business_insights" in executive_summary:
            insights = executive_summary["business_insights"]
            print("\n💡 KEY INSIGHTS:")
            for insight in insights:
                print(f"   {insight}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data for Business Impact Analyzer")
    parser.add_argument("--count", type=int, default=1000, help="Number of requests to generate")
    parser.add_argument("--performance", choices=["good", "mixed", "poor"], default="mixed", 
                       help="Performance level of generated data")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run demo analysis")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_analysis()
    else:
        generate_test_data(args.output, args.count, args.performance)