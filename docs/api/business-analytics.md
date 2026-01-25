# Business Analytics API

This reference covers MemEvolve's business analytics endpoints and executive-level metrics for ROI validation and business impact assessment.

## Overview

The Business Analytics API provides comprehensive business intelligence endpoints for executives and stakeholders to monitor MemEvolve's business value, ROI performance, and investment impact.

## 🎯 Core Endpoints

### Executive Dashboard Data

**Endpoint**: `GET /dashboard-data`  
**Purpose**: Real-time executive metrics with business intelligence

#### Request

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:11436/dashboard-data
```

#### Response

```json
{
  "executive_summary": {
    "roi_percentage": 23.5,
    "token_reduction": 15.2,
    "quality_improvement": 18.7,
    "investment_recommendation": "scale_up",
    "business_value_score": 0.78,
    "performance_rating": "excellent"
  },
  "trends": {
    "roi_trend": "increasing",
    "quality_trend": "stable",
    "performance_trend": "improving",
    "cost_efficiency_trend": "increasing"
  },
  "metrics": {
    "total_requests": 15420,
    "average_tokens_saved": 127.3,
    "quality_score_increase": 0.142,
    "cost_savings": 234.56,
    "implementation_cost": 1000.0,
    "break_even_reached": true,
    "break_even_date": "2026-01-15"
  },
  "business_impact": {
    "token_reduction_analysis": {
      "reduction_percentage": 15.2,
      "confidence_interval": [12.8, 17.6],
      "statistical_significance": 0.95,
      "total_tokens_saved": 15420,
      "cost_savings": 234.56
    },
    "quality_improvement_analysis": {
      "improvement_percentage": 18.7,
      "baseline_quality": 0.658,
      "current_quality": 0.781,
      "confidence_interval": [15.4, 22.0],
      "statistical_significance": 0.92
    },
    "roi_analysis": {
      "roi_percentage": 23.5,
      "confidence_interval": [18.2, 28.8],
      "payback_period_days": 42,
      "net_present_value": 1234.56
    }
  },
  "timestamp": "2026-01-25T15:30:00Z",
  "data_period_days": 30
}
```

### Business Impact Report

**Endpoint**: `GET /business-impact-report`  
**Purpose**: Comprehensive business impact analysis with statistical validation

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 30 | Analysis period in days |
| `format` | string | json | Response format (json, csv) |
| `include_confidence` | boolean | true | Include confidence intervals |
| `significance_level` | float | 0.05 | Statistical significance level |

#### Request

```bash
# 7-day business impact report
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:11436/business-impact-report?days=7&format=json"

# 30-day report with statistical validation
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:11436/business-impact-report?days=30&include_confidence=true&significance_level=0.05"
```

#### Response

```json
{
  "report_metadata": {
    "analysis_period": {
      "start_date": "2026-01-01T00:00:00Z",
      "end_date": "2026-01-31T23:59:59Z",
      "total_days": 30
    },
    "statistical_parameters": {
      "confidence_level": 0.95,
      "significance_level": 0.05,
      "sample_size": 15420
    }
  },
  "executive_summary": {
    "overall_roi": 23.5,
    "investment_recommendation": "scale_up",
    "key_achievements": [
      "15.2% token reduction achieved",
      "18.7% quality improvement observed",
      "Statistically significant results (p < 0.05)"
    ]
  },
  "detailed_analysis": {
    "token_efficiency": {
      "baseline_tokens_per_request": 850.5,
      "current_tokens_per_request": 720.8,
      "reduction_percentage": 15.2,
      "total_tokens_saved": 15420,
      "cost_savings": 234.56,
      "statistical_significance": {
        "p_value": 0.002,
        "confidence_interval": [12.8, 17.6],
        "is_significant": true
      }
    },
    "quality_assessment": {
      "baseline_quality_score": 0.658,
      "current_quality_score": 0.781,
      "improvement_percentage": 18.7,
      "quality_score_delta": 0.123,
      "statistical_significance": {
        "p_value": 0.015,
        "confidence_interval": [15.4, 22.0],
        "is_significant": true
      }
    },
    "roi_calculation": {
      "implementation_cost": 1000.0,
      "operational_cost": 234.56,
      "value_generated": 1234.56,
      "roi_percentage": 23.5,
      "payback_period_days": 42,
      "net_present_value": 1234.56,
      "confidence_interval": [18.2, 28.8]
    }
  },
  "trends_and_forecasts": {
    "roi_trend": {
      "direction": "increasing",
      "monthly_growth_rate": 2.3,
      "projected_roi_90_days": 28.7
    },
    "performance_trends": {
      "token_efficiency_trend": "improving",
      "quality_trend": "stable",
      "cost_efficiency_trend": "increasing"
    }
  },
  "recommendations": [
    "Scale up deployment to capture additional ROI",
    "Monitor quality trends for optimization opportunities",
    "Consider fine-tuning token efficiency parameters"
  ]
}
```

### ROI Calculation

**Endpoint**: `GET /roi-calculation`  
**Purpose**: Detailed ROI analysis with custom parameters

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `implementation_cost` | float | null | Override implementation cost |
| `cost_per_token` | float | 0.0001 | Cost per token |
| `hourly_rate` | float | 50.0 | Hourly rate for time savings |
| `analysis_period` | integer | 30 | Analysis period in days |

#### Request

```bash
# Custom ROI calculation
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:11436/roi-calculation?implementation_cost=1500&cost_per_token=0.00015"
```

#### Response

```json
{
  "roi_calculation": {
    "inputs": {
      "implementation_cost": 1500.0,
      "cost_per_token": 0.00015,
      "hourly_rate": 50.0,
      "analysis_period_days": 30
    },
    "calculations": {
      "token_savings_value": 351.3,
      "quality_improvement_value": 412.8,
      "efficiency_gains_value": 234.5,
      "total_value_generated": 998.6,
      "roi_percentage": -33.4,
      "payback_period_days": null
    },
    "breakdown": {
      "tokens_saved": 15420,
      "cost_per_token_savings": 2.313,
      "quality_score_improvement": 0.123,
      "time_saved_hours": 4.69,
      "efficiency_value": 234.5
    },
    "sensitivity_analysis": {
      "best_case_roi": 15.2,
      "worst_case_roi": -82.0,
      "breakeven_implementation_cost": 998.6
    }
  }
}
```

## 📊 Monitoring Endpoints

### Health Check with Business Analytics

**Endpoint**: `GET /health`  
**Purpose**: System health with business analytics status

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "business_analytics": {
    "enabled": true,
    "data_collection_active": true,
    "last_report_generated": "2026-01-25T15:30:00Z",
    "data_points_collected": 15420,
    "storage_usage_mb": 45.6
  },
  "auto_evolution": {
    "enabled": true,
    "last_evolution": "2026-01-24T12:15:00Z",
    "evolution_count": 5,
    "current_fitness_score": 0.781
  }
}
```

### Business Metrics Status

**Endpoint**: `GET /business-metrics-status`  
**Purpose**: Current status of business metrics collection

#### Response

```json
{
  "metrics_status": {
    "token_efficiency": {
      "tracking_enabled": true,
      "data_points": 15420,
      "last_updated": "2026-01-25T15:30:00Z",
      "collection_health": "healthy"
    },
    "quality_assessment": {
      "tracking_enabled": true,
      "data_points": 14280,
      "last_updated": "2026-01-25T15:25:00Z",
      "collection_health": "healthy"
    },
    "performance_tracking": {
      "tracking_enabled": true,
      "data_points": 15420,
      "last_updated": "2026-01-25T15:30:00Z",
      "collection_health": "healthy"
    }
  },
  "data_retention": {
    "retention_days": 30,
    "total_records": 46260,
    "storage_usage_mb": 45.6,
    "cleanup_last_run": "2026-01-25T03:00:00Z"
  }
}
```

## 🔧 Configuration Endpoints

### Business Analytics Configuration

**Endpoint**: `GET /business-analytics-config`  
**Purpose**: Current business analytics configuration

#### Response

```json
{
  "configuration": {
    "business_analytics": {
      "enabled": true,
      "retention_days": 30,
      "confidence_level": 0.95,
      "significance_level": 0.05
    },
    "cost_parameters": {
      "cost_per_token": 0.0001,
      "implementation_cost": 1000.0,
      "hourly_rate": 50.0
    },
    "business_thresholds": {
      "roi_threshold": 0.15,
      "quality_threshold": 0.10,
      "token_reduction_threshold": 0.10
    },
    "dashboard": {
      "enabled": true,
      "refresh_interval_seconds": 300,
      "cache_duration_seconds": 60
    }
  }
}
```

**Endpoint**: `PUT /business-analytics-config`  
**Purpose**: Update business analytics configuration

#### Request

```bash
curl -X PUT -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "business_analytics": {
      "retention_days": 60,
      "confidence_level": 0.99
    },
    "cost_parameters": {
      "cost_per_token": 0.00015
    }
  }' \
  http://localhost:11436/business-analytics-config
```

## 🚨 Alerting Endpoints

### Business Alerts

**Endpoint**: `GET /business-alerts`  
**Purpose**: Current business alerts and notifications

#### Response

```json
{
  "alerts": [
    {
      "id": "roi_threshold_breach",
      "severity": "warning",
      "message": "ROI has fallen below threshold of 15%",
      "current_value": 12.3,
      "threshold": 15.0,
      "timestamp": "2026-01-25T14:30:00Z",
      "recommendation": "Review configuration or increase implementation period"
    },
    {
      "id": "quality_decline",
      "severity": "info",
      "message": "Quality score trend shows slight decline",
      "current_trend": "decreasing",
      "magnitude": -0.02,
      "timestamp": "2026-01-25T13:15:00Z",
      "recommendation": "Monitor for further decline"
    }
  ],
  "alert_count": 2,
  "last_checked": "2026-01-25T15:30:00Z"
}
```

### Configure Business Alerts

**Endpoint**: `POST /business-alerts`  
**Purpose**: Configure business alert rules

#### Request

```bash
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "roi_threshold",
    "threshold": 0.15,
    "severity": "warning",
    "action": "email",
    "recipients": ["business-team@company.com"],
    "enabled": true
  }' \
  http://localhost:11436/business-alerts
```

## 📈 Data Export Endpoints

### Export Business Data

**Endpoint**: `GET /export-business-data`  
**Purpose**: Export business analytics data for external analysis

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | json | Export format (json, csv, xlsx) |
| `start_date` | string | 30 days ago | Start date (ISO format) |
| `end_date` | string | now | End date (ISO format) |
| `metrics` | string | all | Comma-separated metrics list |

#### Request

```bash
# Export last 30 days as CSV
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:11436/export-business-data?format=csv&start_date=2026-01-01"

# Export specific metrics as Excel
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:11436/export-business-data?format=xlsx&metrics=roi,token_efficiency,quality"
```

#### Response

```json
{
  "export_info": {
    "format": "csv",
    "record_count": 15420,
    "file_size_mb": 12.3,
    "download_url": "/downloads/business_data_2026-01-25.csv",
    "expires_at": "2026-01-26T15:30:00Z"
  },
  "data_summary": {
    "period": "2026-01-01 to 2026-01-25",
    "metrics_included": ["roi", "token_efficiency", "quality", "performance"],
    "total_records": 15420
  }
}
```

## 🛠️ Troubleshooting

### Error Responses

```json
{
  "error": {
    "code": "BUSINESS_ANALYTICS_DISABLED",
    "message": "Business analytics is not enabled",
    "details": {
      "required_setting": "MEMEVOLVE_BUSINESS_ANALYTICS_ENABLED=true",
      "current_setting": "false"
    }
  }
}
```

```json
{
  "error": {
    "code": "INSUFFICIENT_DATA",
    "message": "Insufficient data for analysis",
    "details": {
      "minimum_data_points": 100,
      "current_data_points": 45,
      "recommendation": "Wait for more data or extend analysis period"
    }
  }
}
```

### Common Issues

1. **No Business Data**: Ensure `MEMEVOLVE_BUSINESS_ANALYTICS_ENABLED=true`
2. **Missing Metrics**: Check data collection status via `/business-metrics-status`
3. **Invalid Parameters**: Validate request parameters and ranges
4. **Permission Errors**: Verify API key authorization

## 🔗 Related Resources

- [Business Impact Analyzer](../tools/business-impact-analyzer.md) - Comprehensive analysis tool
- [Auto-Evolution Guide](../user-guide/auto-evolution.md) - Intelligent automatic evolution
- [Performance Analyzer](../tools/performance_analyzer.md) - System performance monitoring
- [Configuration Guide](../user-guide/configuration.md) - Complete configuration reference

---

*Last updated: January 25, 2026*