"""
API routes for MemEvolve proxy server.
"""

import json
import os
import sys
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Import endpoint metrics at module level with robust path resolution
try:
    import os
    from memevolve.utils.endpoint_metrics_collector import get_endpoint_metrics_collector
    ENDPOINT_METRICS_AVAILABLE = True
except ImportError as e:
    get_endpoint_metrics_collector = None
    ENDPOINT_METRICS_AVAILABLE = False
    print(f"Warning: Endpoint metrics not available: {e}")
    print(
        f"Routes file directory: {
            os.path.dirname(__file__) if 'routes.py' in __file__ else 'unknown'}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Sys path: {sys.path[:3]}...")  # Show first few paths

router = APIRouter()


class MemoryStats(BaseModel):
    """Memory system statistics."""
    total_experiences: int
    retrieval_count: int
    last_updated: Optional[str] = None
    architecture: Optional[str] = None


class MemorySearchRequest(BaseModel):
    """Request for searching memory."""
    query: str
    limit: int = 10
    include_metadata: bool = False


class MemorySearchResult(BaseModel):
    """Memory search result."""
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@router.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats():
    """Get memory system statistics."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system:
        raise HTTPException(
            status_code=503, detail="Memory system not enabled")

    try:
        # Get basic stats from memory system
        health = memory_system.get_health_metrics()
        stats = {
            "total_experiences": health.total_units if health else 0,
            "retrieval_count": (
                len(memory_system.get_operation_log())
                if hasattr(memory_system, 'get_operation_log') else 0
            ),
            "last_updated": health.newest_unit_timestamp if health else None,
            "architecture": (
                getattr(memory_system.config, 'architecture',
                        {}).get('name', 'unknown')
                if hasattr(memory_system, 'config') else 'unknown'
            )
        }
        return MemoryStats(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.post("/memory/search", response_model=List[MemorySearchResult])
async def search_memory(request: MemorySearchRequest):
    """Search memory for relevant experiences."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system:
        raise HTTPException(
            status_code=503, detail="Memory system not enabled")

    try:
        # Retrieve relevant memories
        results = memory_system.query_memory(
            query=request.query,
            top_k=request.limit
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(MemorySearchResult(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get(
                    "metadata") if request.include_metadata else None
            ))

        return formatted_results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Memory search failed: {str(e)}")


@router.post("/memory/clear")
async def clear_memory():
    """Clear all memory contents."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system:
        raise HTTPException(
            status_code=503, detail="Memory system not enabled")

    try:
        memory_system.clear_operation_log()
        return {"message": "Memory operation log cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear memory: {str(e)}")


@router.get("/memory/config")
async def get_memory_config():
    """Get current memory configuration."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system or not hasattr(memory_system, 'config'):
        raise HTTPException(
            status_code=503, detail="Memory configuration not available")

    return memory_system.config.__dict__


# Evolution endpoints
@router.post("/evolution/start")
async def start_evolution():
    """Start evolution process."""
    from .server import evolution_manager

    if not evolution_manager:
        raise HTTPException(status_code=503, detail="Evolution not enabled")

    if evolution_manager.start_evolution():
        return {"message": "Evolution started successfully"}
    else:
        raise HTTPException(
            status_code=409, detail="Evolution already running")


@router.post("/evolution/stop")
async def stop_evolution():
    """Stop evolution process."""
    from .server import evolution_manager

    if not evolution_manager:
        raise HTTPException(status_code=503, detail="Evolution not enabled")

    if evolution_manager.stop_evolution():
        return {"message": "Evolution stopped successfully"}
    else:
        raise HTTPException(status_code=409, detail="Evolution not running")


@router.get("/evolution/status")
async def get_evolution_status():
    """Get evolution status."""
    from .server import evolution_manager

    if not evolution_manager:
        raise HTTPException(status_code=503, detail="Evolution not enabled")

    return evolution_manager.get_status()


@router.post("/evolution/record-request")
async def record_api_request(time_seconds: float, success: bool = True):
    """Record an API request for evolution metrics."""
    from .server import evolution_manager

    if evolution_manager:
        evolution_manager.record_api_request(time_seconds, success)

    return {"message": "Request recorded"}


@router.post("/evolution/record-retrieval")
async def record_memory_retrieval(time_seconds: float, success: bool = True):
    """Record a memory retrieval for evolution metrics."""
    from .server import evolution_manager

    if evolution_manager:
        evolution_manager.record_memory_retrieval(time_seconds, success)

    return {"message": "Retrieval recorded"}


def create_web_fallback_html(endpoint_name: str) -> str:
    """Create generic fallback HTML for missing web endpoints."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemEvolve API - {endpoint_name.title()} Not Found</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #1a1a1a;
            color: #e9ecef;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .error-container {{
            text-align: center;
            max-width: 600px;
            padding: 40px;
            background: #2d2d2d;
            border-radius: 12px;
            border: 1px solid #495057;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }}
        .error-title {{
            color: #fa5252;
            font-size: 2em;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        .error-message {{
            font-size: 1.1em;
            line-height: 1.6;
            margin-bottom: 30px;
            color: #adb5bd;
        }}
        .error-path {{
            background: #343a40;
            padding: 15px;
            border-radius: 8px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 14px;
            color: #4dabf7;
            margin: 20px 0;
            border: 1px solid #495057;
        }}
        .error-suggestion {{
            font-size: 1em;
            color: #40c057;
            background: rgba(64, 192, 87, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #40c057;
        }}
        .error-suggestion strong {{
            color: #40c057;
        }}
        .error-suggestion code {{
            background: #343a40;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <h1 class="error-title">🚫 {endpoint_name.title()} Not Found</h1>
        <div class="error-message">
            The <strong>{endpoint_name}</strong> interface could not be loaded.
        </div>
        <div class="error-path">
            http://localhost:11436/{endpoint_name}
        </div>
        <div class="error-suggestion">
            <strong>Solution:</strong> Check that the files exist in <code>web/{endpoint_name}/</code> directory
        </div>
    </div>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the health dashboard HTML page."""
    try:
        with open("web/dashboard/dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        # Fallback to generic error HTML
        html_content = create_web_fallback_html("dashboard")

    return HTMLResponse(content=html_content)


@router.get("/dashboard-data")
@router.get("/dashboard-data")
async def get_dashboard_data():
    """Get dashboard data as JSON for AJAX updates."""

    if not ENDPOINT_METRICS_AVAILABLE:
        return {"error": "Endpoint metrics not available - install required dependencies"}

    try:
        if get_endpoint_metrics_collector is None:
            return {"error": "Endpoint metrics collector not available"}

        # Get metrics collector
        metrics_collector = get_endpoint_metrics_collector()

        # Get comprehensive endpoint statistics
        upstream_stats = metrics_collector.get_endpoint_stats('upstream', limit=100)
        memory_stats = metrics_collector.get_endpoint_stats('memory', limit=100)
        embedding_stats = metrics_collector.get_endpoint_stats('embedding', limit=100)
        pipeline_stats = metrics_collector.get_request_pipeline_stats(limit=1000)

        # Generate executive dashboard data using new metrics
        dashboard_data = {
            "endpoint_statistics": {
                "upstream": upstream_stats,
                "memory": memory_stats,
                "embedding": embedding_stats},
            "pipeline_analysis": pipeline_stats,
            "real_time_metrics": {
                "timestamp": time.time(),
                "requests_processed": pipeline_stats.get(
                    "pipeline_overview",
                    {}).get(
                        "total_requests_analyzed",
                        0),
                "current_roi": pipeline_stats.get(
                    "business_impact",
                    {}).get(
                        "average_roi_score",
                        0),
                "trend_indicators": {
                    "token_efficiency": "improving" if pipeline_stats.get(
                        "token_economics",
                        {}).get(
                            "token_efficiency_ratio",
                            0) > 0.7 else "stable",
                    "time_performance": "stable" if pipeline_stats.get(
                        "performance_metrics",
                        {}).get(
                            "average_memory_overhead_ms",
                            0) < 200 else "degrading",
                    "quality_impact": "stable"}}}

        return dashboard_data
    except Exception as e:
        return {"error": f"Failed to load dashboard data: {str(e)}"}


@router.get("/web/dashboard/dashboard.css", response_class=HTMLResponse)
async def dashboard_css():
    """Serve dashboard CSS file."""
    try:
        with open("web/dashboard/dashboard.css", "r", encoding="utf-8") as f:
            css_content = f.read()
        return HTMLResponse(content=css_content, media_type="text/css")
    except FileNotFoundError:
        return HTMLResponse(
            content="/* Dashboard CSS not found - check web/dashboard/dashboard.css */",
            media_type="text/css")


@router.get("/web/dashboard/dashboard.js", response_class=HTMLResponse)
async def dashboard_js():
    """Serve dashboard JavaScript file."""
    try:
        with open("web/dashboard/dashboard.js", "r", encoding="utf-8") as f:
            js_content = f.read()
        return HTMLResponse(content=js_content, media_type="application/javascript")
    except FileNotFoundError:
        return HTMLResponse(
            content="// Dashboard JS not found - check web/dashboard/dashboard.js",
            media_type="application/javascript")


# Docs endpoints - serve from web/docs directory
@router.get("/docs", response_class=HTMLResponse)
async def docs_index():
    """Serve API documentation."""
    try:
        with open("web/docs/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        # Fallback to generic error HTML
        return HTMLResponse(content=create_web_fallback_html("docs"))


@router.get("/redoc", response_class=HTMLResponse)
async def docs_redoc():
    """Serve ReDoc API documentation."""
    try:
        with open("web/redoc/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        # Fallback to generic error HTML
        return HTMLResponse(content=create_web_fallback_html("redoc"))


@router.get("/openapi.json")
async def docs_openapi():
    """Serve OpenAPI JSON specification."""
    try:
        with open("web/docs/openapi.json", "r", encoding="utf-8") as f:
            content = f.read()
        return JSONResponse(content=json.loads(content))
    except FileNotFoundError:
        # Fallback - return empty for now
        return JSONResponse(content={"info": {"title": "MemEvolve API", "version": "0.1.0"}})
