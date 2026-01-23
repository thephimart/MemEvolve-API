"""
API routes for MemEvolve proxy server.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

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


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the health dashboard HTML page."""
    try:
        with open("web/dashboard/dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        # Fallback to inline HTML if file not found
        html_content = """<!DOCTYPE html>
<html>
<head><title>Dashboard Error</title></head>
<body><h1>Dashboard files not found. Please check web/dashboard/ directory.</h1></body>
</html>"""

    return HTMLResponse(content=html_content)


@router.get("/dashboard-data")
async def get_dashboard_data():
    """Get dashboard data as JSON for AJAX updates."""
    import sys
    from pathlib import Path

    try:
        # Import and use performance analyzer
        sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
        from performance_analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()
        return analyzer.get_dashboard_data()
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
        return HTMLResponse(content="/* CSS file not found */", media_type="text/css")


@router.get("/web/dashboard/dashboard.js", response_class=HTMLResponse)
async def dashboard_js():
    """Serve dashboard JavaScript file."""
    try:
        with open("web/dashboard/dashboard.js", "r", encoding="utf-8") as f:
            js_content = f.read()
        return HTMLResponse(content=js_content, media_type="application/javascript")
    except FileNotFoundError:
        return HTMLResponse(content="// JS file not found", media_type="application/javascript")


# Docs endpoints - serve from web/docs directory
@router.get("/docs", response_class=HTMLResponse)
async def docs_index():
    """Serve API documentation."""
    try:
        with open("web/docs/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        # Fallback to FastAPI's default docs
        from fastapi.openapi.docs import get_swagger_ui_html
        return get_swagger_ui_html(openapi_url="/openapi.json", title="MemEvolve API Docs")


@router.get("/redoc", response_class=HTMLResponse)
async def docs_redoc():
    """Serve ReDoc API documentation."""
    try:
        with open("web/docs/redoc.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        # Fallback to FastAPI's default redoc
        from fastapi.openapi.docs import get_redoc_html
        return get_redoc_html(openapi_url="/openapi.json", title="MemEvolve API ReDoc")


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
