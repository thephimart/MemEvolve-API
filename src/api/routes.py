"""
API routes for MemEvolve proxy server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from memory_system import MemorySystem

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
        raise HTTPException(status_code=503, detail="Memory system not enabled")

    try:
        # Get basic stats from memory system
        health = memory_system.get_health_metrics()
        stats = {
            "total_experiences": health.total_units if health else 0,
            "retrieval_count": len(memory_system.get_operation_log()) if hasattr(memory_system, 'get_operation_log') else 0,
            "last_updated": health.newest_unit_timestamp if health else None,
            "architecture": getattr(memory_system.config, 'architecture', {}).get('name', 'unknown') if hasattr(memory_system, 'config') else 'unknown'
        }
        return MemoryStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.post("/memory/search", response_model=List[MemorySearchResult])
async def search_memory(request: MemorySearchRequest):
    """Search memory for relevant experiences."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not enabled")

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
                metadata=result.get("metadata") if request.include_metadata else None
            ))

        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@router.post("/memory/clear")
async def clear_memory():
    """Clear all memory contents."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not enabled")

    try:
        memory_system.clear_operation_log()
        return {"message": "Memory operation log cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@router.get("/memory/config")
async def get_memory_config():
    """Get current memory configuration."""
    from .server import get_memory_system
    memory_system = get_memory_system()

    if not memory_system or not hasattr(memory_system, 'config'):
        raise HTTPException(status_code=503, detail="Memory configuration not available")

    return memory_system.config.__dict__