"""Memory endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas.models import MemoryRequest, MemoryResponse, RecallRequest, RecallResponse
from memory.manager import MemoryManager

router = APIRouter(prefix="/memory", tags=["memory"])

# Global memory manager
memory_manager: MemoryManager = None


def init_memory_router(mem_manager: MemoryManager):
    """Initialize memory router."""
    global memory_manager
    memory_manager = mem_manager


@router.post("/store", response_model=MemoryResponse)
async def store_memory(request: MemoryRequest):
    """Store information in long-term memory.

    Args:
        request: Memory storage request

    Returns:
        Memory ID
    """
    try:
        memory_id = memory_manager.remember(
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            metadata=request.metadata,
        )

        return MemoryResponse(
            memory_id=memory_id,
            status="success",
        )

    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recall", response_model=RecallResponse)
async def recall_memory(request: RecallRequest):
    """Recall relevant memories.

    Args:
        request: Recall request

    Returns:
        Retrieved memories
    """
    try:
        results = memory_manager.recall(
            query=request.query,
            n_results=request.n_results,
        )

        return RecallResponse(
            results=results["long_term"],
            count=len(results["long_term"]),
        )

    except Exception as e:
        logger.error(f"Memory recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_memory_stats():
    """Get memory statistics.

    Returns:
        Memory stats
    """
    try:
        stats = memory_manager.get_stats()
        return {"status": "success", "stats": stats}

    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
