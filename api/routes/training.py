"""Training endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas.models import TrainingRequest, TrainingResponse
from core.training.incremental_trainer import IncrementalTrainer

router = APIRouter(prefix="/train", tags=["training"])

# Global trainer instance
trainer: IncrementalTrainer = None


def init_training_router(incremental_trainer: IncrementalTrainer):
    """Initialize training router."""
    global trainer
    trainer = incremental_trainer


@router.post("/interaction", response_model=TrainingResponse)
async def train_from_interaction(request: TrainingRequest):
    """Train from a single interaction.

    Args:
        request: Training request

    Returns:
        Training metrics
    """
    try:
        logger.info(f"Training from interaction: {request.prompt[:50]}...")

        metrics = trainer.learn_from_interaction(
            prompt=request.prompt,
            response=request.response,
            feedback=request.feedback,
            num_steps=request.num_steps,
            learning_rate=request.learning_rate,
        )

        return TrainingResponse(
            status="success",
            avg_loss=metrics["avg_loss"],
            num_steps=request.num_steps,
        )

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_training_stats():
    """Get training statistics.

    Returns:
        Training stats
    """
    try:
        stats = trainer.get_training_stats()
        return {"status": "success", "stats": stats}

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint")
async def save_checkpoint(name: str = None):
    """Save training checkpoint.

    Args:
        name: Optional checkpoint name

    Returns:
        Success message
    """
    try:
        trainer.save_incremental_checkpoint(name)
        return {"status": "success", "message": "Checkpoint saved"}

    except Exception as e:
        logger.error(f"Checkpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
