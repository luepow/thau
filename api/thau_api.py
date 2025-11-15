"""
API REST para THAU
Permite agregar datos de entrenamiento y controlar el servicio
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer import get_service, get_config, DataManager

# Crear app
app = FastAPI(
    title="THAU API",
    description="API para entrenar el modelo THAU de forma autónoma",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
service = get_service()
config = get_config()
data_manager = DataManager()


# Modelos Pydantic
class TrainingExample(BaseModel):
    instruction: str
    output: str
    metadata: Optional[Dict] = None


class TrainingBatch(BaseModel):
    examples: List[TrainingExample]


class ServiceStatus(BaseModel):
    status: str
    model_identifier: str
    version: int
    is_running: bool
    training_in_progress: bool
    pending_examples: int
    total_trainings: int
    last_training: Optional[str]


# Endpoints
@app.get("/")
def root():
    """Info básica"""
    return {
        "service": "THAU API",
        "version": "1.0.0",
        "model": config.get_model_identifier(),
        "status": "running" if service.is_running else "stopped"
    }


@app.get("/status", response_model=ServiceStatus)
def get_status():
    """Estado del servicio"""
    stats = service.get_stats()

    return ServiceStatus(
        status="running" if service.is_running else "stopped",
        model_identifier=config.get_model_identifier(),
        version=config.current_version,
        is_running=service.is_running,
        training_in_progress=stats.get("training_in_progress", False),
        pending_examples=stats.get("new_examples_pending", 0),
        total_trainings=stats.get("total_trainings", 0),
        last_training=stats.get("last_training")
    )


@app.post("/training/add")
def add_training_example(example: TrainingExample):
    """Agrega un ejemplo de entrenamiento"""

    result_hash = data_manager.add_example(
        instruction=example.instruction,
        output=example.output,
        metadata=example.metadata
    )

    if result_hash == "duplicate":
        raise HTTPException(status_code=409, detail="Example already exists")

    return {
        "success": True,
        "hash": result_hash,
        "message": "Example added to training queue"
    }


@app.post("/training/batch")
def add_training_batch(batch: TrainingBatch):
    """Agrega múltiples ejemplos de entrenamiento"""

    examples_dict = [ex.dict() for ex in batch.examples]
    results = data_manager.add_batch(examples_dict)

    return {
        "success": True,
        "results": results,
        "message": f"Added {results['added']} new examples"
    }


@app.post("/training/force")
def force_training(background_tasks: BackgroundTasks):
    """Fuerza un entrenamiento inmediato"""

    if service.stats["training_in_progress"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Ejecutar en background
    background_tasks.add_task(service.force_train)

    return {
        "success": True,
        "message": "Training started in background"
    }


@app.get("/stats")
def get_stats():
    """Estadísticas completas"""

    return {
        "service": service.get_stats(),
        "data": data_manager.get_stats(),
        "config": {
            "model_identifier": config.get_model_identifier(),
            "auto_train_enabled": config.auto_train_enabled,
            "auto_train_interval_hours": config.auto_train_interval_hours,
            "min_new_examples": config.min_new_examples,
            "model_size": config.model_size
        }
    }


@app.post("/service/start")
def start_service():
    """Inicia el servicio de entrenamiento"""

    if service.is_running:
        raise HTTPException(status_code=409, detail="Service already running")

    service.start()

    return {
        "success": True,
        "message": "Training service started"
    }


@app.post("/service/stop")
def stop_service():
    """Detiene el servicio de entrenamiento"""

    if not service.is_running:
        raise HTTPException(status_code=409, detail="Service not running")

    service.stop()

    return {
        "success": True,
        "message": "Training service stopped"
    }


@app.get("/examples/pending")
def get_pending_examples():
    """Lista ejemplos pendientes de entrenamiento"""

    examples = data_manager.get_new_examples()

    return {
        "count": len(examples),
        "examples": [
            {
                "hash": ex.get("hash"),
                "instruction_preview": ex.get("instruction", "")[:100],
                "added_at": ex.get("added_at")
            }
            for ex in examples
        ]
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Al iniciar la API, iniciar el servicio de entrenamiento"""
    if config.auto_train_enabled and not service.is_running:
        service.start()


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Al cerrar la API, detener el servicio"""
    if service.is_running:
        service.stop()


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🤖 THAU API - Sistema de Entrenamiento Autónomo")
    print("=" * 80)
    print(f"Modelo: {config.get_model_identifier()}")
    print(f"API: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=8000)
