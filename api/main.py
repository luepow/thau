"""FastAPI main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from core.inference.generator import TextGenerator
from core.training.incremental_trainer import IncrementalTrainer
from memory.manager import MemoryManager

from api.routes import chat, training, memory, vision
from api.schemas.models import HealthResponse

# Initialize config
config = get_config()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=config.LOG_LEVEL,
)
logger.add(
    config.LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    level=config.LOG_LEVEL,
)

# Create FastAPI app
app = FastAPI(
    title="my-llm API",
    description="API for the my-llm modular LLM framework",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_adapter: ModelAdapter = None
text_generator: TextGenerator = None
incremental_trainer: IncrementalTrainer = None
memory_manager: MemoryManager = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global model_adapter, text_generator, incremental_trainer, memory_manager

    logger.info("Initializing my-llm API...")

    try:
        # Initialize model adapter
        logger.info("Loading model...")
        model_adapter = ModelAdapter(
            model_name=config.MODEL_NAME,
            use_quantization=config.USE_QUANTIZATION,
        )
        model_adapter.load_model()
        model_adapter.load_tokenizer()

        # Initialize text generator
        text_generator = TextGenerator(model_adapter=model_adapter, config=config)

        # Initialize incremental trainer
        incremental_trainer = IncrementalTrainer(model_adapter=model_adapter, config=config)

        # Initialize memory manager
        memory_manager = MemoryManager(config=config)

        # Initialize routers with dependencies
        chat.init_chat_router(text_generator, memory_manager)
        training.init_training_router(incremental_trainer)
        memory.init_memory_router(memory_manager)

        logger.info("my-llm API initialized successfully!")

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to my-llm API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            model_loaded=model_adapter.model is not None,
            memory_stats=memory_manager.get_stats(),
            device_info=model_adapter.device_manager.get_device_info(),
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            memory_stats={},
            device_info={},
        )


# Include routers
app.include_router(chat.router)
app.include_router(training.router)
app.include_router(memory.router)
app.include_router(vision.router)


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
