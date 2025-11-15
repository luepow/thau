"""Chat endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from typing import AsyncIterator
import uuid

from api.schemas.models import ChatRequest, ChatResponse
from core.inference.generator import TextGenerator, StreamingGenerator
from memory.manager import MemoryManager

router = APIRouter(prefix="/chat", tags=["chat"])

# Global instances (initialized in main.py)
generator: TextGenerator = None
memory_manager: MemoryManager = None


def init_chat_router(text_generator: TextGenerator, mem_manager: MemoryManager):
    """Initialize chat router with dependencies."""
    global generator, memory_manager
    generator = text_generator
    memory_manager = mem_manager


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """Send a message and get a response.

    Args:
        request: Chat request with message and optional history

    Returns:
        Chat response
    """
    try:
        logger.info(f"Chat request: {request.message[:50]}...")

        # Get conversation history from memory if enabled
        context = []
        if request.use_memory:
            context = memory_manager.get_conversation_history()

        # Add current conversation history if provided
        if request.conversation_history:
            context.extend([msg.dict() for msg in request.conversation_history])

        # Generate response
        response = generator.chat(
            message=request.message,
            conversation_history=context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Update memory
        if request.use_memory:
            memory_manager.update_context("user", request.message)
            memory_manager.update_context("assistant", response)

        conversation_id = str(uuid.uuid4())

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response token by token.

    Args:
        request: Chat request

    Returns:
        Streaming response
    """
    async def generate_stream() -> AsyncIterator[str]:
        try:
            # Get context
            context = []
            if request.use_memory:
                context = memory_manager.get_conversation_history()

            if request.conversation_history:
                context.extend([msg.dict() for msg in request.conversation_history])

            # Format prompt
            if hasattr(generator.tokenizer, "apply_chat_template"):
                messages = context + [{"role": "user", "content": request.message}]
                prompt = generator.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
                prompt += f"\nuser: {request.message}\nassistant:"

            # Stream generation
            streaming_gen = StreamingGenerator(generator.model_adapter)

            for token in streaming_gen.generate_stream(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.get("/history")
async def get_history(n: int = 10):
    """Get conversation history.

    Args:
        n: Number of recent messages

    Returns:
        List of recent messages
    """
    try:
        history = memory_manager.get_conversation_history(n)
        return {"history": history, "count": len(history)}

    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_history():
    """Clear conversation history.

    Returns:
        Success message
    """
    try:
        memory_manager.clear_short_term()
        return {"status": "success", "message": "Conversation history cleared"}

    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
