"""Pydantic models for API schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    """Chat message schema."""
    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    use_memory: bool = Field(default=True, description="Whether to use memory for context")


class ChatResponse(BaseModel):
    """Chat response schema."""
    response: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    """Training request schema."""
    prompt: str = Field(..., description="Training prompt")
    response: str = Field(..., description="Expected response")
    feedback: Optional[str] = Field(default=None, description="Optional feedback")
    num_steps: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=5e-5, gt=0.0)


class TrainingResponse(BaseModel):
    """Training response schema."""
    status: str
    avg_loss: float
    num_steps: int


class MemoryRequest(BaseModel):
    """Memory storage request."""
    content: str = Field(..., description="Content to remember")
    memory_type: str = Field(default="fact", description="Type of memory")
    importance: int = Field(default=5, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    """Memory storage response."""
    memory_id: str
    status: str


class RecallRequest(BaseModel):
    """Memory recall request."""
    query: str = Field(..., description="Search query")
    n_results: int = Field(default=5, ge=1, le=50)


class RecallResponse(BaseModel):
    """Memory recall response."""
    results: List[Dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    memory_stats: Dict[str, Any]
    device_info: Dict[str, Any]


class GenerateRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., description="Generation prompt")
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Stream response")


class GenerateResponse(BaseModel):
    """Text generation response."""
    generated_text: str
    prompt: str
