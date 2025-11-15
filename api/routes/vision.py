"""
Vision API Routes for THAU
Handles image generation and vision-related requests
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path

from capabilities.vision.image_generator import ThauImageGenerator
from capabilities.tools.tool_registry import get_tool_registry


# Router
router = APIRouter(prefix="/vision", tags=["vision"])

# Global generator instance (lazy loaded)
_image_generator = None


def get_image_generator() -> ThauImageGenerator:
    """Get or create image generator instance"""
    global _image_generator
    if _image_generator is None:
        _image_generator = ThauImageGenerator()
    return _image_generator


# Request/Response Models
class ImageGenerationRequest(BaseModel):
    """Request for image generation"""
    prompt: str = Field(..., description="Text description of the image to generate")
    negative_prompt: Optional[str] = Field(
        "blurry, bad quality, distorted",
        description="What to avoid in the image"
    )
    num_inference_steps: int = Field(30, ge=10, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    width: int = Field(512, ge=256, le=1024, description="Image width")
    height: int = Field(512, ge=256, le=1024, description="Image height")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ImageGenerationResponse(BaseModel):
    """Response from image generation"""
    success: bool
    image_path: Optional[str]
    image_url: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]


class ConversationRequest(BaseModel):
    """Request for conversation with tool detection"""
    message: str = Field(..., description="User's message")
    auto_generate_image: bool = Field(True, description="Auto-generate images if requested")


class ConversationResponse(BaseModel):
    """Response from conversation"""
    response: str
    tool_used: Optional[str]
    image_generated: bool = False
    image_path: Optional[str]
    image_url: Optional[str]


# Endpoints
@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image from a text prompt

    Example:
        POST /vision/generate
        {
            "prompt": "a cute robot learning to code",
            "num_inference_steps": 30,
            "width": 512,
            "height": 512
        }
    """
    try:
        generator = get_image_generator()

        result = generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )

        # Generate URL for image
        image_url = None
        if result['success'] and result['path']:
            # Relative URL
            image_url = f"/vision/image/{result['path'].name}"

        return ImageGenerationResponse(
            success=result['success'],
            image_path=str(result['path']) if result['path'] else None,
            image_url=image_url,
            metadata=result.get('metadata'),
            error=result.get('error')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/{filename}")
async def get_image(filename: str):
    """
    Serve a generated image

    Example:
        GET /vision/image/20250114_120000_cute_robot.png
    """
    generator = get_image_generator()
    image_path = generator.output_dir / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/png")


@router.post("/chat", response_model=ConversationResponse)
async def chat_with_vision(request: ConversationRequest):
    """
    Chat with THAU with automatic image generation detection

    Example:
        POST /vision/chat
        {
            "message": "Genera una imagen de un gato espacial",
            "auto_generate_image": true
        }
    """
    try:
        # Detect if image generation is needed
        registry = get_tool_registry()
        tool = registry.detect_tool_needed(request.message)

        # Response defaults
        response_text = "Procesando tu solicitud..."
        tool_used = None
        image_generated = False
        image_path = None
        image_url = None

        if tool and tool.name == "generate_image" and request.auto_generate_image:
            # Extract parameters
            params = registry.extract_parameters(request.message, tool)

            # Generate image
            generator = get_image_generator()
            result = generator.generate_image(**params)

            if result['success']:
                image_generated = True
                image_path = str(result['path'])
                image_url = f"/vision/image/{result['path'].name}"

                response_text = (
                    f"¡Listo! He generado la imagen. "
                    f"Puedes verla en: {image_url}"
                )
            else:
                response_text = f"Lo siento, hubo un error generando la imagen: {result.get('error')}"

            tool_used = tool.name

        else:
            # Normal conversation (would use LLM here)
            response_text = "Entendido. ¿En qué más puedo ayudarte?"

        return ConversationResponse(
            response=response_text,
            tool_used=tool_used,
            image_generated=image_generated,
            image_path=image_path,
            image_url=image_url
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get vision statistics

    Example:
        GET /vision/stats
    """
    try:
        generator = get_image_generator()
        stats = generator.get_stats()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples")
async def get_examples():
    """
    Get example prompts for image generation

    Example:
        GET /vision/examples
    """
    return {
        "examples": [
            "a cute robot learning to code, digital art",
            "a futuristic cityscape at sunset, cyberpunk style",
            "an AI brain made of circuits, glowing, artistic",
            "a serene mountain landscape with aurora borealis",
            "abstract representation of recursion, colorful, modern art",
            "a library filled with floating holographic books",
        ],
        "tips": [
            "Be specific about the style (digital art, photorealistic, oil painting, etc.)",
            "Include details about lighting, colors, and mood",
            "Mention what to avoid in the negative_prompt",
            "Higher guidance_scale (7-15) follows the prompt more closely",
            "More inference_steps (30-50) produces higher quality",
        ]
    }
