"""
THAU-Vision: API Server
=======================

FastAPI server for THAU-Vision capabilities.

Endpoints:
- POST /caption: Generate image caption
- POST /answer: Answer question about image
- POST /identify: Identify objects in image
- POST /describe: Get detailed scene description
- POST /compare: Compare two images
- POST /extract_text: OCR text extraction
- POST /learn: Learn to recognize an object
- GET /health: Health check
"""

import io
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from PIL import Image


# Request/Response Models
class CaptionRequest(BaseModel):
    """Request for image captioning."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    style: str = Field("detailed", description="Caption style: brief, detailed, poetic")
    language: str = Field("es", description="Output language: es, en")


class AnswerRequest(BaseModel):
    """Request for visual Q&A."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    question: str = Field(..., description="Question about the image")


class IdentifyRequest(BaseModel):
    """Request for object identification."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    detail_level: str = Field("normal", description="Detail level: basic, normal, detailed")


class CompareRequest(BaseModel):
    """Request for image comparison."""
    image1_base64: Optional[str] = None
    image2_base64: Optional[str] = None
    image1_url: Optional[str] = None
    image2_url: Optional[str] = None
    aspect: str = Field("general", description="Comparison aspect")


class LearnRequest(BaseModel):
    """Request for learning a new object."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    object_name: str = Field(..., description="Name of the object")
    description: Optional[str] = Field(None, description="Optional description")


class ClassifyRequest(BaseModel):
    """Request for image classification."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    categories: List[str] = Field(..., description="List of categories")


class VisionResponse(BaseModel):
    """Standard response model."""
    success: bool = True
    result: Any = None
    error: Optional[str] = None


class VisionAPI:
    """
    THAU-Vision API wrapper.

    Provides REST endpoints for all vision capabilities.
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """
        Initialize API.

        Args:
            model: Pre-loaded THAUVisionModel
            model_path: Path to load model from
            host: Server host
            port: Server port
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")

        self.host = host
        self.port = port

        # Create FastAPI app
        self.app = FastAPI(
            title="THAU-Vision API",
            description="Vision-Language Model API for image understanding",
            version="1.0.0",
        )

        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Load model
        self.model = model
        self.model_path = model_path

        # Create ImageQA helper
        self.image_qa = None

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.on_event("startup")
        async def startup():
            """Load model on startup."""
            from ..inference import ImageQA

            if self.model is not None:
                self.image_qa = ImageQA(model=self.model)
            elif self.model_path is not None:
                self.image_qa = ImageQA(model_path=self.model_path)
            else:
                print("Warning: No model loaded. Loading default...")
                try:
                    from ..models import THAUVisionModel
                    model = THAUVisionModel(config_name="thau-vision-tiny")
                    self.image_qa = ImageQA(model=model)
                except Exception as e:
                    print(f"Could not load model: {e}")

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "THAU-Vision API",
                "version": "1.0.0",
                "status": "running",
            }

        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                "status": "healthy",
                "model_loaded": self.image_qa is not None,
            }

        @self.app.post("/caption", response_model=VisionResponse)
        async def caption(request: CaptionRequest):
            """Generate image caption."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.caption(
                    image,
                    style=request.style,
                    language=request.language,
                )
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/caption/upload", response_model=VisionResponse)
        async def caption_upload(
            file: UploadFile = File(...),
            style: str = Form("detailed"),
            language: str = Form("es"),
        ):
            """Caption uploaded image."""
            try:
                image = Image.open(io.BytesIO(await file.read())).convert("RGB")
                result = self.image_qa.caption(image, style=style, language=language)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/answer", response_model=VisionResponse)
        async def answer(request: AnswerRequest):
            """Answer question about image."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.answer(image, request.question)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/answer/upload", response_model=VisionResponse)
        async def answer_upload(
            file: UploadFile = File(...),
            question: str = Form(...),
        ):
            """Answer question about uploaded image."""
            try:
                image = Image.open(io.BytesIO(await file.read())).convert("RGB")
                result = self.image_qa.answer(image, question)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/identify", response_model=VisionResponse)
        async def identify(request: IdentifyRequest):
            """Identify objects in image."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.identify_objects(image, detail_level=request.detail_level)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/describe", response_model=VisionResponse)
        async def describe(request: CaptionRequest):
            """Get detailed scene description."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.describe_scene(image)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/compare", response_model=VisionResponse)
        async def compare(request: CompareRequest):
            """Compare two images."""
            try:
                image1 = self._get_image(request.image1_base64, request.image1_url)
                image2 = self._get_image(request.image2_base64, request.image2_url)
                result = self.image_qa.compare_images(image1, image2, aspect=request.aspect)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/extract_text", response_model=VisionResponse)
        async def extract_text(request: CaptionRequest):
            """Extract text from image (OCR)."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.extract_text(image)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/classify", response_model=VisionResponse)
        async def classify(request: ClassifyRequest):
            """Classify image into categories."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.classify(image, request.categories)
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

        @self.app.post("/learn", response_model=VisionResponse)
        async def learn(request: LearnRequest):
            """Learn to recognize a new object."""
            try:
                image = self._get_image(request.image_base64, request.image_url)
                result = self.image_qa.learn_object(
                    image,
                    request.object_name,
                    request.description,
                )
                return VisionResponse(result=result)
            except Exception as e:
                return VisionResponse(success=False, error=str(e))

    def _get_image(
        self,
        base64_data: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Image.Image:
        """Get image from base64 or URL."""
        if base64_data:
            # Decode base64
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            img_bytes = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        elif url:
            # Download from URL
            import urllib.request
            with urllib.request.urlopen(url) as response:
                img_bytes = response.read()
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        else:
            raise ValueError("Either image_base64 or image_url must be provided")

    def run(self):
        """Run the API server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)


# Convenience functions
def create_vision_api(
    model=None,
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> VisionAPI:
    """Create and return a VisionAPI instance."""
    return VisionAPI(model=model, model_path=model_path, host=host, port=port)


def run_vision_server(
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Run the vision API server."""
    api = create_vision_api(model_path=model_path, host=host, port=port)
    api.run()


# Test
if __name__ == "__main__":
    print("THAU-Vision API Server")
    print("="*40)

    if HAS_FASTAPI:
        print("FastAPI available")
        print("\nTo start server:")
        print("  python -m thau_vision.api.server")
        print("\nEndpoints:")
        print("  POST /caption - Generate caption")
        print("  POST /answer - Visual Q&A")
        print("  POST /identify - Object detection")
        print("  POST /describe - Scene description")
        print("  POST /compare - Compare images")
        print("  POST /extract_text - OCR")
        print("  POST /classify - Classification")
        print("  POST /learn - Learn object")
    else:
        print("FastAPI not available")
        print("Install with: pip install fastapi uvicorn")
