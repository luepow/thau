"""
THAU-Vision: Camera Processor
=============================

Real-time camera processing for live image understanding.

Features:
- Webcam capture and processing
- Frame-by-frame analysis
- Object tracking
- Scene change detection
- Interactive Q&A about live feed
- Recording and snapshot capabilities
"""

import time
import threading
from queue import Queue
from typing import Optional, Dict, List, Callable, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Optional OpenCV import
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


@dataclass
class FrameResult:
    """Result from processing a frame."""
    frame_id: int
    timestamp: float
    image: Optional[Image.Image] = None
    caption: Optional[str] = None
    objects: Optional[List[str]] = None
    response: Optional[str] = None
    metadata: Optional[Dict] = None


class CameraProcessor:
    """
    Real-time camera processing with THAU-Vision.

    Capabilities:
    - Live camera feed processing
    - Periodic frame analysis
    - Object detection updates
    - Scene change detection
    - Interactive questioning
    - Recording support
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        camera_id: int = 0,
        frame_rate: int = 30,
        process_every_n: int = 30,  # Process every Nth frame
        device: Optional[torch.device] = None,
    ):
        """
        Initialize camera processor.

        Args:
            model: Pre-loaded THAUVisionModel
            model_path: Path to load model from
            camera_id: Camera device ID
            frame_rate: Target frame rate
            process_every_n: Process every N frames (for performance)
            device: Device to use
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")

        self.camera_id = camera_id
        self.frame_rate = frame_rate
        self.process_every_n = process_every_n
        self.device = device

        # Load model
        self.model = model
        if model_path is not None and model is None:
            from ..models import THAUVisionModel
            self.model = THAUVisionModel.from_pretrained(model_path, device=device)

        # Camera state
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False

        # Frame processing
        self.frame_count = 0
        self.last_processed_frame: Optional[FrameResult] = None
        self.results_queue: Queue = Queue(maxsize=100)

        # Processing thread
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_frame_processed: Optional[Callable[[FrameResult], None]] = None
        self.on_scene_change: Optional[Callable[[FrameResult], None]] = None

        # Recording
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.snapshots_dir = Path("snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

        # Scene change detection
        self.last_caption = ""
        self.scene_change_threshold = 0.5  # Similarity threshold

    def start(self):
        """Start camera capture and processing."""
        if self.is_running:
            return

        # Open camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)

        self.is_running = True
        self.is_paused = False

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"Camera {self.camera_id} started")

    def stop(self):
        """Stop camera capture."""
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2)

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        print("Camera stopped")

    def pause(self):
        """Pause processing (camera still captures)."""
        self.is_paused = True

    def resume(self):
        """Resume processing."""
        self.is_paused = False

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running:
            if self.camera is None:
                break

            ret, frame = self.camera.read()
            if not ret:
                continue

            self.frame_count += 1

            # Recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            # Process every Nth frame
            if not self.is_paused and self.frame_count % self.process_every_n == 0:
                self._process_frame(frame)

            # Small delay to control frame rate
            time.sleep(1.0 / self.frame_rate)

    def _process_frame(self, frame: np.ndarray):
        """Process a single frame."""
        try:
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Create result
            result = FrameResult(
                frame_id=self.frame_count,
                timestamp=time.time(),
                image=image,
            )

            # Generate caption if model available
            if self.model is not None:
                result.caption = self.model.caption(image, "Que ves en este momento?")

                # Detect scene change
                if self._detect_scene_change(result.caption):
                    if self.on_scene_change:
                        self.on_scene_change(result)

            self.last_processed_frame = result

            # Add to queue
            if not self.results_queue.full():
                self.results_queue.put(result)

            # Callback
            if self.on_frame_processed:
                self.on_frame_processed(result)

        except Exception as e:
            print(f"Error processing frame: {e}")

    def _detect_scene_change(self, caption: str) -> bool:
        """Detect if scene has changed significantly."""
        if not self.last_caption:
            self.last_caption = caption
            return True

        # Simple word overlap comparison
        words1 = set(self.last_caption.lower().split())
        words2 = set(caption.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        is_change = overlap < self.scene_change_threshold

        if is_change:
            self.last_caption = caption

        return is_change

    def get_current_frame(self) -> Optional[Image.Image]:
        """Get the current camera frame."""
        if self.camera is None:
            return None

        ret, frame = self.camera.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def ask(self, question: str) -> str:
        """
        Ask a question about the current camera view.

        Args:
            question: Question to ask

        Returns:
            Answer based on current frame
        """
        frame = self.get_current_frame()
        if frame is None:
            return "Camera not available"

        if self.model is None:
            return "Model not loaded"

        return self.model.answer(frame, question)

    def describe_current(self) -> str:
        """Get description of current view."""
        frame = self.get_current_frame()
        if frame is None:
            return "Camera not available"

        if self.model is None:
            return "Model not loaded"

        return self.model.caption(frame, "Describe detalladamente lo que ves:")

    def identify_current(self) -> List[str]:
        """Identify objects in current view."""
        frame = self.get_current_frame()
        if frame is None:
            return []

        if self.model is None:
            return []

        response = self.model.answer(frame, "Lista los objetos que ves (separados por coma):")
        return [obj.strip() for obj in response.split(",") if obj.strip()]

    def snapshot(self, filename: Optional[str] = None) -> str:
        """
        Take a snapshot of current view.

        Args:
            filename: Optional filename

        Returns:
            Path to saved snapshot
        """
        frame = self.get_current_frame()
        if frame is None:
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"

        path = self.snapshots_dir / filename
        frame.save(path)

        return str(path)

    def start_recording(self, filename: Optional[str] = None):
        """Start recording video."""
        if self.is_recording:
            return

        if self.camera is None:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"

        # Get frame dimensions
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.snapshots_dir / filename),
            fourcc,
            self.frame_rate,
            (width, height),
        )

        self.is_recording = True
        print(f"Started recording: {filename}")

    def stop_recording(self):
        """Stop recording video."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        print("Recording stopped")

    def get_latest_results(self, n: int = 10) -> List[FrameResult]:
        """Get latest processed frame results."""
        results = []
        while not self.results_queue.empty() and len(results) < n:
            results.append(self.results_queue.get_nowait())
        return results

    def set_callbacks(
        self,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        on_scene_change: Optional[Callable[[FrameResult], None]] = None,
    ):
        """Set callback functions."""
        if on_frame:
            self.on_frame_processed = on_frame
        if on_scene_change:
            self.on_scene_change = on_scene_change

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class InteractiveCameraSession:
    """
    Interactive camera session with THAU-Vision.

    Provides a command-line interface for camera interaction.
    """

    def __init__(self, processor: CameraProcessor):
        """
        Initialize interactive session.

        Args:
            processor: CameraProcessor instance
        """
        self.processor = processor
        self.running = False

    def run(self):
        """Run interactive session."""
        print("\n" + "="*50)
        print("THAU-Vision Interactive Camera Session")
        print("="*50)
        print("\nCommands:")
        print("  ask <question> - Ask about current view")
        print("  describe       - Describe current view")
        print("  identify       - Identify objects")
        print("  snapshot       - Take snapshot")
        print("  record         - Start/stop recording")
        print("  pause/resume   - Pause/resume processing")
        print("  quit           - Exit")
        print("-"*50 + "\n")

        self.processor.start()
        self.running = True

        try:
            while self.running:
                try:
                    cmd = input("THAU-Vision> ").strip()
                except EOFError:
                    break

                if not cmd:
                    continue

                self._handle_command(cmd)

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.processor.stop()

    def _handle_command(self, cmd: str):
        """Handle a user command."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "quit" or command == "exit":
            self.running = False

        elif command == "ask":
            if not args:
                print("Usage: ask <question>")
                return
            response = self.processor.ask(args)
            print(f"THAU: {response}")

        elif command == "describe":
            description = self.processor.describe_current()
            print(f"THAU: {description}")

        elif command == "identify":
            objects = self.processor.identify_current()
            print(f"Objects: {', '.join(objects)}")

        elif command == "snapshot":
            path = self.processor.snapshot()
            print(f"Saved: {path}")

        elif command == "record":
            if self.processor.is_recording:
                self.processor.stop_recording()
            else:
                self.processor.start_recording()

        elif command == "pause":
            self.processor.pause()
            print("Processing paused")

        elif command == "resume":
            self.processor.resume()
            print("Processing resumed")

        else:
            print(f"Unknown command: {command}")


# Convenience functions
def create_camera_processor(
    model_path: Optional[str] = None,
    camera_id: int = 0,
    **kwargs,
) -> CameraProcessor:
    """Create a camera processor instance."""
    return CameraProcessor(
        model_path=model_path,
        camera_id=camera_id,
        **kwargs,
    )


def run_interactive_camera(
    model_path: Optional[str] = None,
    camera_id: int = 0,
):
    """Run interactive camera session."""
    processor = create_camera_processor(model_path=model_path, camera_id=camera_id)
    session = InteractiveCameraSession(processor)
    session.run()


# Test
if __name__ == "__main__":
    print("Camera Processor module loaded.")

    if HAS_CV2:
        print("OpenCV available - camera functions enabled")
        print("\nUsage:")
        print("  processor = create_camera_processor()")
        print("  processor.start()")
        print("  processor.ask('What do you see?')")
        print("  processor.stop()")
    else:
        print("OpenCV not available - install with: pip install opencv-python")
