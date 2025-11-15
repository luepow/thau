"""
THAU Model Export to GGUF Format
Exports TinyLlama model to GGUF format for Ollama and LMStudio
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import get_config


class GGUFExporter:
    """Exports HuggingFace models to GGUF format for Ollama/LMStudio"""

    def __init__(self, output_dir: str = "./export/models"):
        """
        Initialize GGUF Exporter

        Args:
            output_dir: Directory to save exported models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llama_cpp_dir = Path("./export/llama.cpp")
        self.config = get_config()

        logger.info(f"🚀 GGUF Exporter initialized")
        logger.info(f"   Output directory: {self.output_dir}")

    def check_requirements(self) -> bool:
        """Check if required tools are available"""
        logger.info("🔍 Checking requirements...")

        # Check Python packages
        try:
            import transformers
            logger.info("   ✅ transformers installed")
        except ImportError:
            logger.error("   ❌ transformers not installed. Run: pip install transformers")
            return False

        # Check if git is available
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            logger.info("   ✅ git available")
        except:
            logger.error("   ❌ git not available")
            return False

        return True

    def clone_llama_cpp(self) -> bool:
        """Clone llama.cpp repository if not present"""
        if self.llama_cpp_dir.exists():
            logger.info(f"✅ llama.cpp already cloned at {self.llama_cpp_dir}")
            return True

        logger.info("📥 Cloning llama.cpp repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(self.llama_cpp_dir)],
                check=True
            )
            logger.info("✅ llama.cpp cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to clone llama.cpp: {e}")
            return False

    def install_llama_cpp_requirements(self) -> bool:
        """Install llama.cpp Python requirements"""
        logger.info("📦 Installing llama.cpp requirements...")

        requirements_file = self.llama_cpp_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("⚠️  llama.cpp requirements.txt not found")
            return True

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            logger.info("✅ llama.cpp requirements installed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install requirements: {e}")
            return False

    def download_model(self, model_name: Optional[str] = None) -> Path:
        """
        Download model from HuggingFace

        Args:
            model_name: HuggingFace model identifier (default from config)

        Returns:
            Path to downloaded model
        """
        if model_name is None:
            model_name = self.config.MODEL_NAME

        logger.info(f"📥 Downloading model: {model_name}")

        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = self.output_dir / "hf_model" / model_name.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("   Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)

            logger.info("   Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            model.save_pretrained(model_path)

            logger.info(f"✅ Model downloaded to: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise

    def convert_to_gguf(self, model_path: Path, output_name: str = "thau") -> Path:
        """
        Convert HuggingFace model to GGUF format

        Args:
            model_path: Path to HuggingFace model
            output_name: Name for output GGUF file

        Returns:
            Path to GGUF file
        """
        logger.info(f"🔄 Converting to GGUF format...")

        convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            # Try alternative script names
            convert_script = self.llama_cpp_dir / "convert-hf-to-gguf.py"
            if not convert_script.exists():
                logger.error(f"❌ Conversion script not found in {self.llama_cpp_dir}")
                raise FileNotFoundError("convert_hf_to_gguf.py not found")

        output_file = self.output_dir / f"{output_name}.gguf"

        try:
            cmd = [
                sys.executable,
                str(convert_script),
                str(model_path),
                "--outfile", str(output_file),
                "--outtype", "f16"  # Float16 precision
            ]

            logger.info(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            logger.info(result.stdout)
            logger.info(f"✅ GGUF model created: {output_file}")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Conversion failed: {e}")
            logger.error(f"   stdout: {e.stdout}")
            logger.error(f"   stderr: {e.stderr}")
            raise

    def quantize_model(self, gguf_path: Path, quantization: str = "Q4_K_M") -> Path:
        """
        Quantize GGUF model

        Args:
            gguf_path: Path to GGUF model
            quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)

        Returns:
            Path to quantized model
        """
        logger.info(f"🗜️  Quantizing model to {quantization}...")

        # Build quantize binary if needed
        quantize_bin = self.llama_cpp_dir / "quantize"
        if not quantize_bin.exists():
            logger.info("   Building llama.cpp quantize tool...")
            try:
                subprocess.run(
                    ["make", "quantize"],
                    cwd=self.llama_cpp_dir,
                    check=True
                )
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to build quantize tool")
                logger.info("   You may need to install build tools (make, gcc, etc.)")
                raise

        output_file = gguf_path.parent / f"{gguf_path.stem}-{quantization}.gguf"

        try:
            cmd = [
                str(quantize_bin),
                str(gguf_path),
                str(output_file),
                quantization
            ]

            logger.info(f"   Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            logger.info(f"✅ Quantized model created: {output_file}")

            # Show file sizes
            original_size = gguf_path.stat().st_size / (1024**3)
            quantized_size = output_file.stat().st_size / (1024**3)

            logger.info(f"   Original size: {original_size:.2f} GB")
            logger.info(f"   Quantized size: {quantized_size:.2f} GB")
            logger.info(f"   Compression ratio: {(1 - quantized_size/original_size)*100:.1f}%")

            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Quantization failed: {e}")
            raise

    def export_full_pipeline(
        self,
        model_name: Optional[str] = None,
        quantize: bool = True,
        quantization_type: str = "Q4_K_M"
    ) -> dict:
        """
        Run full export pipeline

        Args:
            model_name: HuggingFace model name
            quantize: Whether to quantize the model
            quantization_type: Type of quantization

        Returns:
            Dict with paths to exported models
        """
        logger.info("="*60)
        logger.info("🚀 Starting THAU Model Export Pipeline")
        logger.info("="*60)

        # Step 1: Check requirements
        if not self.check_requirements():
            raise RuntimeError("Requirements check failed")

        # Step 2: Clone llama.cpp
        if not self.clone_llama_cpp():
            raise RuntimeError("Failed to clone llama.cpp")

        # Step 3: Install llama.cpp requirements
        if not self.install_llama_cpp_requirements():
            logger.warning("⚠️  Some requirements may be missing")

        # Step 4: Download model
        model_path = self.download_model(model_name)

        # Step 5: Convert to GGUF
        gguf_path = self.convert_to_gguf(model_path, output_name="thau-f16")

        result = {
            "hf_model": str(model_path),
            "gguf_f16": str(gguf_path)
        }

        # Step 6: Quantize (optional)
        if quantize:
            quantized_path = self.quantize_model(gguf_path, quantization_type)
            result["gguf_quantized"] = str(quantized_path)
            result["quantization_type"] = quantization_type

        logger.info("="*60)
        logger.info("✅ Export pipeline completed successfully!")
        logger.info("="*60)
        logger.info(f"📁 Exported models:")
        for key, path in result.items():
            logger.info(f"   {key}: {path}")

        return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Export THAU model to GGUF format")

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name (default from config)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./export/models",
        help="Output directory for exported models"
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization step"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q8_0"],
        help="Quantization type"
    )

    args = parser.parse_args()

    try:
        exporter = GGUFExporter(output_dir=args.output_dir)

        result = exporter.export_full_pipeline(
            model_name=args.model,
            quantize=not args.no_quantize,
            quantization_type=args.quantization
        )

        print("\n" + "="*60)
        print("🎉 SUCCESS! Your THAU model is ready for Ollama/LMStudio")
        print("="*60)
        print("\nNext steps:")
        print("1. Import to Ollama:")
        print(f"   ollama create thau -f export/Modelfile")
        print("\n2. Import to LMStudio:")
        print(f"   Open LMStudio and load: {result.get('gguf_quantized', result['gguf_f16'])}")
        print("\n3. Test the model:")
        print("   python export/test_exported_model.py")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
