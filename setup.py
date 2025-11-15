"""Setup configuration for my-llm project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="my-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular framework for developing LLMs from scratch with incremental learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my-llm",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "data"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "chromadb>=0.4.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "sentence-transformers>=2.2.0",
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "my-llm=api.main:run_server",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
