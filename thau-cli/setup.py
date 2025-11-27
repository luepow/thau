#!/usr/bin/env python3
"""
THAU CLI - Command Line Interface for THAU AI System

Instalación:
    pip install thau-cli

Uso:
    thau chat "Explica este código"
    thau explain main.py
    thau refactor src/app.py
    thau plan "Create authentication system"
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thau-cli",
    version="1.0.0",
    author="Luis Eduardo Perez",
    author_email="luepow@example.com",
    description="THAU AI Command Line Interface - Like Claude Code for THAU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luepow/thau",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "requests>=2.31.0",
        "websockets>=11.0.0",
        "rich>=13.0.0",
        "prompt_toolkit>=3.0.0",
        "pyyaml>=6.0.0",
        "aiohttp>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "thau=thau_cli.cli:main",
        ],
    },
)
