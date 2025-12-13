"""PDF text extraction utility for training data generation."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")


class PDFExtractor:
    """Extract and process text from PDF files for training."""

    def __init__(self, output_dir: str = "./data/extracted"):
        """Initialize PDF extractor.

        Args:
            output_dir: Directory to save extracted text
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting text from: {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        text_content = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- Page {page_num} ---\n{text}")

        doc.close()

        full_text = "\n\n".join(text_content)
        logger.info(f"Extracted {len(full_text):,} characters from {len(text_content)} pages")

        return full_text

    def extract_chapters(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract text organized by chapters/sections.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chapter dictionaries with 'title' and 'content'
        """
        full_text = self.extract_text(pdf_path)

        # Common chapter patterns
        chapter_patterns = [
            r'(?:Chapter|CHAPTER|Capítulo|CAPÍTULO)\s+\d+[:\.\s]+([^\n]+)',
            r'(?:Part|PART|Parte|PARTE)\s+\d+[:\.\s]+([^\n]+)',
            r'\n(\d+\.\s+[A-Z][^\n]+)\n',  # Numbered sections
        ]

        chapters = []
        current_chapter = {"title": "Introduction", "content": ""}

        lines = full_text.split('\n')

        for line in lines:
            is_chapter = False
            for pattern in chapter_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous chapter
                    if current_chapter["content"].strip():
                        chapters.append(current_chapter)
                    # Start new chapter
                    current_chapter = {
                        "title": line.strip(),
                        "content": ""
                    }
                    is_chapter = True
                    break

            if not is_chapter:
                current_chapter["content"] += line + "\n"

        # Don't forget last chapter
        if current_chapter["content"].strip():
            chapters.append(current_chapter)

        logger.info(f"Extracted {len(chapters)} chapters/sections")
        return chapters

    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from text.

        Args:
            text: Text content to search

        Returns:
            List of code blocks with context
        """
        code_blocks = []

        # Pattern for code blocks (indented or with markers)
        patterns = [
            # Triple backticks
            r'```(?:python)?\n(.*?)```',
            # >>> Python REPL
            r'(>>>.*?(?:\n(?:>>>|\.\.\.).*?)*)',
            # Indented code (4+ spaces)
            r'\n((?:    .*\n)+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(match.strip()) > 10:  # Filter tiny snippets
                    code_blocks.append({
                        "code": match.strip(),
                        "pattern": pattern[:20]
                    })

        logger.info(f"Found {len(code_blocks)} code blocks")
        return code_blocks

    def clean_text(self, text: str) -> str:
        """Clean extracted text for training.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)

        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove page numbers
        text = re.sub(r'\n\d+\n', '\n', text)

        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def save_extracted(self, pdf_path: str, content: str, suffix: str = "") -> Path:
        """Save extracted content to file.

        Args:
            pdf_path: Original PDF path (for naming)
            content: Content to save
            suffix: Optional suffix for filename

        Returns:
            Path to saved file
        """
        pdf_name = Path(pdf_path).stem
        output_name = f"{pdf_name}{suffix}.txt"
        output_path = self.output_dir / output_name

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Saved to: {output_path}")
        return output_path

    def process_pdf(self, pdf_path: str) -> Dict:
        """Full processing pipeline for a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted data
        """
        # Extract full text
        raw_text = self.extract_text(pdf_path)

        # Clean text
        clean_text = self.clean_text(raw_text)

        # Extract chapters
        chapters = self.extract_chapters(pdf_path)

        # Extract code blocks
        code_blocks = self.extract_code_blocks(clean_text)

        # Save outputs
        self.save_extracted(pdf_path, clean_text, "_clean")

        # Save chapters as JSON
        chapters_path = self.output_dir / f"{Path(pdf_path).stem}_chapters.json"
        with open(chapters_path, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, indent=2, ensure_ascii=False)

        return {
            "pdf_path": str(pdf_path),
            "total_chars": len(clean_text),
            "num_chapters": len(chapters),
            "num_code_blocks": len(code_blocks),
            "output_dir": str(self.output_dir),
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract text from PDF for training")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", default="./data/extracted", help="Output directory")
    args = parser.parse_args()

    extractor = PDFExtractor(output_dir=args.output_dir)
    result = extractor.process_pdf(args.pdf_path)

    print("\n=== Extraction Complete ===")
    print(f"Characters extracted: {result['total_chars']:,}")
    print(f"Chapters found: {result['num_chapters']}")
    print(f"Code blocks found: {result['num_code_blocks']}")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
