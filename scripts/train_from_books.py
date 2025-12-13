#!/usr/bin/env python3
"""
THAU Training from Books - Extrae conocimiento de PDFs y entrena a THAU

Procesa múltiples PDFs de diferentes categorías:
- Programación (Python, Dart, C#, Go, Rust, etc.)
- Frameworks (Next.js, Django, React Native, Spring)
- DevOps (Git, Docker, Linux, PowerShell)
- Bases de datos (MySQL, PostgreSQL)
- Contabilidad (NIIF)
- Marketing (Kotler)
- Hardware (Arduino)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️  PyMuPDF no instalado. Instalando...")
    os.system(f"{sys.executable} -m pip install pymupdf")
    import fitz
    HAS_PYMUPDF = True

from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("data/datasets/books")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Categorías de libros
BOOK_CATEGORIES = {
    # Programación
    "python": ["Aprende Python", "python"],
    "dart": ["Dart", "dart", "flutter"],
    "csharp": ["C#", "csharp", ".NET"],
    "go": ["GO", "golang"],
    "rust": ["Rust"],
    "algorithm": ["Algorithm", "algoritmo"],

    # Frontend
    "css": ["CSS", "css"],
    "nextjs": ["next.js", "NextJs", "Next"],
    "react": ["React", "react-native"],

    # Backend
    "django": ["Django"],
    "spring": ["Spring"],

    # DevOps
    "git": ["GIT", "git", "GitFlow", "Github", "progit"],
    "docker": ["Docker"],
    "linux": ["Linux", "GNU"],
    "powershell": ["PowerShell"],

    # Databases
    "mysql": ["Mysql", "mysql"],
    "postgresql": ["Postgresql", "postgres"],

    # Contabilidad/Finanzas
    "niif": ["NIIF", "NIFF", "contabilidad"],

    # Marketing
    "marketing": ["Marketing", "Kotler"],

    # Hardware
    "arduino": ["Arduino"],
}


@dataclass
class BookInfo:
    """Información de un libro"""
    path: str
    filename: str
    category: str
    title: str
    pages: int = 0
    extracted_text: str = ""


@dataclass
class TrainingExample:
    """Ejemplo de entrenamiento"""
    instruction: str
    input: str
    output: str
    category: str
    source: str


class PDFExtractor:
    """Extrae texto de PDFs"""

    def __init__(self):
        self.processed_hashes = set()

    def extract_text(self, pdf_path: str) -> Tuple[str, int]:
        """Extrae texto de un PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            pages = len(doc)

            for page_num in range(pages):
                page = doc[page_num]
                text += page.get_text()
                text += "\n\n"

            doc.close()

            # Limpiar texto
            text = self._clean_text(text)

            return text, pages

        except Exception as e:
            logger.error(f"Error extrayendo {pdf_path}: {e}")
            return "", 0

    def _clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        # Remover múltiples espacios
        text = re.sub(r' +', ' ', text)
        # Remover múltiples líneas vacías
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remover caracteres especiales problemáticos
        text = text.replace('\x00', '')
        return text.strip()


class QAGenerator:
    """Genera pares pregunta-respuesta del texto"""

    # Templates por categoría
    TEMPLATES = {
        "python": [
            ("¿Cómo se {action} en Python?", "En Python, para {action} se usa: {content}"),
            ("Explica {concept} en Python", "{content}"),
            ("¿Qué es {concept} en Python?", "{content}"),
            ("Muestra un ejemplo de {concept} en Python", "```python\n{code}\n```"),
        ],
        "dart": [
            ("¿Cómo se {action} en Dart?", "En Dart, {content}"),
            ("Explica {concept} en Dart/Flutter", "{content}"),
            ("¿Cuál es la sintaxis de {concept} en Dart?", "{content}"),
        ],
        "git": [
            ("¿Cómo hago {action} en Git?", "Para {action} en Git: {content}"),
            ("Explica el comando git {command}", "{content}"),
            ("¿Qué es {concept} en Git?", "{content}"),
            ("¿Cuál es la diferencia entre {concept1} y {concept2} en Git?", "{content}"),
        ],
        "nextjs": [
            ("¿Cómo se {action} en Next.js?", "En Next.js, {content}"),
            ("Explica {concept} en Next.js", "{content}"),
            ("¿Cómo funciona {concept} en Next.js 14?", "{content}"),
        ],
        "docker": [
            ("¿Cómo {action} con Docker?", "{content}"),
            ("Explica {concept} en Docker", "{content}"),
            ("¿Qué hace el comando docker {command}?", "{content}"),
        ],
        "niif": [
            ("¿Qué establece la NIIF sobre {concept}?", "Según las NIIF, {content}"),
            ("Explica {concept} según las normas NIIF", "{content}"),
            ("¿Cómo se contabiliza {concept} según NIIF?", "{content}"),
        ],
        "default": [
            ("Explica {concept}", "{content}"),
            ("¿Qué es {concept}?", "{content}"),
            ("¿Cómo funciona {concept}?", "{content}"),
        ]
    }

    def __init__(self):
        self.examples: List[TrainingExample] = []

    def extract_sections(self, text: str) -> List[Dict]:
        """Extrae secciones del texto"""
        sections = []

        # Buscar títulos de sección (líneas cortas seguidas de contenido)
        lines = text.split('\n')
        current_section = {"title": "", "content": ""}

        for i, line in enumerate(lines):
            line = line.strip()

            # Detectar título de sección
            if len(line) < 100 and line and not line.endswith('.'):
                # Posible título
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": line, "content": ""}
            else:
                current_section["content"] += line + " "

        if current_section["content"]:
            sections.append(current_section)

        return sections

    def extract_code_blocks(self, text: str) -> List[Dict]:
        """Extrae bloques de código"""
        code_blocks = []

        # Patrones para detectar código
        patterns = [
            # Código con comentarios explicativos
            (r'(#.*?\n)?((?:def |class |import |from |if |for |while |return ).*?)(?:\n{2,}|\Z)',
             'python'),
            # Código JavaScript/TypeScript
            (r'(//.*?\n)?((?:function |const |let |var |import |export |class ).*?)(?:\n{2,}|\Z)',
             'javascript'),
            # Comandos de terminal
            (r'\$\s*(.+?)(?:\n|$)', 'bash'),
        ]

        for pattern, lang in patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                code = match.group(0).strip()
                if len(code) > 20:
                    code_blocks.append({"code": code, "lang": lang})

        return code_blocks

    def generate_qa_from_section(self, section: Dict, category: str, source: str) -> List[TrainingExample]:
        """Genera QA desde una sección"""
        examples = []
        title = section["title"]
        content = section["content"].strip()

        if len(content) < 50:
            return examples

        # Truncar contenido muy largo
        if len(content) > 2000:
            content = content[:2000] + "..."

        templates = self.TEMPLATES.get(category, self.TEMPLATES["default"])

        # Generar ejemplo básico
        if title:
            examples.append(TrainingExample(
                instruction=f"Explica {title}",
                input="",
                output=content,
                category=category,
                source=source
            ))

            # Generar variaciones
            examples.append(TrainingExample(
                instruction=f"¿Qué es {title}?",
                input="",
                output=content,
                category=category,
                source=source
            ))

        return examples

    def generate_qa_from_code(self, code_block: Dict, category: str, source: str) -> List[TrainingExample]:
        """Genera QA desde código"""
        examples = []
        code = code_block["code"]
        lang = code_block["lang"]

        if len(code) < 30:
            return examples

        # Detectar qué hace el código
        if "def " in code:
            # Es una función
            func_match = re.search(r'def\s+(\w+)', code)
            if func_match:
                func_name = func_match.group(1)
                examples.append(TrainingExample(
                    instruction=f"Escribe una función {func_name} en {lang}",
                    input="",
                    output=f"```{lang}\n{code}\n```",
                    category=category,
                    source=source
                ))

        elif "class " in code:
            # Es una clase
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                examples.append(TrainingExample(
                    instruction=f"Crea una clase {class_name} en {lang}",
                    input="",
                    output=f"```{lang}\n{code}\n```",
                    category=category,
                    source=source
                ))

        return examples


class BookProcessor:
    """Procesa libros y genera datasets"""

    def __init__(self):
        self.extractor = PDFExtractor()
        self.qa_gen = QAGenerator()
        self.books: List[BookInfo] = []
        self.all_examples: List[TrainingExample] = []

    def categorize_book(self, filename: str) -> str:
        """Determina la categoría de un libro"""
        filename_lower = filename.lower()

        for category, keywords in BOOK_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in filename_lower:
                    return category

        return "general"

    def process_book(self, pdf_path: str) -> Optional[BookInfo]:
        """Procesa un libro individual"""
        path = Path(pdf_path)

        if not path.exists():
            logger.warning(f"Archivo no encontrado: {pdf_path}")
            return None

        filename = path.name
        category = self.categorize_book(filename)

        logger.info(f"📖 Procesando: {filename} (categoría: {category})")

        # Extraer texto
        text, pages = self.extractor.extract_text(str(path))

        if not text:
            logger.warning(f"No se pudo extraer texto de {filename}")
            return None

        book = BookInfo(
            path=str(path),
            filename=filename,
            category=category,
            title=filename.replace('.pdf', ''),
            pages=pages,
            extracted_text=text
        )

        self.books.append(book)

        # Generar ejemplos de entrenamiento
        self._generate_training_examples(book)

        logger.info(f"   ✅ {pages} páginas, {len(text)} caracteres")

        return book

    def _generate_training_examples(self, book: BookInfo):
        """Genera ejemplos de entrenamiento desde un libro"""
        text = book.extracted_text
        category = book.category

        # Extraer secciones
        sections = self.qa_gen.extract_sections(text)

        for section in sections[:100]:  # Limitar a 100 secciones por libro
            examples = self.qa_gen.generate_qa_from_section(
                section, category, book.filename
            )
            self.all_examples.extend(examples)

        # Extraer código
        code_blocks = self.qa_gen.extract_code_blocks(text)

        for block in code_blocks[:50]:  # Limitar a 50 bloques de código
            examples = self.qa_gen.generate_qa_from_code(
                block, category, book.filename
            )
            self.all_examples.extend(examples)

        logger.info(f"   📝 {len(self.all_examples)} ejemplos totales generados")

    def process_all_books(self, pdf_paths: List[str]):
        """Procesa todos los libros"""
        logger.info(f"🚀 Procesando {len(pdf_paths)} libros...")

        for path in pdf_paths:
            self.process_book(path)

        logger.info(f"\n📊 Resumen:")
        logger.info(f"   Libros procesados: {len(self.books)}")
        logger.info(f"   Ejemplos generados: {len(self.all_examples)}")

        # Estadísticas por categoría
        categories = {}
        for ex in self.all_examples:
            categories[ex.category] = categories.get(ex.category, 0) + 1

        logger.info(f"\n   Por categoría:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            logger.info(f"     {cat}: {count}")

    def save_datasets(self):
        """Guarda los datasets generados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Dataset completo
        all_file = OUTPUT_DIR / f"books_training_{timestamp}.jsonl"
        with open(all_file, 'w', encoding='utf-8') as f:
            for ex in self.all_examples:
                entry = {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "category": ex.category,
                    "source": ex.source
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"\n💾 Dataset guardado: {all_file}")

        # Datasets por categoría
        by_category = {}
        for ex in self.all_examples:
            if ex.category not in by_category:
                by_category[ex.category] = []
            by_category[ex.category].append(ex)

        for category, examples in by_category.items():
            cat_file = OUTPUT_DIR / f"{category}_training_{timestamp}.jsonl"
            with open(cat_file, 'w', encoding='utf-8') as f:
                for ex in examples:
                    entry = {
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            logger.info(f"   {category}: {len(examples)} ejemplos -> {cat_file}")

        # Chat format para Ollama
        chat_file = OUTPUT_DIR / f"books_chat_format_{timestamp}.jsonl"
        with open(chat_file, 'w', encoding='utf-8') as f:
            for ex in self.all_examples:
                entry = {
                    "messages": [
                        {"role": "user", "content": ex.instruction},
                        {"role": "assistant", "content": ex.output}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"\n📦 Chat format: {chat_file}")

        return str(all_file)


def main():
    """Función principal"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              THAU Training from Books                                 ║
║                                                                      ║
║  Extrae conocimiento de PDFs para entrenar a THAU                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Lista de PDFs a procesar
    pdf_paths = [
        # Git
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/progit.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/GitFlow_vs_TrunkBased_Presentation.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje GIT.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Github.pdf',

        # Next.js
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/next.js_v14_documentation.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/NextJs.pdf',

        # Arduino
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Arduino Books.pdf',

        # Marketing
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Fundamentos del Marketing-Kotler.pdf',

        # NIIF/Contabilidad
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/NIIF-2019-Completas.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/2008guiarapidaspainNIFF.pdf',

        # Programación
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Apprendizaje Dart.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprende Python.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Algorithm.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje C#.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje CSS.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Dart.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Django.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Docker.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje GNU Linux.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje GO.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/aprendizaje LINUX.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Mysql.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje NET.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Postgresql.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje PowerShell.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje React-native.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Aprendizaje Rust.pdf',
        '/Users/lperez/Library/CloudStorage/Dropbox/Books/Apress.Expert.Spring.MVC.and.Web.Flow.Feb.2006.pdf',
    ]

    # Filtrar PDFs que existen
    existing_pdfs = [p for p in pdf_paths if Path(p).exists()]

    logger.info(f"📚 {len(existing_pdfs)}/{len(pdf_paths)} PDFs encontrados")

    if not existing_pdfs:
        logger.error("No se encontraron PDFs para procesar")
        return

    # Procesar
    processor = BookProcessor()
    processor.process_all_books(existing_pdfs)

    # Guardar
    dataset_file = processor.save_datasets()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ Procesamiento completado                                          ║
║                                                                      ║
║  Dataset principal: {dataset_file}
║                                                                      ║
║  Para entrenar THAU:                                                 ║
║  python scripts/train_from_dataset.py {dataset_file}
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
