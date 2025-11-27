"""
Sistema de Aprendizaje desde Fuentes Externas para THAU
Permite aprender desde:
- Páginas web (documentación, tutoriales, artículos)
- Archivos PDF (libros, papers, manuales)
- Búsquedas en internet
"""

import json
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import requests

# Intentar importar dependencias opcionales
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class ExternalLearningSystem:
    """
    Sistema para que THAU aprenda de fuentes externas:
    - Web scraping de documentación
    - Lectura de PDFs
    - Búsquedas en internet
    """

    def __init__(
        self,
        data_dir: Path = Path("./data/external_learning"),
        qa_output_dir: Path = Path("./data/self_questioning"),
    ):
        self.data_dir = data_dir
        self.qa_output_dir = qa_output_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qa_output_dir.mkdir(parents=True, exist_ok=True)

        # Log de fuentes procesadas
        self.sources_log = self.data_dir / "sources_log.jsonl"

        # Headers para requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) THAU-LearningBot/1.0"
        }

    # ==================== WEB LEARNING ====================

    def learn_from_url(
        self,
        url: str,
        topic: str = "general",
        generate_qa: bool = True,
        use_model: str = "gemini"
    ) -> Dict:
        """
        Aprende de una página web

        Args:
            url: URL de la página
            topic: Tema/categoría del contenido
            generate_qa: Si generar pares Q&A del contenido
            use_model: Modelo para generar Q&A (gemini, coder, etc.)

        Returns:
            Dict con contenido extraído y Q&A generados
        """
        print(f"\n🌐 Aprendiendo de: {url}")

        # Extraer contenido
        content = self._extract_web_content(url)
        if not content:
            return {"success": False, "error": "No se pudo extraer contenido"}

        print(f"   📄 Extraídos {len(content)} caracteres")

        result = {
            "success": True,
            "url": url,
            "topic": topic,
            "content_length": len(content),
            "timestamp": datetime.now().isoformat(),
            "qa_generated": 0
        }

        # Guardar contenido raw
        self._save_content(url, content, topic)

        # Generar Q&A si se solicita
        if generate_qa:
            qa_pairs = self._generate_qa_from_content(content, topic, use_model)
            result["qa_generated"] = len(qa_pairs)
            print(f"   💡 Generados {len(qa_pairs)} pares Q&A")

        # Log
        self._log_source(url, "web", result)

        return result

    def learn_from_documentation(
        self,
        base_url: str,
        pages: List[str],
        topic: str,
        use_model: str = "gemini",
        delay: float = 2.0
    ) -> Dict:
        """
        Aprende de múltiples páginas de documentación

        Args:
            base_url: URL base (ej: "https://docs.python.org/3/")
            pages: Lista de paths relativos
            topic: Tema/categoría
            use_model: Modelo para Q&A
            delay: Delay entre requests

        Returns:
            Estadísticas del aprendizaje
        """
        print(f"\n📚 Aprendiendo documentación de: {base_url}")
        print(f"   Páginas: {len(pages)}")

        stats = {
            "total_pages": len(pages),
            "successful": 0,
            "failed": 0,
            "total_qa": 0,
            "errors": []
        }

        for i, page in enumerate(pages):
            url = f"{base_url.rstrip('/')}/{page.lstrip('/')}"
            print(f"\n   [{i+1}/{len(pages)}] {page}")

            try:
                result = self.learn_from_url(url, topic, generate_qa=True, use_model=use_model)
                if result["success"]:
                    stats["successful"] += 1
                    stats["total_qa"] += result.get("qa_generated", 0)
                else:
                    stats["failed"] += 1
                    stats["errors"].append({"url": url, "error": result.get("error")})
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"url": url, "error": str(e)})

            time.sleep(delay)

        print(f"\n✅ Documentación procesada:")
        print(f"   Exitosas: {stats['successful']}/{stats['total_pages']}")
        print(f"   Q&A total: {stats['total_qa']}")

        return stats

    def search_and_learn(
        self,
        query: str,
        num_results: int = 5,
        topic: str = "general",
        use_model: str = "gemini"
    ) -> Dict:
        """
        Busca en internet y aprende de los resultados

        Args:
            query: Consulta de búsqueda
            num_results: Número de resultados a procesar
            topic: Tema/categoría
            use_model: Modelo para Q&A

        Returns:
            Estadísticas del aprendizaje
        """
        print(f"\n🔍 Buscando: '{query}'")

        # Usar Gemini para buscar y obtener info
        search_prompt = f"""Busca información actualizada sobre: {query}

Proporciona un resumen técnico detallado con:
1. Conceptos clave
2. Ejemplos prácticos de código si aplica
3. Mejores prácticas
4. Recursos recomendados

Responde en español de forma técnica y educativa."""

        # Ejecutar búsqueda con Gemini
        try:
            result = subprocess.run(
                ["gemini", "-p", search_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and result.stdout.strip():
                content = result.stdout.strip()
                print(f"   📄 Obtenidos {len(content)} caracteres")

                # Generar Q&A del contenido
                qa_pairs = self._generate_qa_from_content(content, topic, use_model)

                return {
                    "success": True,
                    "query": query,
                    "content_length": len(content),
                    "qa_generated": len(qa_pairs)
                }
        except Exception as e:
            print(f"   ❌ Error: {e}")

        return {"success": False, "query": query, "error": "Búsqueda fallida"}

    # ==================== PDF LEARNING ====================

    def learn_from_pdf(
        self,
        pdf_path: str,
        topic: str = "general",
        generate_qa: bool = True,
        use_model: str = "gemini",
        pages_per_batch: int = 10
    ) -> Dict:
        """
        Aprende de un archivo PDF

        Args:
            pdf_path: Ruta al archivo PDF
            topic: Tema/categoría
            generate_qa: Si generar Q&A
            use_model: Modelo para Q&A
            pages_per_batch: Páginas por batch para procesar

        Returns:
            Estadísticas del aprendizaje
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"success": False, "error": f"Archivo no encontrado: {pdf_path}"}

        print(f"\n📖 Aprendiendo de PDF: {pdf_path.name}")

        # Extraer texto del PDF
        text, num_pages = self._extract_pdf_text(pdf_path)
        if not text:
            return {"success": False, "error": "No se pudo extraer texto del PDF"}

        print(f"   📄 Extraídas {num_pages} páginas, {len(text)} caracteres")

        result = {
            "success": True,
            "file": str(pdf_path),
            "topic": topic,
            "pages": num_pages,
            "content_length": len(text),
            "timestamp": datetime.now().isoformat(),
            "qa_generated": 0
        }

        # Guardar contenido
        self._save_content(str(pdf_path), text, topic, source_type="pdf")

        # Generar Q&A en batches
        if generate_qa:
            # Dividir texto en chunks manejables
            chunks = self._split_text(text, max_chars=4000)
            total_qa = 0

            for i, chunk in enumerate(chunks):
                print(f"   Procesando chunk {i+1}/{len(chunks)}...")
                qa_pairs = self._generate_qa_from_content(chunk, topic, use_model)
                total_qa += len(qa_pairs)
                time.sleep(1)  # Rate limiting

            result["qa_generated"] = total_qa
            print(f"   💡 Generados {total_qa} pares Q&A total")

        # Log
        self._log_source(str(pdf_path), "pdf", result)

        return result

    def learn_from_pdf_directory(
        self,
        directory: str,
        topic: str = "general",
        use_model: str = "gemini",
        recursive: bool = True
    ) -> Dict:
        """
        Aprende de todos los PDFs en un directorio

        Args:
            directory: Directorio con PDFs
            topic: Tema/categoría
            use_model: Modelo para Q&A
            recursive: Buscar recursivamente

        Returns:
            Estadísticas totales
        """
        directory = Path(directory)
        if not directory.exists():
            return {"success": False, "error": f"Directorio no encontrado: {directory}"}

        # Encontrar PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdfs = list(directory.glob(pattern))

        print(f"\n📚 Encontrados {len(pdfs)} PDFs en {directory}")

        stats = {
            "total_files": len(pdfs),
            "successful": 0,
            "failed": 0,
            "total_qa": 0,
            "total_pages": 0,
            "errors": []
        }

        for i, pdf in enumerate(pdfs):
            print(f"\n[{i+1}/{len(pdfs)}] {pdf.name}")

            try:
                result = self.learn_from_pdf(str(pdf), topic, use_model=use_model)
                if result["success"]:
                    stats["successful"] += 1
                    stats["total_qa"] += result.get("qa_generated", 0)
                    stats["total_pages"] += result.get("pages", 0)
                else:
                    stats["failed"] += 1
                    stats["errors"].append({"file": str(pdf), "error": result.get("error")})
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"file": str(pdf), "error": str(e)})

        print(f"\n✅ PDFs procesados:")
        print(f"   Exitosos: {stats['successful']}/{stats['total_files']}")
        print(f"   Páginas total: {stats['total_pages']}")
        print(f"   Q&A generados: {stats['total_qa']}")

        return stats

    # ==================== RESEARCH MODE ====================

    def research_topic(
        self,
        topic: str,
        depth: str = "medium",
        use_model: str = "gemini"
    ) -> Dict:
        """
        Investiga un tema en profundidad usando múltiples fuentes

        Args:
            topic: Tema a investigar
            depth: Profundidad (quick, medium, deep)
            use_model: Modelo para Q&A

        Returns:
            Resultados de la investigación
        """
        print(f"\n🔬 Investigando: '{topic}' (profundidad: {depth})")

        # Configurar número de consultas según profundidad
        num_queries = {"quick": 3, "medium": 6, "deep": 10}.get(depth, 5)

        # Generar consultas relacionadas
        queries = self._generate_research_queries(topic, num_queries, use_model)

        stats = {
            "topic": topic,
            "depth": depth,
            "queries_executed": 0,
            "total_qa": 0,
            "subtopics": []
        }

        for query in queries:
            print(f"\n   🔍 Consultando: {query}")
            result = self.search_and_learn(query, topic=topic, use_model=use_model)

            if result.get("success"):
                stats["queries_executed"] += 1
                stats["total_qa"] += result.get("qa_generated", 0)
                stats["subtopics"].append(query)

            time.sleep(2)  # Rate limiting

        print(f"\n✅ Investigación completada:")
        print(f"   Consultas: {stats['queries_executed']}")
        print(f"   Q&A generados: {stats['total_qa']}")

        return stats

    # ==================== HELPER METHODS ====================

    def _extract_web_content(self, url: str) -> Optional[str]:
        """Extrae contenido de texto limpio de una URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            if HAS_BS4:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remover scripts, styles, etc.
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()

                # Obtener texto
                text = soup.get_text(separator='\n', strip=True)

                # Limpiar líneas vacías múltiples
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return '\n'.join(lines)
            else:
                # Fallback básico sin BeautifulSoup
                import html
                text = re.sub(r'<[^>]+>', ' ', response.text)
                text = html.unescape(text)
                return ' '.join(text.split())

        except Exception as e:
            print(f"   ❌ Error extrayendo: {e}")
            return None

    def _extract_pdf_text(self, pdf_path: Path) -> Tuple[Optional[str], int]:
        """Extrae texto de un PDF"""
        try:
            if HAS_PDFPLUMBER:
                # Usar pdfplumber (mejor para PDFs complejos)
                text_parts = []
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return '\n\n'.join(text_parts), num_pages

            elif HAS_PYPDF2:
                # Usar PyPDF2 (más básico)
                text_parts = []
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    num_pages = len(reader.pages)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return '\n\n'.join(text_parts), num_pages

            else:
                print("   ⚠️ Instala pdfplumber o PyPDF2: pip install pdfplumber")
                return None, 0

        except Exception as e:
            print(f"   ❌ Error leyendo PDF: {e}")
            return None, 0

    def _generate_qa_from_content(
        self,
        content: str,
        topic: str,
        use_model: str
    ) -> List[Dict]:
        """Genera pares Q&A a partir de contenido"""
        # Limitar contenido para el prompt
        content_truncated = content[:3000] if len(content) > 3000 else content

        prompt = f"""Basándote en el siguiente contenido sobre {topic}, genera 5 pares de pregunta-respuesta educativos.

CONTENIDO:
{content_truncated}

INSTRUCCIONES:
- Genera preguntas técnicas y específicas
- Las respuestas deben ser completas pero concisas
- Incluye ejemplos de código cuando sea relevante
- Responde en español

FORMATO (JSON):
[
  {{"question": "¿Pregunta 1?", "answer": "Respuesta 1"}},
  {{"question": "¿Pregunta 2?", "answer": "Respuesta 2"}}
]

Genera solo el JSON, sin texto adicional:"""

        qa_pairs = []

        try:
            if use_model == "gemini":
                result = subprocess.run(
                    ["gemini", "-p", prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    response = result.stdout.strip()
                    # Intentar parsear JSON
                    qa_pairs = self._parse_qa_json(response)

            elif use_model == "coder":
                # Usar Ollama coder
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5-coder:1.5b-base",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 1000}
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    text = response.json().get("response", "")
                    qa_pairs = self._parse_qa_json(text)

        except Exception as e:
            print(f"   ⚠️ Error generando Q&A: {e}")

        # Guardar Q&A generados
        if qa_pairs:
            self._save_qa_pairs(qa_pairs, topic)

        return qa_pairs

    def _parse_qa_json(self, text: str) -> List[Dict]:
        """Intenta parsear JSON de Q&A desde texto"""
        # Buscar JSON array en el texto
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    return [
                        item for item in data
                        if isinstance(item, dict) and "question" in item and "answer" in item
                    ]
            except json.JSONDecodeError:
                pass

        # Fallback: crear Q&A del texto directamente
        return []

    def _generate_research_queries(
        self,
        topic: str,
        num_queries: int,
        use_model: str
    ) -> List[str]:
        """Genera consultas de investigación relacionadas"""
        prompt = f"""Genera {num_queries} consultas de búsqueda técnicas para investigar a fondo sobre: {topic}

Las consultas deben cubrir:
- Conceptos básicos
- Uso avanzado
- Mejores prácticas
- Casos de uso
- Comparaciones con alternativas

Responde solo con las consultas, una por línea:"""

        queries = [topic]  # Siempre incluir el tema principal

        try:
            result = subprocess.run(
                ["gemini", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                queries.extend([
                    line.strip().lstrip('0123456789.-) ')
                    for line in lines
                    if line.strip() and len(line.strip()) > 5
                ][:num_queries])
        except:
            pass

        return queries[:num_queries]

    def _split_text(self, text: str, max_chars: int = 4000) -> List[str]:
        """Divide texto en chunks manejables"""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _save_content(
        self,
        source: str,
        content: str,
        topic: str,
        source_type: str = "web"
    ):
        """Guarda contenido extraído"""
        # Crear nombre de archivo seguro
        safe_name = re.sub(r'[^\w\-]', '_', source)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_type}_{topic}_{safe_name}_{timestamp}.txt"

        filepath = self.data_dir / "raw_content" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SOURCE: {source}\n")
            f.write(f"TOPIC: {topic}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write(content)

    def _save_qa_pairs(self, qa_pairs: List[Dict], topic: str):
        """Guarda pares Q&A al archivo de self-questioning"""
        today = datetime.now().strftime("%Y%m%d")
        qa_file = self.qa_output_dir / f"qa_{today}.jsonl"

        with open(qa_file, 'a', encoding='utf-8') as f:
            for qa in qa_pairs:
                entry = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": topic,
                    "source": "external_learning",
                    "timestamp": datetime.now().isoformat(),
                    "cognitive_age": 12
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _log_source(self, source: str, source_type: str, result: Dict):
        """Registra fuente procesada"""
        entry = {
            "source": source,
            "type": source_type,
            "timestamp": datetime.now().isoformat(),
            **result
        }

        with open(self.sources_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ==================== CLI INTERFACE ====================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="THAU External Learning - Aprende de Web y PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Aprender de una URL
  python external_learning.py --url https://docs.python.org/3/tutorial/

  # Aprender de un PDF
  python external_learning.py --pdf /path/to/book.pdf --topic python

  # Aprender de todos los PDFs en un directorio
  python external_learning.py --pdf-dir ./books/ --topic programming

  # Investigar un tema
  python external_learning.py --research "machine learning" --depth deep

  # Buscar y aprender
  python external_learning.py --search "Python asyncio tutorial"
        """
    )

    parser.add_argument("--url", help="URL para aprender")
    parser.add_argument("--pdf", help="Ruta a PDF para aprender")
    parser.add_argument("--pdf-dir", help="Directorio con PDFs")
    parser.add_argument("--search", help="Buscar y aprender de internet")
    parser.add_argument("--research", help="Investigar tema en profundidad")
    parser.add_argument("--topic", default="general", help="Tema/categoría")
    parser.add_argument("--depth", choices=["quick", "medium", "deep"], default="medium")
    parser.add_argument("--model", default="gemini", help="Modelo para Q&A")

    args = parser.parse_args()

    system = ExternalLearningSystem()

    if args.url:
        result = system.learn_from_url(args.url, args.topic, use_model=args.model)
        print(f"\nResultado: {result}")

    elif args.pdf:
        result = system.learn_from_pdf(args.pdf, args.topic, use_model=args.model)
        print(f"\nResultado: {result}")

    elif args.pdf_dir:
        result = system.learn_from_pdf_directory(args.pdf_dir, args.topic, use_model=args.model)
        print(f"\nResultado: {result}")

    elif args.search:
        result = system.search_and_learn(args.search, topic=args.topic, use_model=args.model)
        print(f"\nResultado: {result}")

    elif args.research:
        result = system.research_topic(args.research, depth=args.depth, use_model=args.model)
        print(f"\nResultado: {result}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
