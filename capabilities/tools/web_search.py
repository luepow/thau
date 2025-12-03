"""
THAU Web Search & Research Tools

Herramientas para búsqueda web e investigación autónoma:
- WebSearchTool: Búsqueda en DuckDuckGo (sin API key)
- WebFetcher: Extracción de contenido de páginas web
- ResearchAgent: Investigación autónoma multi-fuente

Características:
- Sin dependencia de APIs de pago
- Extracción inteligente de contenido
- Cache de resultados
- Rate limiting para evitar bloqueos
- Resumen automático de información
"""

import re
import json
import time
import hashlib
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False


class SearchEngine(Enum):
    """Motores de búsqueda soportados"""
    DUCKDUCKGO = "duckduckgo"
    DUCKDUCKGO_LITE = "duckduckgo_lite"


@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    title: str
    url: str
    snippet: str
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WebPage:
    """Contenido de página web"""
    url: str
    title: str
    content: str
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    meta_description: str = ""
    fetch_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:5000],  # Limitar
            "links_count": len(self.links),
            "success": self.success,
            "error": self.error
        }


@dataclass
class ResearchResult:
    """Resultado de investigación"""
    query: str
    summary: str
    sources: List[SearchResult]
    key_facts: List[str]
    related_topics: List[str]
    confidence: float
    research_time: float

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": [s.to_dict() for s in self.sources],
            "key_facts": self.key_facts,
            "related_topics": self.related_topics,
            "confidence": self.confidence,
            "research_time": self.research_time
        }


class SearchCache:
    """Cache simple para resultados de búsqueda"""

    def __init__(self, cache_dir: str = "./data/cache/search"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=24)  # Cache válido por 24 horas

    def _get_cache_key(self, query: str) -> str:
        """Genera clave de cache"""
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[List[Dict]]:
        """Obtiene resultado del cache"""
        key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data["timestamp"])

            if datetime.now() - cached_time > self.ttl:
                cache_file.unlink()
                return None

            return data["results"]
        except Exception:
            return None

    def set(self, query: str, results: List[Dict]) -> None:
        """Guarda resultado en cache"""
        key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"

        data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


class WebSearchTool:
    """
    Herramienta de búsqueda web usando DuckDuckGo

    Características:
    - Sin API key requerida
    - Rate limiting automático
    - Cache de resultados
    - Parsing robusto de HTML
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        rate_limit: float = 1.0,  # segundos entre requests
        timeout: float = 10.0
    ):
        self.cache = SearchCache() if cache_enabled else None
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.last_request = 0.0

        # Headers para parecer un navegador real
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def _wait_rate_limit(self) -> None:
        """Espera para respetar rate limit"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def search(
        self,
        query: str,
        num_results: int = 5,
        region: str = "es-es"
    ) -> List[SearchResult]:
        """
        Busca en DuckDuckGo

        Args:
            query: Consulta de búsqueda
            num_results: Número de resultados deseados
            region: Región para resultados (es-es, en-us, etc.)

        Returns:
            Lista de SearchResult
        """
        # Verificar cache
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                return [SearchResult(**r) for r in cached[:num_results]]

        if not HTTPX_AVAILABLE:
            return self._search_fallback(query, num_results)

        results = []

        try:
            # Intentar DuckDuckGo HTML
            results = self._search_duckduckgo_html(query, num_results, region)
        except Exception as e:
            print(f"[WebSearch] Error en búsqueda: {e}")
            # Fallback
            results = self._search_fallback(query, num_results)

        # Guardar en cache
        if self.cache and results:
            self.cache.set(query, [r.to_dict() for r in results])

        return results

    def _search_duckduckgo_html(
        self,
        query: str,
        num_results: int,
        region: str
    ) -> List[SearchResult]:
        """Búsqueda usando DuckDuckGo HTML"""
        self._wait_rate_limit()

        # URL de DuckDuckGo HTML
        url = "https://html.duckduckgo.com/html/"
        params = {
            "q": query,
            "kl": region,
            "df": ""  # Cualquier fecha
        }

        results = []

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.post(url, data=params, headers=self.headers)

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            html = response.text

            if BS4_AVAILABLE:
                results = self._parse_ddg_html_bs4(html, num_results)
            else:
                results = self._parse_ddg_html_regex(html, num_results)

        return results

    def _parse_ddg_html_bs4(self, html: str, num_results: int) -> List[SearchResult]:
        """Parsea HTML de DuckDuckGo con BeautifulSoup"""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        # Buscar resultados
        for result_div in soup.select(".result"):
            if len(results) >= num_results:
                break

            try:
                # Título y URL
                title_elem = result_div.select_one(".result__title a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")

                # Extraer URL real del redirect de DDG
                if "uddg=" in url:
                    url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])

                # Snippet
                snippet_elem = result_div.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                if title and url:
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo"
                    ))

            except Exception:
                continue

        return results

    def _parse_ddg_html_regex(self, html: str, num_results: int) -> List[SearchResult]:
        """Parsea HTML de DuckDuckGo con regex (fallback)"""
        results = []

        # Patrón para extraer resultados
        pattern = r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]*)</a>'

        matches = re.findall(pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(matches[:num_results]):
            # Extraer URL real
            if "uddg=" in url:
                url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])

            snippet = snippets[i] if i < len(snippets) else ""

            results.append(SearchResult(
                title=title.strip(),
                url=url,
                snippet=snippet.strip(),
                source="duckduckgo"
            ))

        return results

    def _search_fallback(self, query: str, num_results: int) -> List[SearchResult]:
        """Fallback cuando no hay httpx disponible"""
        # Usar urllib estándar
        import urllib.request

        self._wait_rate_limit()

        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                html = response.read().decode("utf-8")

            if BS4_AVAILABLE:
                return self._parse_ddg_html_bs4(html, num_results)
            else:
                return self._parse_ddg_html_regex(html, num_results)

        except Exception as e:
            print(f"[WebSearch] Fallback error: {e}")
            return []


class WebFetcher:
    """
    Extractor de contenido de páginas web

    Características:
    - Extracción de texto limpio
    - Conversión HTML a Markdown
    - Extracción de metadatos
    - Manejo de diferentes encodings
    """

    def __init__(
        self,
        timeout: float = 15.0,
        max_content_length: int = 100000,  # 100KB máximo
        rate_limit: float = 0.5
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.rate_limit = rate_limit
        self.last_request = 0.0

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        }

        # Configurar html2text si disponible
        if HTML2TEXT_AVAILABLE:
            self.h2t = html2text.HTML2Text()
            self.h2t.ignore_links = False
            self.h2t.ignore_images = True
            self.h2t.body_width = 0
            self.h2t.ignore_emphasis = True
        else:
            self.h2t = None

    def _wait_rate_limit(self) -> None:
        """Espera para respetar rate limit"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def fetch(self, url: str) -> WebPage:
        """
        Obtiene y procesa contenido de una URL

        Args:
            url: URL a obtener

        Returns:
            WebPage con contenido procesado
        """
        start_time = time.time()
        self._wait_rate_limit()

        try:
            html = self._fetch_html(url)
            title, content, links, meta = self._extract_content(html, url)

            return WebPage(
                url=url,
                title=title,
                content=content,
                links=links[:20],  # Limitar links
                meta_description=meta,
                fetch_time=time.time() - start_time,
                success=True
            )

        except Exception as e:
            return WebPage(
                url=url,
                title="",
                content="",
                fetch_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _fetch_html(self, url: str) -> str:
        """Obtiene HTML de URL"""
        if HTTPX_AVAILABLE:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()

                # Verificar tamaño
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_content_length:
                    raise ValueError(f"Content too large: {content_length} bytes")

                return response.text
        else:
            import urllib.request
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="ignore")

    def _extract_content(self, html: str, base_url: str) -> Tuple[str, str, List[str], str]:
        """Extrae contenido útil del HTML"""
        if BS4_AVAILABLE:
            return self._extract_with_bs4(html, base_url)
        else:
            return self._extract_with_regex(html)

    def _extract_with_bs4(self, html: str, base_url: str) -> Tuple[str, str, List[str], str]:
        """Extrae contenido usando BeautifulSoup"""
        soup = BeautifulSoup(html, "html.parser")

        # Eliminar elementos no deseados
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()

        # Título
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Meta description
        meta = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta = meta_tag.get("content", "")

        # Contenido principal
        # Intentar encontrar el contenido principal
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"class": re.compile(r"content|main|article|post", re.I)}) or
            soup.find("body")
        )

        if main_content:
            if self.h2t:
                content = self.h2t.handle(str(main_content))
            else:
                content = main_content.get_text(separator="\n", strip=True)
        else:
            content = soup.get_text(separator="\n", strip=True)

        # Limpiar contenido
        content = self._clean_text(content)

        # Extraer links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = urllib.parse.urljoin(base_url, href)
            if href.startswith("http"):
                links.append(href)

        return title, content, links, meta

    def _extract_with_regex(self, html: str) -> Tuple[str, str, List[str], str]:
        """Extrae contenido usando regex (fallback)"""
        # Título
        title_match = re.search(r"<title>([^<]*)</title>", html, re.I)
        title = title_match.group(1) if title_match else ""

        # Meta description
        meta_match = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', html, re.I)
        meta = meta_match.group(1) if meta_match else ""

        # Remover scripts y styles
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.I)

        # Remover tags HTML
        content = re.sub(r"<[^>]+>", " ", html)

        # Limpiar
        content = self._clean_text(content)

        # Links
        links = re.findall(r'href="(https?://[^"]+)"', html)

        return title, content, list(set(links)), meta

    def _clean_text(self, text: str) -> str:
        """Limpia texto extraído"""
        # Reemplazar múltiples espacios/newlines
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remover caracteres especiales
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # Limitar longitud
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "..."

        return text.strip()


class ResearchAgent:
    """
    Agente de Investigación Autónoma

    Realiza investigación multi-fuente:
    1. Busca información sobre el tema
    2. Extrae contenido de las mejores fuentes
    3. Sintetiza y resume la información
    4. Identifica hechos clave y temas relacionados
    """

    def __init__(
        self,
        search_tool: WebSearchTool = None,
        fetcher: WebFetcher = None,
        max_sources: int = 3,
        verbose: bool = True
    ):
        self.search_tool = search_tool or WebSearchTool()
        self.fetcher = fetcher or WebFetcher()
        self.max_sources = max_sources
        self.verbose = verbose

    def research(
        self,
        query: str,
        depth: str = "normal"  # "quick", "normal", "deep"
    ) -> ResearchResult:
        """
        Realiza investigación sobre un tema

        Args:
            query: Tema a investigar
            depth: Profundidad de investigación

        Returns:
            ResearchResult con información recopilada
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n[Research] Investigando: {query}")
            print(f"[Research] Profundidad: {depth}")

        # Configurar según profundidad
        num_results = {"quick": 3, "normal": 5, "deep": 8}.get(depth, 5)
        num_sources = {"quick": 1, "normal": 3, "deep": 5}.get(depth, 3)

        # 1. Buscar información
        if self.verbose:
            print("[Research] Buscando fuentes...")

        search_results = self.search_tool.search(query, num_results=num_results)

        if self.verbose:
            print(f"[Research] Encontradas {len(search_results)} fuentes")

        if not search_results:
            return ResearchResult(
                query=query,
                summary="No se encontraron resultados para la búsqueda.",
                sources=[],
                key_facts=[],
                related_topics=[],
                confidence=0.0,
                research_time=time.time() - start_time
            )

        # 2. Extraer contenido de las mejores fuentes
        if self.verbose:
            print(f"[Research] Extrayendo contenido de {num_sources} fuentes...")

        pages = []
        for result in search_results[:num_sources]:
            if self.verbose:
                print(f"   - {result.url[:60]}...")

            page = self.fetcher.fetch(result.url)
            if page.success:
                pages.append(page)

        # 3. Analizar y sintetizar información
        if self.verbose:
            print("[Research] Analizando información...")

        summary, key_facts, related = self._synthesize(query, search_results, pages)

        # 4. Calcular confianza
        confidence = self._calculate_confidence(search_results, pages)

        research_time = time.time() - start_time

        if self.verbose:
            print(f"[Research] Completado en {research_time:.2f}s")
            print(f"[Research] Confianza: {confidence:.0%}")

        return ResearchResult(
            query=query,
            summary=summary,
            sources=search_results[:num_sources],
            key_facts=key_facts,
            related_topics=related,
            confidence=confidence,
            research_time=research_time
        )

    def _synthesize(
        self,
        query: str,
        results: List[SearchResult],
        pages: List[WebPage]
    ) -> Tuple[str, List[str], List[str]]:
        """Sintetiza información de múltiples fuentes"""
        # Combinar snippets
        all_text = []
        for result in results:
            if result.snippet:
                all_text.append(result.snippet)

        # Agregar contenido de páginas (primeros párrafos)
        for page in pages:
            if page.content:
                # Tomar primeros 500 caracteres
                all_text.append(page.content[:500])

        combined = "\n\n".join(all_text)

        # Generar resumen básico
        # (En producción, esto usaría el modelo LLM)
        summary = self._generate_basic_summary(query, combined, results)

        # Extraer hechos clave
        key_facts = self._extract_key_facts(combined)

        # Identificar temas relacionados
        related = self._find_related_topics(combined, query)

        return summary, key_facts, related

    def _generate_basic_summary(
        self,
        query: str,
        content: str,
        results: List[SearchResult]
    ) -> str:
        """Genera resumen básico sin LLM"""
        # Usar snippets como resumen
        if results:
            summary_parts = [f"Sobre '{query}':\n"]

            for i, result in enumerate(results[:3], 1):
                if result.snippet:
                    summary_parts.append(f"{i}. {result.snippet}")

            summary_parts.append(f"\nFuentes consultadas: {len(results)}")

            return "\n".join(summary_parts)

        return f"No se encontró información suficiente sobre '{query}'."

    def _extract_key_facts(self, text: str) -> List[str]:
        """Extrae hechos clave del texto"""
        facts = []

        # Buscar oraciones que parecen hechos
        sentences = re.split(r'[.!?]', text)

        fact_indicators = [
            r'\b\d{4}\b',  # Años
            r'\b\d+%\b',   # Porcentajes
            r'\bes\b',     # "es" definitivo
            r'\bson\b',    # "son" definitivo
            r'\bfue\b',    # "fue" histórico
            r'\bcreó\b',   # Creación
            r'\binventó\b', # Invención
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 200:
                continue

            # Verificar si parece un hecho
            for pattern in fact_indicators:
                if re.search(pattern, sentence, re.I):
                    facts.append(sentence)
                    break

            if len(facts) >= 5:
                break

        return facts

    def _find_related_topics(self, text: str, query: str) -> List[str]:
        """Encuentra temas relacionados"""
        # Extraer palabras capitalizadas que no están en la query
        query_words = set(query.lower().split())

        # Buscar términos importantes
        terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Filtrar y contar
        term_counts = {}
        for term in terms:
            if term.lower() not in query_words and len(term) > 3:
                term_counts[term] = term_counts.get(term, 0) + 1

        # Ordenar por frecuencia
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)

        return [term for term, count in sorted_terms[:5] if count > 1]

    def _calculate_confidence(
        self,
        results: List[SearchResult],
        pages: List[WebPage]
    ) -> float:
        """Calcula confianza en los resultados"""
        if not results:
            return 0.0

        confidence = 0.5  # Base

        # Más resultados = más confianza
        confidence += min(len(results) * 0.05, 0.2)

        # Páginas exitosamente obtenidas
        successful_pages = sum(1 for p in pages if p.success)
        if pages:
            confidence += (successful_pages / len(pages)) * 0.2

        # Snippets con contenido
        snippets_with_content = sum(1 for r in results if r.snippet and len(r.snippet) > 50)
        if results:
            confidence += (snippets_with_content / len(results)) * 0.1

        return min(confidence, 1.0)


# Funciones de utilidad para integración con THAU

def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Función simple para búsqueda web

    Returns:
        Dict con resultados formateados
    """
    tool = WebSearchTool()
    results = tool.search(query, num_results)

    return {
        "success": len(results) > 0,
        "query": query,
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


def fetch_url(url: str) -> Dict[str, Any]:
    """
    Función simple para obtener contenido de URL

    Returns:
        Dict con contenido de la página
    """
    fetcher = WebFetcher()
    page = fetcher.fetch(url)

    return {
        "success": page.success,
        "url": url,
        "title": page.title,
        "content": page.content[:3000],  # Limitar
        "error": page.error
    }


def research_topic(query: str, depth: str = "normal") -> Dict[str, Any]:
    """
    Función simple para investigar un tema

    Returns:
        Dict con resultados de investigación
    """
    agent = ResearchAgent(verbose=False)
    result = agent.research(query, depth)

    return result.to_dict()


if __name__ == "__main__":
    print("=" * 60)
    print("  THAU Web Search & Research Tools - Demo")
    print("=" * 60)

    # Verificar dependencias
    print("\n[Status] Dependencias:")
    print(f"   httpx: {'✓' if HTTPX_AVAILABLE else '✗ (usando urllib)'}")
    print(f"   beautifulsoup4: {'✓' if BS4_AVAILABLE else '✗ (usando regex)'}")
    print(f"   html2text: {'✓' if HTML2TEXT_AVAILABLE else '✗'}")

    # Test 1: Búsqueda simple
    print("\n" + "-" * 60)
    print("[Test 1] Búsqueda web simple")
    print("-" * 60)

    search = WebSearchTool()
    results = search.search("Python programming language", num_results=3)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.title}")
        print(f"   URL: {r.url[:60]}...")
        print(f"   {r.snippet[:100]}...")

    # Test 2: Fetch de página
    if results:
        print("\n" + "-" * 60)
        print("[Test 2] Extracción de contenido")
        print("-" * 60)

        fetcher = WebFetcher()
        page = fetcher.fetch(results[0].url)

        print(f"Título: {page.title}")
        print(f"Éxito: {page.success}")
        print(f"Tiempo: {page.fetch_time:.2f}s")
        print(f"Contenido ({len(page.content)} chars):")
        print(page.content[:300] + "...")

    # Test 3: Investigación completa
    print("\n" + "-" * 60)
    print("[Test 3] Investigación autónoma")
    print("-" * 60)

    research = ResearchAgent(verbose=True)
    result = research.research("Qué es inteligencia artificial", depth="quick")

    print(f"\nResumen:")
    print(result.summary)
    print(f"\nHechos clave: {result.key_facts}")
    print(f"Temas relacionados: {result.related_topics}")
    print(f"Confianza: {result.confidence:.0%}")

    print("\n" + "=" * 60)
    print("  Demo completada!")
    print("=" * 60)
