#!/usr/bin/env python3
"""
THAU Advanced Training - SVG, Audio, Auto-Learning
Entrenamiento avanzado para hacer de THAU el mejor LLM para desarrollo

Capacidades a entrenar:
1. Generación de SVG (logos, iconos, gráficos)
2. Razonamiento paso a paso (CoT)
3. Desarrollo completo de aplicaciones
4. Procesamiento de Audio (Speech-to-Text con Whisper)
5. Auto-aprendizaje del código generado
6. Análisis de proyectos
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import logging
from typing import List, Dict, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Rutas base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LEARNING_DIR = DATA_DIR / "learning"

# Crear directorios
LEARNING_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PROCESADOR DE AUDIO (Speech-to-Text)
# ============================================================================

class AudioProcessor:
    """Procesa audio para speech-to-text usando Whisper"""

    def __init__(self):
        self.whisper_available = self._check_whisper()
        self.model = None

    def _check_whisper(self) -> bool:
        """Verifica si Whisper está disponible"""
        try:
            import whisper
            logger.info("✓ Whisper disponible")
            return True
        except ImportError:
            logger.warning("⚠ Whisper no disponible. Instala con: pip install openai-whisper")
            return False

    def load_model(self, model_size: str = "base"):
        """Carga el modelo Whisper"""
        if not self.whisper_available:
            return False
        try:
            import whisper
            logger.info(f"Cargando modelo Whisper ({model_size})...")
            self.model = whisper.load_model(model_size)
            logger.info("✓ Modelo Whisper cargado")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False

    def transcribe(self, audio_path: str, language: str = "es") -> Optional[str]:
        """Transcribe audio a texto"""
        if not self.model:
            if not self.load_model():
                return None

        try:
            result = self.model.transcribe(audio_path, language=language)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribiendo: {e}")
            return None

    def create_training_entry(self, audio_path: str, transcription: str) -> Dict:
        """Crea entrada de entrenamiento desde audio"""
        return {
            "instruction": f"Transcribe y procesa el siguiente comando de voz",
            "input": f"[Audio: {Path(audio_path).name}]",
            "output": transcription,
            "source": "audio_transcription",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# SISTEMA DE AUTO-APRENDIZAJE
# ============================================================================

class SelfLearner:
    """Sistema de auto-aprendizaje del código generado"""

    def __init__(self):
        self.learning_file = LEARNING_DIR / "auto_learning.jsonl"
        self.code_history = LEARNING_DIR / "code_history.jsonl"
        self.feedback_file = LEARNING_DIR / "feedback.jsonl"

    def save_generated_code(self, instruction: str, code: str,
                           language: str = "python", feedback: str = "success"):
        """Guarda código generado para auto-aprendizaje"""
        entry = {
            "instruction": instruction,
            "output": code,
            "language": language,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.code_history, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"✓ Código guardado: {instruction[:50]}...")

    def get_successful_examples(self, limit: int = 100) -> List[Dict]:
        """Obtiene ejemplos exitosos para reentrenamiento"""
        examples = []
        if self.code_history.exists():
            with open(self.code_history, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("feedback") == "success":
                            examples.append(entry)
                    except:
                        continue
        return examples[-limit:]

    def create_learning_dataset(self) -> Optional[Path]:
        """Crea dataset de aprendizaje desde código exitoso"""
        examples = self.get_successful_examples()

        if not examples:
            logger.warning("No hay ejemplos exitosos para aprendizaje")
            return None

        output_file = LEARNING_DIR / f"learning_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for ex in examples:
                training_entry = {
                    "instruction": ex["instruction"],
                    "output": ex["output"]
                }
                f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")

        logger.info(f"✓ Dataset de aprendizaje: {output_file} ({len(examples)} ejemplos)")
        return output_file

    def learn_from_project(self, project_path: Path):
        """Aprende del código de un proyecto"""
        if not project_path.exists():
            return

        code_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.svg']

        for ext in code_extensions:
            for file_path in project_path.rglob(f"*{ext}"):
                # Ignorar node_modules, venv, etc.
                if any(skip in str(file_path) for skip in ['node_modules', 'venv', '.git', '__pycache__']):
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8')
                    if len(content) > 100 and len(content) < 50000:  # Archivos razonables
                        instruction = f"Genera un archivo {ext.lstrip('.')} para {file_path.stem}"
                        self.save_generated_code(instruction, content, ext.lstrip('.'))
                except Exception:
                    continue

        logger.info(f"✓ Proyecto procesado: {project_path.name}")

    def learn_from_all_projects(self):
        """Procesa todos los proyectos para aprendizaje"""
        projects_dir = Path.home() / "thau_projects"

        if not projects_dir.exists():
            logger.warning(f"Directorio no encontrado: {projects_dir}")
            return

        projects = [d for d in projects_dir.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

        logger.info(f"Procesando {len(projects)} proyectos...")

        for project in projects:
            self.learn_from_project(project)

        logger.info("✓ Auto-aprendizaje completado")

# ============================================================================
# DATASET DE ENTRENAMIENTO SVG
# ============================================================================

SVG_TRAINING_DATA = [
    # Logos simples
    {
        "prompt": "Crea un logo SVG minimalista para una empresa de tecnología llamada 'TechFlow'",
        "response": """Aquí está el logo SVG para TechFlow:

**logo.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 60" width="200" height="60">
  <defs>
    <linearGradient id="techGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea"/>
      <stop offset="100%" style="stop-color:#764ba2"/>
    </linearGradient>
  </defs>
  <!-- Icono de flujo -->
  <circle cx="25" cy="30" r="8" fill="url(#techGrad)"/>
  <circle cx="45" cy="20" r="6" fill="url(#techGrad)" opacity="0.8"/>
  <circle cx="45" cy="40" r="6" fill="url(#techGrad)" opacity="0.8"/>
  <path d="M33 30 L39 20 M33 30 L39 40" stroke="url(#techGrad)" stroke-width="2" fill="none"/>
  <!-- Texto -->
  <text x="60" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    Tech<tspan fill="url(#techGrad)">Flow</tspan>
  </text>
</svg>
```

Este logo incluye:
- Un icono que representa flujo de datos con círculos conectados
- Gradiente moderno de azul a púrpura
- Tipografía limpia y profesional
- El texto "Flow" resaltado con el gradiente"""
    },
    {
        "prompt": "Genera un icono SVG de usuario/perfil para una aplicación",
        "response": """Aquí está el icono de usuario SVG:

**user-icon.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <!-- Cabeza -->
  <circle cx="12" cy="8" r="4"/>
  <!-- Cuerpo -->
  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
</svg>
```

Variante con fondo circular:

**user-avatar.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48">
  <!-- Fondo circular -->
  <circle cx="24" cy="24" r="23" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <!-- Cabeza -->
  <circle cx="24" cy="18" r="7" fill="#6366f1"/>
  <!-- Cuerpo -->
  <ellipse cx="24" cy="38" rx="12" ry="8" fill="#6366f1"/>
</svg>
```"""
    },
    {
        "prompt": "Crea un logo SVG para una cafetería llamada 'Coffee House'",
        "response": """Aquí está el logo SVG para Coffee House:

**coffee-house-logo.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">
  <defs>
    <linearGradient id="coffeeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#8B4513"/>
      <stop offset="100%" style="stop-color:#5D3A1A"/>
    </linearGradient>
  </defs>

  <!-- Taza -->
  <path d="M25 45 L25 85 Q25 95 35 95 L75 95 Q85 95 85 85 L85 45 Z"
        fill="url(#coffeeGrad)"/>

  <!-- Asa de la taza -->
  <path d="M85 55 Q105 55 105 70 Q105 85 85 85"
        fill="none" stroke="#8B4513" stroke-width="8" stroke-linecap="round"/>

  <!-- Vapor -->
  <path d="M40 35 Q45 25 40 15" fill="none" stroke="#999" stroke-width="3" stroke-linecap="round">
    <animate attributeName="d"
             values="M40 35 Q45 25 40 15;M40 35 Q35 25 40 15;M40 35 Q45 25 40 15"
             dur="2s" repeatCount="indefinite"/>
  </path>
  <path d="M55 30 Q60 20 55 10" fill="none" stroke="#999" stroke-width="3" stroke-linecap="round">
    <animate attributeName="d"
             values="M55 30 Q60 20 55 10;M55 30 Q50 20 55 10;M55 30 Q60 20 55 10"
             dur="2.5s" repeatCount="indefinite"/>
  </path>
  <path d="M70 35 Q75 25 70 15" fill="none" stroke="#999" stroke-width="3" stroke-linecap="round">
    <animate attributeName="d"
             values="M70 35 Q75 25 70 15;M70 35 Q65 25 70 15;M70 35 Q75 25 70 15"
             dur="1.8s" repeatCount="indefinite"/>
  </path>

  <!-- Texto -->
  <text x="60" y="115" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="#5D3A1A">
    COFFEE HOUSE
  </text>
</svg>
```

Características del logo:
- Taza de café con gradiente marrón
- Vapor animado que se mueve
- Tipografía serif elegante
- Diseño cálido y acogedor"""
    },
    {
        "prompt": "Genera iconos SVG para un menú de navegación (home, search, settings, profile)",
        "response": """Aquí están los iconos SVG para el menú de navegación:

**icons/home.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
  <polyline points="9 22 9 12 15 12 15 22"/>
</svg>
```

**icons/search.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="11" cy="11" r="8"/>
  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
</svg>
```

**icons/settings.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="3"/>
  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
</svg>
```

**icons/profile.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
  <circle cx="12" cy="7" r="4"/>
</svg>
```

Uso en React:
```jsx
import { ReactComponent as HomeIcon } from './icons/home.svg';
import { ReactComponent as SearchIcon } from './icons/search.svg';

<nav>
  <HomeIcon className="w-6 h-6 text-gray-600 hover:text-blue-500" />
  <SearchIcon className="w-6 h-6 text-gray-600 hover:text-blue-500" />
</nav>
```"""
    },
    {
        "prompt": "Crea un logo animado SVG para una startup de inteligencia artificial",
        "response": """Aquí está el logo animado SVG para una startup de IA:

**ai-logo-animated.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <defs>
    <linearGradient id="aiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#00d4ff"/>
      <stop offset="50%" style="stop-color:#7c3aed"/>
      <stop offset="100%" style="stop-color:#f472b6"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Cerebro estilizado -->
  <g filter="url(#glow)">
    <!-- Nodos de red neuronal -->
    <circle cx="100" cy="60" r="12" fill="url(#aiGrad)">
      <animate attributeName="r" values="12;14;12" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="60" cy="100" r="10" fill="url(#aiGrad)">
      <animate attributeName="r" values="10;12;10" dur="2.3s" repeatCount="indefinite"/>
    </circle>
    <circle cx="140" cy="100" r="10" fill="url(#aiGrad)">
      <animate attributeName="r" values="10;12;10" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="80" cy="140" r="8" fill="url(#aiGrad)">
      <animate attributeName="r" values="8;10;8" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="120" cy="140" r="8" fill="url(#aiGrad)">
      <animate attributeName="r" values="8;10;8" dur="2.1s" repeatCount="indefinite"/>
    </circle>

    <!-- Conexiones -->
    <line x1="100" y1="60" x2="60" y2="100" stroke="url(#aiGrad)" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="1.5s" repeatCount="indefinite"/>
    </line>
    <line x1="100" y1="60" x2="140" y2="100" stroke="url(#aiGrad)" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="1.7s" repeatCount="indefinite"/>
    </line>
    <line x1="60" y1="100" x2="80" y2="140" stroke="url(#aiGrad)" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="2s" repeatCount="indefinite"/>
    </line>
    <line x1="140" y1="100" x2="120" y2="140" stroke="url(#aiGrad)" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="1.9s" repeatCount="indefinite"/>
    </line>
    <line x1="80" y1="140" x2="120" y2="140" stroke="url(#aiGrad)" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="2.2s" repeatCount="indefinite"/>
    </line>
  </g>

  <!-- Texto -->
  <text x="100" y="180" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="url(#aiGrad)">
    NEURAL AI
  </text>
</svg>
```

Características:
- Red neuronal animada con nodos pulsantes
- Conexiones que parpadean simulando actividad
- Gradiente moderno de cian a púrpura a rosa
- Efecto de brillo (glow) para aspecto futurista
- Animaciones CSS puras, sin JavaScript"""
    },
]

# ============================================================================
# DATASET DE RAZONAMIENTO (Chain of Thought)
# ============================================================================

REASONING_TRAINING_DATA = [
    {
        "prompt": "Si tengo 3 manzanas y compro 2 bolsas con 4 manzanas cada una, ¿cuántas manzanas tengo en total?",
        "response": """Vamos a resolver esto paso a paso:

**Paso 1: Identificar lo que tenemos inicialmente**
- Manzanas iniciales: 3

**Paso 2: Calcular las manzanas compradas**
- Número de bolsas: 2
- Manzanas por bolsa: 4
- Total comprado: 2 × 4 = 8 manzanas

**Paso 3: Sumar todo**
- Manzanas iniciales + Manzanas compradas
- 3 + 8 = 11 manzanas

**Respuesta: Tengo 11 manzanas en total.**"""
    },
    {
        "prompt": "Una función recursiva calcula factorial. Si factorial(5) llama a factorial(4), ¿cuántas llamadas totales hay hasta llegar al caso base?",
        "response": """Analicemos el flujo de ejecución paso a paso:

**Paso 1: Entender la recursión de factorial**
```
factorial(n) = n × factorial(n-1)  si n > 1
factorial(1) = 1                   (caso base)
```

**Paso 2: Trazar las llamadas**
1. factorial(5) → llama a factorial(4)
2. factorial(4) → llama a factorial(3)
3. factorial(3) → llama a factorial(2)
4. factorial(2) → llama a factorial(1)
5. factorial(1) → caso base, retorna 1

**Paso 3: Contar las llamadas**
- factorial(5): 1 llamada
- factorial(4): 1 llamada
- factorial(3): 1 llamada
- factorial(2): 1 llamada
- factorial(1): 1 llamada (caso base)

**Total: 5 llamadas**

**Respuesta: Hay 5 llamadas en total, incluyendo factorial(5) y el caso base factorial(1).**

La pila de ejecución se vería así:
```
factorial(5)
  └── factorial(4)
        └── factorial(3)
              └── factorial(2)
                    └── factorial(1) ← caso base
```"""
    },
    {
        "prompt": "Diseña la arquitectura para un sistema de e-commerce que debe manejar 10,000 usuarios concurrentes",
        "response": """Vamos a diseñar esta arquitectura paso a paso:

**Paso 1: Análisis de Requerimientos**
- 10,000 usuarios concurrentes
- Operaciones típicas: navegación, búsqueda, carrito, checkout
- Necesitamos: alta disponibilidad, escalabilidad, baja latencia

**Paso 2: Estimación de Carga**
- 10,000 usuarios × ~5 requests/min = 50,000 req/min
- ≈ 833 requests/segundo
- Con picos: ~2,000 requests/segundo

**Paso 3: Arquitectura Propuesta**

```
                    ┌─────────────────┐
                    │   CloudFlare    │
                    │   (CDN + WAF)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Load Balancer  │
                    │    (HAProxy)    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  API Server │   │  API Server │   │  API Server │
    │   (Node.js) │   │   (Node.js) │   │   (Node.js) │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │    Redis    │ │  Postgres │ │ Elasticsearch│
       │   (Cache)   │ │ (Primary) │ │  (Search)   │
       └─────────────┘ └─────┬─────┘ └─────────────┘
                             │
                       ┌─────▼─────┐
                       │  Postgres │
                       │ (Replica) │
                       └───────────┘
```

**Paso 4: Componentes Clave**

1. **CDN (CloudFlare)**
   - Cachea assets estáticos
   - Protección DDoS
   - SSL termination

2. **Load Balancer**
   - Distribuye tráfico entre servidores
   - Health checks
   - Session affinity si necesario

3. **API Servers (3+ instancias)**
   - Stateless para escalado horizontal
   - Node.js por su manejo eficiente de I/O
   - Auto-scaling basado en CPU/memoria

4. **Redis**
   - Cache de sesiones
   - Cache de productos populares
   - Rate limiting
   - Colas de trabajo

5. **PostgreSQL**
   - Primary para escrituras
   - Read replicas para consultas
   - Connection pooling (PgBouncer)

6. **Elasticsearch**
   - Búsqueda de productos
   - Filtrado y facetas
   - Autocompletado

**Paso 5: Estrategias de Escalado**
- Horizontal: Añadir más API servers
- Vertical: Aumentar recursos de DB
- Sharding: Por región o por vendor

**Respuesta: Esta arquitectura soporta 10,000+ usuarios con alta disponibilidad y permite escalar según demanda.**"""
    },
    {
        "prompt": "Un bug hace que los usuarios vean datos de otros usuarios. ¿Cómo investigarías y resolverías esto?",
        "response": """Esto es un problema crítico de seguridad. Vamos a abordarlo sistemáticamente:

**Paso 1: Contención Inmediata (0-30 min)**
- Activar modo mantenimiento si es necesario
- Notificar al equipo de seguridad
- Preservar logs para análisis forense

**Paso 2: Identificar el Alcance**
Preguntas clave:
- ¿Qué datos se están filtrando?
- ¿Cuántos usuarios afectados?
- ¿Cuándo empezó?
- ¿Hay patrón en los usuarios afectados?

**Paso 3: Análisis de Causas Comunes**

1. **Problemas de Caché**
```python
# MAL - Cache sin usuario
cache_key = f"user_data"

# BIEN - Cache con usuario
cache_key = f"user_data:{user_id}"
```

2. **Race Conditions**
```python
# MAL - Variable global/compartida
current_user = None

def get_data():
    return db.query(User).filter_by(id=current_user.id)

# BIEN - Usuario en contexto de request
def get_data(user_id):
    return db.query(User).filter_by(id=user_id)
```

3. **Consultas SQL sin filtro**
```python
# MAL
SELECT * FROM orders;

# BIEN
SELECT * FROM orders WHERE user_id = $1;
```

4. **Tokens/Sesiones mezcladas**
```python
# Verificar que cada request valide el usuario
@require_auth
def get_profile(user_id):
    if request.user.id != user_id:
        raise Forbidden("No autorizado")
```

**Paso 4: Debugging**
```python
# Añadir logging temporal
import logging

@app.before_request
def log_request():
    logging.info(f"User: {current_user.id if current_user else 'None'}")
    logging.info(f"Session: {session.get('user_id')}")
    logging.info(f"Headers: {request.headers.get('Authorization')}")
```

**Paso 5: Verificación de Fix**
```python
# Test de regresión
def test_user_isolation():
    user1 = create_user("user1")
    user2 = create_user("user2")

    # Login como user1
    client.login(user1)
    data1 = client.get("/profile")

    # Login como user2
    client.login(user2)
    data2 = client.get("/profile")

    # Verificar aislamiento
    assert data1["id"] == user1.id
    assert data2["id"] == user2.id
    assert data1["id"] != data2["id"]
```

**Paso 6: Post-mortem**
- Documentar la causa raíz
- Añadir tests para prevenir regresión
- Revisar código similar
- Actualizar guías de desarrollo

**Respuesta: El enfoque sistemático de contención, investigación, fix y prevención es crítico para bugs de seguridad.**"""
    },
]

# ============================================================================
# DATASET DE DESARROLLO COMPLETO
# ============================================================================

DEVELOPMENT_TRAINING_DATA = [
    {
        "prompt": "Crea un componente de contador con React y hooks",
        "response": """Aquí está el componente de contador completo con React hooks:

**src/components/Counter.jsx**
```jsx
import React, { useState, useCallback } from 'react';

/**
 * Componente Counter - Un contador interactivo con React hooks
 *
 * @param {Object} props
 * @param {number} props.initialValue - Valor inicial del contador (default: 0)
 * @param {number} props.step - Incremento/decremento por clic (default: 1)
 * @param {number} props.min - Valor mínimo permitido
 * @param {number} props.max - Valor máximo permitido
 * @param {function} props.onChange - Callback cuando cambia el valor
 */
export default function Counter({
  initialValue = 0,
  step = 1,
  min = -Infinity,
  max = Infinity,
  onChange
}) {
  const [count, setCount] = useState(initialValue);

  const increment = useCallback(() => {
    setCount(prev => {
      const newValue = Math.min(prev + step, max);
      onChange?.(newValue);
      return newValue;
    });
  }, [step, max, onChange]);

  const decrement = useCallback(() => {
    setCount(prev => {
      const newValue = Math.max(prev - step, min);
      onChange?.(newValue);
      return newValue;
    });
  }, [step, min, onChange]);

  const reset = useCallback(() => {
    setCount(initialValue);
    onChange?.(initialValue);
  }, [initialValue, onChange]);

  return (
    <div className="flex flex-col items-center gap-4 p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold text-gray-800">Contador</h2>

      <div className="flex items-center gap-4">
        <button
          onClick={decrement}
          disabled={count <= min}
          className="w-12 h-12 rounded-full bg-red-500 text-white text-2xl font-bold
                     hover:bg-red-600 disabled:bg-gray-300 disabled:cursor-not-allowed
                     transition-colors duration-200 flex items-center justify-center"
          aria-label="Decrementar"
        >
          -
        </button>

        <span className="text-4xl font-bold text-gray-900 min-w-[100px] text-center">
          {count}
        </span>

        <button
          onClick={increment}
          disabled={count >= max}
          className="w-12 h-12 rounded-full bg-green-500 text-white text-2xl font-bold
                     hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed
                     transition-colors duration-200 flex items-center justify-center"
          aria-label="Incrementar"
        >
          +
        </button>
      </div>

      <button
        onClick={reset}
        className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800
                   hover:bg-gray-100 rounded-lg transition-colors duration-200"
      >
        Resetear
      </button>
    </div>
  );
}
```

**Uso del componente:**
```jsx
import Counter from './components/Counter';

function App() {
  const handleChange = (value) => {
    console.log('Nuevo valor:', value);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <Counter
        initialValue={10}
        step={5}
        min={0}
        max={100}
        onChange={handleChange}
      />
    </div>
  );
}
```

**Tests del componente:**

**src/components/Counter.test.jsx**
```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import Counter from './Counter';

describe('Counter', () => {
  test('muestra valor inicial', () => {
    render(<Counter initialValue={5} />);
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  test('incrementa al hacer clic en +', () => {
    render(<Counter initialValue={0} />);
    fireEvent.click(screen.getByLabelText('Incrementar'));
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  test('decrementa al hacer clic en -', () => {
    render(<Counter initialValue={5} />);
    fireEvent.click(screen.getByLabelText('Decrementar'));
    expect(screen.getByText('4')).toBeInTheDocument();
  });

  test('respeta valor máximo', () => {
    render(<Counter initialValue={9} max={10} />);
    fireEvent.click(screen.getByLabelText('Incrementar'));
    fireEvent.click(screen.getByLabelText('Incrementar'));
    expect(screen.getByText('10')).toBeInTheDocument();
  });

  test('resetea al valor inicial', () => {
    render(<Counter initialValue={5} />);
    fireEvent.click(screen.getByLabelText('Incrementar'));
    fireEvent.click(screen.getByText('Resetear'));
    expect(screen.getByText('5')).toBeInTheDocument();
  });
});
```"""
    },
    {
        "prompt": "Crea una API REST completa para gestión de tareas (TODO) con FastAPI",
        "response": """Aquí está la API REST completa para gestión de tareas:

**main.py**
```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import uuid

app = FastAPI(
    title="TODO API",
    description="API REST para gestión de tareas",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS
# ============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None

class Task(TaskBase):
    id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TaskList(BaseModel):
    tasks: List[Task]
    total: int
    page: int
    per_page: int

# ============================================================================
# BASE DE DATOS EN MEMORIA (para demo)
# ============================================================================

tasks_db: dict[str, Task] = {}

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
def health_check():
    \"\"\"Verificar que la API está funcionando\"\"\"
    return {
        "status": "ok",
        "message": "TODO API funcionando correctamente",
        "version": "1.0.0"
    }

@app.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED, tags=["Tasks"])
def create_task(task: TaskCreate):
    \"\"\"Crear una nueva tarea\"\"\"
    now = datetime.utcnow()
    new_task = Task(
        id=str(uuid.uuid4()),
        title=task.title,
        description=task.description,
        priority=task.priority,
        due_date=task.due_date,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now
    )
    tasks_db[new_task.id] = new_task
    return new_task

@app.get("/tasks", response_model=TaskList, tags=["Tasks"])
def list_tasks(
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    page: int = 1,
    per_page: int = 10
):
    \"\"\"Listar todas las tareas con filtros opcionales\"\"\"
    filtered = list(tasks_db.values())

    if status:
        filtered = [t for t in filtered if t.status == status]
    if priority:
        filtered = [t for t in filtered if t.priority == priority]

    # Ordenar por fecha de creación (más recientes primero)
    filtered.sort(key=lambda x: x.created_at, reverse=True)

    # Paginación
    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = filtered[start:end]

    return TaskList(
        tasks=paginated,
        total=total,
        page=page,
        per_page=per_page
    )

@app.get("/tasks/{task_id}", response_model=Task, tags=["Tasks"])
def get_task(task_id: str):
    \"\"\"Obtener una tarea por ID\"\"\"
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tarea con ID {task_id} no encontrada"
        )
    return tasks_db[task_id]

@app.put("/tasks/{task_id}", response_model=Task, tags=["Tasks"])
def update_task(task_id: str, task_update: TaskUpdate):
    \"\"\"Actualizar una tarea existente\"\"\"
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tarea con ID {task_id} no encontrada"
        )

    existing_task = tasks_db[task_id]
    update_data = task_update.model_dump(exclude_unset=True)

    # Si se marca como completada, registrar fecha
    if update_data.get("status") == TaskStatus.COMPLETED and existing_task.status != TaskStatus.COMPLETED:
        update_data["completed_at"] = datetime.utcnow()

    # Actualizar campos
    for field, value in update_data.items():
        setattr(existing_task, field, value)

    existing_task.updated_at = datetime.utcnow()
    tasks_db[task_id] = existing_task

    return existing_task

@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
def delete_task(task_id: str):
    \"\"\"Eliminar una tarea\"\"\"
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tarea con ID {task_id} no encontrada"
        )
    del tasks_db[task_id]

@app.post("/tasks/{task_id}/complete", response_model=Task, tags=["Tasks"])
def complete_task(task_id: str):
    \"\"\"Marcar una tarea como completada\"\"\"
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tarea con ID {task_id} no encontrada"
        )

    task = tasks_db[task_id]
    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.utcnow()
    task.updated_at = datetime.utcnow()

    return task

@app.get("/tasks/stats/summary", tags=["Stats"])
def get_stats():
    \"\"\"Obtener estadísticas de las tareas\"\"\"
    tasks = list(tasks_db.values())

    return {
        "total": len(tasks),
        "by_status": {
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "in_progress": len([t for t in tasks if t.status == TaskStatus.IN_PROGRESS]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
        },
        "by_priority": {
            "low": len([t for t in tasks if t.priority == TaskPriority.LOW]),
            "medium": len([t for t in tasks if t.priority == TaskPriority.MEDIUM]),
            "high": len([t for t in tasks if t.priority == TaskPriority.HIGH]),
            "urgent": len([t for t in tasks if t.priority == TaskPriority.URGENT]),
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**requirements.txt**
```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
```

**Ejecutar:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

La documentación estará disponible en:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc"""
    },
]

# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def create_training_dataset():
    """Crea el dataset de entrenamiento combinado"""

    all_data = []

    # Añadir datos de SVG
    for item in SVG_TRAINING_DATA:
        all_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })

    # Añadir datos de razonamiento
    for item in REASONING_TRAINING_DATA:
        all_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })

    # Añadir datos de desarrollo
    for item in DEVELOPMENT_TRAINING_DATA:
        all_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })

    return all_data


def save_dataset(data: list, output_path: str):
    """Guarda el dataset en formato JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Dataset guardado: {output_path} ({len(data)} ejemplos)")


def train_with_ollama(dataset_path: str, model_name: str = "thau:advanced"):
    """Entrena el modelo con Ollama"""

    print(f"\n🚀 Iniciando entrenamiento de {model_name}...")

    # Crear Modelfile
    modelfile_content = f"""# THAU Advanced - Desarrollo + SVG + Razonamiento
FROM thau:unified

SYSTEM \"\"\"Eres THAU Advanced, un asistente de IA especializado en:

1. **Desarrollo de Software**
   - Código completo y funcional
   - React, Next.js, Python, FastAPI
   - Arquitectura limpia y patrones de diseño

2. **Generación de SVG**
   - Logos y branding
   - Iconos y gráficos
   - Animaciones SVG

3. **Razonamiento Paso a Paso**
   - Chain of Thought (CoT)
   - Resolución de problemas complejos
   - Debugging y optimización

Siempre:
- Generas código COMPLETO, no fragmentos
- Explicas tu razonamiento
- Usas formatos claros con markdown
- Respondes en español

Para SVG usa el formato:
**nombre.svg**
```svg
<svg>...</svg>
```

Para código:
**ruta/archivo.ext**
```lenguaje
// código completo
```
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
"""

    modelfile_path = Path(__file__).parent.parent / "Modelfile_thau_advanced"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"Modelfile creado: {modelfile_path}")

    # Crear modelo con Ollama
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"✅ Modelo {model_name} creado exitosamente")

            # Actualizar thau:latest
            subprocess.run(
                ["ollama", "cp", model_name, "thau:latest"],
                capture_output=True
            )
            print(f"✅ thau:latest actualizado")
        else:
            print(f"❌ Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Timeout en creación del modelo")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="THAU Advanced Training System")
    parser.add_argument("--train", action="store_true", help="Ejecutar entrenamiento completo")
    parser.add_argument("--learn", action="store_true", help="Aprender de proyectos existentes")
    parser.add_argument("--audio", type=str, help="Procesar archivo de audio")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas")
    parser.add_argument("--all", action="store_true", help="Ejecutar todo: aprender + entrenar")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         THAU Advanced Training - SVG + Audio + Auto-Learning                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Capacidades:                                                                ║
║  • Generación de SVG (logos, iconos, gráficos animados)                     ║
║  • Procesamiento de Audio (Speech-to-Text con Whisper)                      ║
║  • Auto-aprendizaje del código generado                                      ║
║  • Análisis y mejora de proyectos                                           ║
║  • Razonamiento paso a paso (Chain of Thought)                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Instanciar clases
    learner = SelfLearner()
    audio_processor = AudioProcessor()

    if args.audio:
        # Procesar archivo de audio
        print(f"🎤 Procesando audio: {args.audio}")
        text = audio_processor.transcribe(args.audio)
        if text:
            print(f"\n📝 Transcripción:\n{text}")
            # Guardar para aprendizaje
            learner.save_generated_code(
                f"Transcribir comando de voz",
                text,
                "text",
                "success"
            )
        else:
            print("❌ No se pudo procesar el audio")
        return

    if args.stats:
        # Mostrar estadísticas
        examples = learner.get_successful_examples(limit=1000)
        print(f"\n{'='*50}")
        print("ESTADÍSTICAS DE AUTO-APRENDIZAJE")
        print(f"{'='*50}")
        print(f"Total ejemplos exitosos: {len(examples)}")

        if examples:
            languages = {}
            for ex in examples:
                lang = ex.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1

            print("\nPor lenguaje:")
            for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
                print(f"  {lang}: {count}")
        print(f"{'='*50}\n")
        return

    if args.learn or args.all:
        # Aprender de proyectos existentes
        print("📚 Iniciando auto-aprendizaje de proyectos...")
        learner.learn_from_all_projects()
        learner.create_learning_dataset()

    if args.train or args.all or (not args.learn and not args.stats and not args.audio):
        # Ejecutar entrenamiento
        datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Cargar datasets adicionales
        additional_datasets = []

        # SVG dataset
        svg_file = datasets_dir / "svg_assets_training.jsonl"
        if svg_file.exists():
            logger.info(f"✓ Cargando: {svg_file.name}")
            additional_datasets.append(svg_file)

        # Project analysis dataset
        project_file = datasets_dir / "project_analysis_training.jsonl"
        if project_file.exists():
            logger.info(f"✓ Cargando: {project_file.name}")
            additional_datasets.append(project_file)

        # Auto-learning dataset
        learning_file = learner.create_learning_dataset()
        if learning_file:
            additional_datasets.append(learning_file)

        # Crear dataset principal
        print("📊 Creando dataset de entrenamiento...")
        data = create_training_dataset()

        # Añadir datos de archivos adicionales
        for dataset_file in additional_datasets:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'instruction' in entry and 'output' in entry:
                            data.append({
                                "messages": [
                                    {"role": "user", "content": entry['instruction']},
                                    {"role": "assistant", "content": entry['output']}
                                ]
                            })
                    except:
                        continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = datasets_dir / f"thau_advanced_{timestamp}.jsonl"
        save_dataset(data, str(output_path))

        print(f"""
📊 Estadísticas del dataset:
   - SVG: {len(SVG_TRAINING_DATA)} ejemplos base
   - Razonamiento: {len(REASONING_TRAINING_DATA)} ejemplos
   - Desarrollo: {len(DEVELOPMENT_TRAINING_DATA)} ejemplos
   - Datasets adicionales: {len(additional_datasets)} archivos
   - Total combinado: {len(data)} ejemplos
""")

        # Entrenar
        train_with_ollama(str(output_path))

        print("\n✅ Entrenamiento completado!")
        print("\n📌 Modelo disponible como: thau:advanced")
        print("   Usa: ollama run thau:advanced")


if __name__ == "__main__":
    main()
