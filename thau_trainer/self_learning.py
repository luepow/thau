"""
Sistema de Auto-Aprendizaje y Auto-Generación de Datasets para THAU
Permite que THAU genere sus propios datos de entrenamiento de forma autónoma
"""

import json
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict
import hashlib


class KnowledgeGapDetector:
    """Detecta brechas de conocimiento en las respuestas de THAU"""

    def __init__(self):
        self.low_confidence_threshold = 0.6
        self.uncertainty_markers = [
            "no estoy seguro",
            "no sé",
            "no tengo información",
            "desconozco",
            "tal vez",
            "posiblemente",
            "creo que",
        ]
        self.gaps_log = Path("./data/logs/knowledge_gaps.jsonl")
        self.gaps_log.parent.mkdir(parents=True, exist_ok=True)

    def detect_gap(self, question: str, answer: str, confidence: float = None) -> Dict:
        """
        Detecta si hay brecha de conocimiento

        Returns:
            Dict con información de la brecha o None
        """
        gap_detected = False
        gap_type = None
        reason = []

        # 1. Respuesta muy corta (menos de 20 chars)
        if len(answer.strip()) < 20:
            gap_detected = True
            gap_type = "short_response"
            reason.append("Respuesta demasiado corta")

        # 2. Marcadores de incertidumbre
        answer_lower = answer.lower()
        for marker in self.uncertainty_markers:
            if marker in answer_lower:
                gap_detected = True
                gap_type = "uncertainty"
                reason.append(f"Marcador de incertidumbre: '{marker}'")
                break

        # 3. Confianza baja
        if confidence is not None and confidence < self.low_confidence_threshold:
            gap_detected = True
            gap_type = "low_confidence"
            reason.append(f"Confianza baja: {confidence:.2f}")

        if gap_detected:
            gap_info = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "gap_type": gap_type,
                "reasons": reason,
                "topic": self._extract_topic(question)
            }

            # Log the gap
            self._log_gap(gap_info)

            return gap_info

        return None

    def _extract_topic(self, question: str) -> str:
        """Extrae el tópico principal de una pregunta"""
        # Simple: primera palabra significativa
        words = question.lower().split()
        stopwords = {"qué", "cuál", "cómo", "por", "para", "es", "un", "una", "el", "la", "los", "las"}

        for word in words:
            clean = word.strip("¿?.,;:")
            if clean not in stopwords and len(clean) > 3:
                return clean

        return "general"

    def _log_gap(self, gap_info: Dict):
        """Registra brecha detectada"""
        with open(self.gaps_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(gap_info, ensure_ascii=False) + '\n')

    def get_top_gaps(self, n: int = 10) -> List[Dict]:
        """Obtiene las brechas más frecuentes por tópico"""
        if not self.gaps_log.exists():
            return []

        topic_counts = defaultdict(list)

        with open(self.gaps_log, 'r', encoding='utf-8') as f:
            for line in f:
                gap = json.loads(line)
                topic_counts[gap['topic']].append(gap)

        # Ordenar por frecuencia
        sorted_topics = sorted(topic_counts.items(), key=lambda x: len(x[1]), reverse=True)

        return sorted_topics[:n]


class DatasetGenerator:
    """Genera datasets sintéticos usando el modelo base"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.generated_datasets_dir = Path("./data/datasets/auto_generated")
        self.generated_datasets_dir.mkdir(parents=True, exist_ok=True)

    def generate_examples_for_topic(
        self,
        topic: str,
        num_examples: int = 5,
        difficulty: str = "medium",
        age_level: int = 3
    ) -> List[Dict]:
        """
        Genera ejemplos de entrenamiento para un tópico específico

        Args:
            topic: Tópico a cubrir
            num_examples: Cantidad de ejemplos a generar
            difficulty: Nivel de dificultad (easy, medium, hard)
            age_level: Edad cognitiva objetivo (0-15)
        """

        # Mapear edad a complejidad
        complexity_map = {
            0: "palabras individuales",
            1: "frases de 2-3 palabras",
            3: "explicaciones simples de 2-3 oraciones",
            6: "explicaciones con ejemplos paso a paso",
            11: "razonamiento abstracto con múltiples perspectivas",
            13: "análisis crítico y razonamiento multi-paso",
            15: "explicaciones expertas con profundidad técnica"
        }

        complexity = complexity_map.get(age_level, "explicaciones de dificultad media")

        prompt = f"""Eres un generador de datasets educativos.

Tu tarea es generar {num_examples} pares de pregunta-respuesta sobre el tópico: "{topic}".

Nivel de complejidad: {complexity}
Edad objetivo: {age_level} años

IMPORTANTE:
1. Las respuestas deben ser apropiadas para la edad cognitiva
2. Cada pregunta debe ser diferente
3. Cubre diferentes aspectos del tópico
4. Las respuestas deben ser educativas y precisas

Formato de salida (JSON):
{{
  "examples": [
    {{
      "instruction": "pregunta aquí",
      "input": "",
      "output": "respuesta apropiada aquí"
    }}
  ]
}}

Genera los {num_examples} ejemplos ahora:"""

        try:
            # Llamar a Ollama para generar
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                generated_text = response.json()['response']

                # Intentar extraer JSON
                examples = self._extract_examples(generated_text)

                if examples:
                    # Guardar dataset generado
                    self._save_generated_dataset(topic, examples, age_level)
                    return examples

        except Exception as e:
            print(f"Error generando ejemplos: {e}")

        return []

    def _extract_examples(self, text: str) -> List[Dict]:
        """Extrae ejemplos del texto generado"""
        try:
            # Buscar bloque JSON
            start = text.find('{')
            end = text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)

                if 'examples' in data:
                    return data['examples']
        except:
            pass

        return []

    def _save_generated_dataset(self, topic: str, examples: List[Dict], age_level: int):
        """Guarda dataset generado automáticamente"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_safe = topic.replace(" ", "_").replace("/", "_")
        filename = f"age_{age_level}_{topic_safe}_{timestamp}.jsonl"

        filepath = self.generated_datasets_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"✅ Dataset auto-generado guardado: {filepath}")

    def generate_from_gaps(self, gaps: List[Tuple[str, List[Dict]]], age_level: int = 3) -> int:
        """
        Genera datasets para cubrir brechas de conocimiento

        Returns:
            Número total de ejemplos generados
        """
        total_generated = 0

        for topic, gap_list in gaps:
            print(f"\n🔍 Generando ejemplos para tópico: {topic} ({len(gap_list)} brechas detectadas)")

            # Generar 5-10 ejemplos por tópico
            num_examples = min(10, max(5, len(gap_list) * 2))

            examples = self.generate_examples_for_topic(
                topic=topic,
                num_examples=num_examples,
                age_level=age_level
            )

            total_generated += len(examples)
            print(f"  Generados: {len(examples)} ejemplos")

        return total_generated


class SelfLearningManager:
    """
    Gestor principal del sistema de auto-aprendizaje
    Coordina detección de brechas y generación de datasets
    """

    def __init__(self, cognitive_manager=None):
        self.gap_detector = KnowledgeGapDetector()
        self.dataset_generator = DatasetGenerator()
        self.cognitive_manager = cognitive_manager

        self.stats_file = Path("./data/logs/self_learning_stats.json")
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict:
        """Carga estadísticas de auto-aprendizaje"""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)

        return {
            "total_gaps_detected": 0,
            "total_datasets_generated": 0,
            "total_examples_generated": 0,
            "last_generation": None,
            "topics_covered": []
        }

    def _save_stats(self):
        """Guarda estadísticas"""
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def process_interaction(self, question: str, answer: str, confidence: float = None):
        """
        Procesa una interacción para detectar brechas de conocimiento
        """
        gap = self.gap_detector.detect_gap(question, answer, confidence)

        if gap:
            self.stats["total_gaps_detected"] += 1
            print(f"\n⚠️  Brecha de conocimiento detectada:")
            print(f"   Tópico: {gap['topic']}")
            print(f"   Razones: {', '.join(gap['reasons'])}")

            self._save_stats()

            return gap

        return None

    def auto_generate_missing_knowledge(self, min_gaps: int = 5) -> Dict:
        """
        Genera automáticamente datasets para cubrir brechas de conocimiento

        Args:
            min_gaps: Mínimo de brechas por tópico para generar dataset
        """
        # Obtener brechas más frecuentes
        top_gaps = self.gap_detector.get_top_gaps(n=10)

        if not top_gaps:
            print("ℹ️  No hay brechas de conocimiento para cubrir")
            return {"generated": 0}

        # Filtrar por frecuencia
        gaps_to_fill = [(topic, gaps) for topic, gaps in top_gaps if len(gaps) >= min_gaps]

        if not gaps_to_fill:
            print(f"ℹ️  No hay tópicos con suficientes brechas (mínimo: {min_gaps})")
            return {"generated": 0}

        print(f"\n🧠 Iniciando auto-generación de conocimiento...")
        print(f"   Tópicos a cubrir: {len(gaps_to_fill)}")

        # Obtener edad cognitiva actual
        age_level = 3  # Default
        if self.cognitive_manager:
            age_level = self.cognitive_manager.current_age

        # Generar datasets
        total_examples = self.dataset_generator.generate_from_gaps(gaps_to_fill, age_level)

        # Actualizar estadísticas
        self.stats["total_datasets_generated"] += len(gaps_to_fill)
        self.stats["total_examples_generated"] += total_examples
        self.stats["last_generation"] = datetime.now().isoformat()
        self.stats["topics_covered"].extend([topic for topic, _ in gaps_to_fill])

        self._save_stats()

        print(f"\n✅ Auto-generación completada:")
        print(f"   Datasets generados: {len(gaps_to_fill)}")
        print(f"   Ejemplos totales: {total_examples}")

        return {
            "datasets_generated": len(gaps_to_fill),
            "examples_generated": total_examples,
            "topics": [topic for topic, _ in gaps_to_fill]
        }

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del auto-aprendizaje"""
        return self.stats


# CLI para testing
if __name__ == "__main__":
    manager = SelfLearningManager()

    # Simular algunas interacciones con brechas
    print("Simulando detección de brechas...")

    manager.process_interaction(
        "¿Qué es la computación cuántica?",
        "No estoy seguro",
        confidence=0.3
    )

    manager.process_interaction(
        "¿Cómo funciona blockchain?",
        "No tengo información suficiente",
        confidence=0.4
    )

    manager.process_interaction(
        "Explica machine learning",
        "Es algo relacionado con IA",
        confidence=0.5
    )

    # Auto-generar conocimiento
    print("\n" + "="*60)
    result = manager.auto_generate_missing_knowledge(min_gaps=1)

    print("\n" + "="*60)
    print("\nEstadísticas:")
    print(json.dumps(manager.get_stats(), indent=2))
