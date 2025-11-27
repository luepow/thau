"""
Sistema de Aprendizaje en Vivo para THAU
Permite que THAU aprenda y evolucione con cada interacción,
como un ser vivo que mejora con la experiencia.

Características:
- Aprendizaje inmediato de cada conversación
- Detección de temas nuevos para investigar
- Memoria a corto y largo plazo
- Auto-corrección basada en feedback
- Evolución gradual del conocimiento
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading


class LiveLearningSystem:
    """
    Sistema de aprendizaje en vivo que permite a THAU:
    1. Aprender de cada pregunta que le hacen
    2. Recordar interacciones previas
    3. Mejorar respuestas basándose en feedback
    4. Investigar automáticamente temas desconocidos
    """

    def __init__(
        self,
        data_dir: Path = Path("./data/live_learning"),
        qa_output_dir: Path = Path("./data/self_questioning"),
        memory_size: int = 100,  # Interacciones en memoria corta
        learn_threshold: float = 0.7,  # Umbral de confianza para aprender
        auto_research: bool = True,  # Investigar temas desconocidos
    ):
        self.data_dir = data_dir
        self.qa_output_dir = qa_output_dir
        self.memory_size = memory_size
        self.learn_threshold = learn_threshold
        self.auto_research = auto_research

        # Crear directorios
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qa_output_dir.mkdir(parents=True, exist_ok=True)

        # Memoria a corto plazo (últimas interacciones)
        self.short_term_memory: deque = deque(maxlen=memory_size)

        # Registro de aprendizaje
        self.learning_log = self.data_dir / "learning_log.jsonl"

        # Temas conocidos vs desconocidos
        self.known_topics = self._load_known_topics()
        self.unknown_topics_queue: List[str] = []

        # Estadísticas de interacción
        self.stats = self._load_stats()

        # Lock para thread safety
        self._lock = threading.Lock()

        print(f"🧠 LiveLearningSystem inicializado")
        print(f"   Temas conocidos: {len(self.known_topics)}")
        print(f"   Interacciones totales: {self.stats.get('total_interactions', 0)}")

    # ==================== CORE LEARNING ====================

    def process_interaction(
        self,
        user_input: str,
        model_response: str,
        confidence: float = 0.8,
        feedback: Optional[str] = None,
        learn_immediately: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa una interacción usuario-modelo y aprende de ella.

        Args:
            user_input: Pregunta/input del usuario
            model_response: Respuesta generada por THAU
            confidence: Confianza en la respuesta (0-1)
            feedback: Feedback opcional del usuario
            learn_immediately: Si aprender inmediatamente o encolar

        Returns:
            Dict con resultado del procesamiento
        """
        with self._lock:
            timestamp = datetime.now()

            # Crear registro de interacción
            interaction = {
                "id": self._generate_interaction_id(user_input, timestamp),
                "timestamp": timestamp.isoformat(),
                "user_input": user_input,
                "model_response": model_response,
                "confidence": confidence,
                "feedback": feedback,
                "topics": self._extract_topics(user_input),
                "learned": False
            }

            # Agregar a memoria corta
            self.short_term_memory.append(interaction)

            # Actualizar estadísticas
            self.stats["total_interactions"] = self.stats.get("total_interactions", 0) + 1
            self.stats["last_interaction"] = timestamp.isoformat()

            result = {
                "interaction_id": interaction["id"],
                "topics_detected": interaction["topics"],
                "will_learn": False,
                "needs_research": []
            }

            # Decidir si aprender de esta interacción
            should_learn = (
                confidence >= self.learn_threshold and
                len(model_response) > 50 and
                not self._is_duplicate_interaction(user_input)
            )

            if should_learn and learn_immediately:
                self._learn_from_interaction(interaction)
                interaction["learned"] = True
                result["will_learn"] = True

            # Detectar temas desconocidos
            unknown = self._detect_unknown_topics(interaction["topics"])
            if unknown:
                result["needs_research"] = unknown
                self.unknown_topics_queue.extend(unknown)

            # Guardar log
            self._log_interaction(interaction)

            # Auto-investigar si está habilitado
            if self.auto_research and unknown:
                threading.Thread(
                    target=self._research_topics,
                    args=(unknown,),
                    daemon=True
                ).start()

            self._save_stats()

            return result

    def learn_from_feedback(
        self,
        interaction_id: str,
        feedback: str,
        correct_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aprende del feedback del usuario sobre una interacción.

        Args:
            interaction_id: ID de la interacción
            feedback: Feedback (positive/negative/correction)
            correct_response: Respuesta correcta si el feedback es negativo

        Returns:
            Resultado del aprendizaje
        """
        # Buscar interacción en memoria
        interaction = None
        for item in self.short_term_memory:
            if item["id"] == interaction_id:
                interaction = item
                break

        if not interaction:
            return {"success": False, "error": "Interacción no encontrada"}

        result = {
            "success": True,
            "feedback_type": feedback,
            "learned": False
        }

        if feedback == "positive":
            # Reforzar este conocimiento
            interaction["confidence"] = min(1.0, interaction["confidence"] + 0.1)
            if not interaction["learned"]:
                self._learn_from_interaction(interaction)
                result["learned"] = True

        elif feedback == "negative" and correct_response:
            # Aprender la corrección
            corrected = interaction.copy()
            corrected["model_response"] = correct_response
            corrected["confidence"] = 0.9  # Alta confianza en corrección del usuario
            self._learn_from_interaction(corrected)
            result["learned"] = True
            result["correction_applied"] = True

        # Actualizar stats
        self.stats["feedback_received"] = self.stats.get("feedback_received", 0) + 1
        if feedback == "positive":
            self.stats["positive_feedback"] = self.stats.get("positive_feedback", 0) + 1
        else:
            self.stats["negative_feedback"] = self.stats.get("negative_feedback", 0) + 1

        self._save_stats()

        return result

    # ==================== MEMORY & RECALL ====================

    def recall_similar(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Busca interacciones similares en la memoria.

        Args:
            query: Texto a buscar
            limit: Número máximo de resultados

        Returns:
            Lista de interacciones similares
        """
        query_words = set(query.lower().split())
        scored_interactions = []

        for interaction in self.short_term_memory:
            input_words = set(interaction["user_input"].lower().split())
            overlap = len(query_words & input_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(input_words))
                scored_interactions.append((score, interaction))

        # Ordenar por score y retornar top N
        scored_interactions.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_interactions[:limit]]

    def get_context_for_query(self, query: str) -> str:
        """
        Genera contexto relevante basado en interacciones previas.

        Args:
            query: Pregunta actual

        Returns:
            Contexto formateado para incluir en el prompt
        """
        similar = self.recall_similar(query, limit=3)

        if not similar:
            return ""

        context_parts = ["Contexto de conversaciones previas:"]
        for interaction in similar:
            context_parts.append(
                f"- Usuario preguntó: {interaction['user_input'][:100]}..."
                f"\n  Respuesta: {interaction['model_response'][:200]}..."
            )

        return "\n".join(context_parts)

    # ==================== TOPIC MANAGEMENT ====================

    def _extract_topics(self, text: str) -> List[str]:
        """Extrae temas/keywords del texto"""
        # Keywords técnicos comunes
        tech_keywords = {
            "python", "javascript", "java", "c++", "rust", "go", "typescript",
            "react", "angular", "vue", "node", "django", "flask", "fastapi",
            "sql", "mongodb", "postgresql", "redis", "docker", "kubernetes",
            "api", "rest", "graphql", "websocket", "microservices",
            "machine learning", "deep learning", "neural network", "ai",
            "algorithm", "data structure", "database", "cache", "async",
            "function", "class", "method", "variable", "loop", "array",
        }

        words = text.lower().split()
        topics = []

        for word in words:
            # Limpiar puntuación
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in tech_keywords:
                topics.append(clean_word)

        # También detectar frases de dos palabras
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        for bigram in bigrams:
            clean_bigram = ' '.join(''.join(c for c in w if c.isalnum()) for w in bigram.split())
            if clean_bigram in tech_keywords:
                topics.append(clean_bigram)

        return list(set(topics))

    def _detect_unknown_topics(self, topics: List[str]) -> List[str]:
        """Detecta temas que no conocemos bien"""
        unknown = []
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower not in self.known_topics:
                # Verificar si tenemos suficientes Q&A sobre este tema
                qa_count = self._count_qa_for_topic(topic_lower)
                if qa_count < 5:  # Umbral mínimo
                    unknown.append(topic)
        return unknown

    def _count_qa_for_topic(self, topic: str) -> int:
        """Cuenta cuántos Q&A tenemos sobre un tema"""
        count = 0
        for qa_file in self.qa_output_dir.glob("qa_*.jsonl"):
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if topic in entry.get("question", "").lower():
                                count += 1
                            if topic in entry.get("answer", "").lower():
                                count += 1
                        except:
                            pass
            except:
                pass
        return count

    def add_known_topic(self, topic: str):
        """Marca un tema como conocido"""
        self.known_topics.add(topic.lower())
        self._save_known_topics()

    # ==================== LEARNING INTERNALS ====================

    def _learn_from_interaction(self, interaction: Dict):
        """Aprende de una interacción guardándola como Q&A"""
        today = datetime.now().strftime("%Y%m%d")
        qa_file = self.qa_output_dir / f"qa_{today}.jsonl"

        entry = {
            "question": interaction["user_input"],
            "answer": interaction["model_response"],
            "category": "live_learning",
            "source": "user_interaction",
            "confidence": interaction["confidence"],
            "topics": interaction.get("topics", []),
            "timestamp": interaction["timestamp"],
            "cognitive_age": 12
        }

        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Marcar temas como conocidos
        for topic in interaction.get("topics", []):
            self.add_known_topic(topic)

        # Actualizar stats
        self.stats["total_learned"] = self.stats.get("total_learned", 0) + 1

    def _research_topics(self, topics: List[str]):
        """Investiga temas desconocidos en background"""
        try:
            from thau_trainer.external_learning import ExternalLearningSystem

            external = ExternalLearningSystem()
            for topic in topics[:3]:  # Limitar investigación
                print(f"🔬 Auto-investigando: {topic}")
                external.search_and_learn(topic, topic=topic)
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ Error en auto-investigación: {e}")

    def _is_duplicate_interaction(self, user_input: str) -> bool:
        """Verifica si es una interacción duplicada reciente"""
        input_hash = hashlib.md5(user_input.lower().encode()).hexdigest()

        for interaction in list(self.short_term_memory)[-10:]:
            existing_hash = hashlib.md5(
                interaction["user_input"].lower().encode()
            ).hexdigest()
            if input_hash == existing_hash:
                return True
        return False

    def _generate_interaction_id(self, user_input: str, timestamp: datetime) -> str:
        """Genera ID único para interacción"""
        content = f"{timestamp.isoformat()}-{user_input[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # ==================== PERSISTENCE ====================

    def _log_interaction(self, interaction: Dict):
        """Guarda interacción en log"""
        with open(self.learning_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + '\n')

    def _load_known_topics(self) -> set:
        """Carga temas conocidos"""
        topics_file = self.data_dir / "known_topics.json"
        if topics_file.exists():
            with open(topics_file, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_known_topics(self):
        """Guarda temas conocidos"""
        topics_file = self.data_dir / "known_topics.json"
        with open(topics_file, 'w') as f:
            json.dump(list(self.known_topics), f)

    def _load_stats(self) -> Dict:
        """Carga estadísticas"""
        stats_file = self.data_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_stats(self):
        """Guarda estadísticas"""
        stats_file = self.data_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    # ==================== PUBLIC API ====================

    def get_stats(self) -> Dict:
        """Retorna estadísticas del sistema"""
        return {
            **self.stats,
            "memory_size": len(self.short_term_memory),
            "known_topics": len(self.known_topics),
            "pending_research": len(self.unknown_topics_queue)
        }

    def get_learning_summary(self) -> str:
        """Genera un resumen del estado de aprendizaje"""
        stats = self.get_stats()
        return f"""
🧠 THAU Live Learning Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total interacciones: {stats.get('total_interactions', 0)}
📚 Conocimiento aprendido: {stats.get('total_learned', 0)} Q&A
💭 Memoria activa: {stats.get('memory_size', 0)} interacciones
🏷️  Temas conocidos: {stats.get('known_topics', 0)}
🔬 Temas por investigar: {stats.get('pending_research', 0)}
👍 Feedback positivo: {stats.get('positive_feedback', 0)}
👎 Feedback negativo: {stats.get('negative_feedback', 0)}
⏰ Última interacción: {stats.get('last_interaction', 'N/A')}
        """


# ==================== CHAT WRAPPER ====================

class ThauLiveChatWrapper:
    """
    Wrapper que integra Live Learning con el chat de THAU.
    Cada conversación mejora automáticamente el modelo.
    """

    def __init__(self, model_manager=None):
        self.live_learning = LiveLearningSystem()
        self.model_manager = model_manager

    def chat(
        self,
        user_message: str,
        context: Optional[str] = None,
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa un mensaje de chat con aprendizaje en vivo.

        Args:
            user_message: Mensaje del usuario
            context: Contexto adicional
            learn: Si aprender de esta interacción

        Returns:
            Respuesta y metadata de aprendizaje
        """
        # Obtener contexto de interacciones previas
        memory_context = self.live_learning.get_context_for_query(user_message)

        # Generar respuesta (aquí iría la llamada al modelo)
        # Por ahora simulamos
        if self.model_manager:
            response = self.model_manager.generate_text(
                f"{memory_context}\n\nUsuario: {user_message}\nAsistente:",
                max_new_tokens=200
            )
        else:
            response = f"[Respuesta simulada para: {user_message}]"

        # Procesar interacción para aprendizaje
        learning_result = {}
        if learn:
            learning_result = self.live_learning.process_interaction(
                user_input=user_message,
                model_response=response,
                confidence=0.8
            )

        return {
            "response": response,
            "learning": learning_result,
            "used_memory": bool(memory_context)
        }

    def provide_feedback(
        self,
        interaction_id: str,
        is_helpful: bool,
        correct_response: Optional[str] = None
    ) -> Dict:
        """Proporciona feedback sobre una respuesta"""
        feedback_type = "positive" if is_helpful else "negative"
        return self.live_learning.learn_from_feedback(
            interaction_id,
            feedback_type,
            correct_response
        )


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THAU Live Learning System")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen")
    parser.add_argument("--chat", action="store_true", help="Modo chat interactivo")

    args = parser.parse_args()

    system = LiveLearningSystem()

    if args.stats:
        print(json.dumps(system.get_stats(), indent=2))
    elif args.summary:
        print(system.get_learning_summary())
    elif args.chat:
        print("🧠 THAU Live Chat (escriba 'salir' para terminar)")
        print("-" * 50)

        wrapper = ThauLiveChatWrapper()

        while True:
            user_input = input("\n👤 Tú: ").strip()
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break

            result = wrapper.chat(user_input)
            print(f"\n🤖 THAU: {result['response']}")

            if result.get('learning', {}).get('will_learn'):
                print("   📚 [Aprendiendo de esta interacción]")

            if result.get('learning', {}).get('needs_research'):
                topics = result['learning']['needs_research']
                print(f"   🔬 [Investigando: {', '.join(topics)}]")

        print("\n" + system.get_learning_summary())
    else:
        parser.print_help()
