"""
Sistema de Auto-Cuestionamiento Autónomo para THAU
Permite que THAU genere preguntas para sí mismo y aprenda de forma autónoma
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import random
import requests


class SelfQuestioningSystem:
    """
    Sistema que permite a THAU hacerse preguntas a sí mismo
    para aprender de forma autónoma con límites de seguridad
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:1.5b-base",
        max_questions_per_hour: int = 10,
        max_questions_per_day: int = 100,
        data_dir: Path = Path("./data/self_questioning")
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_questions_per_hour = max_questions_per_hour
        self.max_questions_per_day = max_questions_per_day
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Registro de actividad
        self.activity_log = self._load_activity_log()

        # Temas de exploración progresivos por edad cognitiva
        self.topic_templates = {
            0: [  # Recién nacido
                "¿Qué es {concept}?",
                "¿Cómo se usa {concept}?",
                "¿Para qué sirve {concept}?",
            ],
            1: [  # 1 año
                "¿Cuál es la diferencia entre {concept1} y {concept2}?",
                "¿Cómo funciona {concept}?",
                "¿Qué hace {concept}?",
            ],
            2: [  # 2 años
                "¿Por qué es importante {concept}?",
                "¿Cuándo se usa {concept}?",
                "¿Dónde encuentro {concept}?",
            ],
            3: [  # 3+ años
                "¿Cómo se relaciona {concept1} con {concept2}?",
                "¿Cuáles son las ventajas de {concept}?",
                "¿Qué problemas resuelve {concept}?",
            ]
        }

        # Conceptos base para explorar (irán creciendo)
        self.exploration_concepts = [
            # Programación básica
            "variable", "función", "loop", "condicional", "array",
            "objeto", "clase", "método", "parámetro", "retorno",

            # Conceptos intermedios
            "API", "base de datos", "servidor", "cliente", "HTTP",
            "JSON", "SQL", "REST", "autenticación", "encriptación",

            # Conceptos avanzados
            "algoritmo", "recursión", "complejidad", "optimización",
            "paralelismo", "concurrencia", "caché", "índice",

            # DevOps y arquitectura
            "contenedor", "orquestación", "CI/CD", "microservicio",
            "escalabilidad", "disponibilidad", "monitoreo",

            # Conceptos de datos
            "normalización", "transacción", "consistencia", "replicación",
            "particionamiento", "agregación", "indexación",
        ]

    def _load_activity_log(self) -> Dict:
        """Carga registro de actividad"""
        log_file = self.data_dir / "activity_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)

        return {
            "questions_asked": [],
            "total_questions": 0,
            "last_question_time": None,
            "daily_count": {},
            "hourly_count": {}
        }

    def _save_activity_log(self):
        """Guarda registro de actividad"""
        log_file = self.data_dir / "activity_log.json"

        with open(log_file, 'w') as f:
            json.dump(self.activity_log, f, indent=2)

    def _can_ask_question(self) -> tuple[bool, str]:
        """
        Verifica si puede hacer más preguntas según límites de seguridad

        Returns:
            (puede_preguntar, razón_si_no)
        """
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        # Verificar límite por hora
        hourly_count = self.activity_log.get("hourly_count", {}).get(current_hour, 0)
        if hourly_count >= self.max_questions_per_hour:
            return False, f"Límite por hora alcanzado ({self.max_questions_per_hour} preguntas/hora)"

        # Verificar límite por día
        daily_count = self.activity_log.get("daily_count", {}).get(current_day, 0)
        if daily_count >= self.max_questions_per_day:
            return False, f"Límite diario alcanzado ({self.max_questions_per_day} preguntas/día)"

        # Verificar tiempo mínimo entre preguntas (30 segundos)
        last_time = self.activity_log.get("last_question_time")
        if last_time:
            last_dt = datetime.fromisoformat(last_time)
            if (now - last_dt).total_seconds() < 30:
                return False, "Debe esperar al menos 30 segundos entre preguntas"

        return True, "OK"

    def _record_question(self, question: str, answer: str):
        """Registra una pregunta realizada"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        # Actualizar contadores
        if "hourly_count" not in self.activity_log:
            self.activity_log["hourly_count"] = {}
        if "daily_count" not in self.activity_log:
            self.activity_log["daily_count"] = {}

        self.activity_log["hourly_count"][current_hour] = \
            self.activity_log["hourly_count"].get(current_hour, 0) + 1
        self.activity_log["daily_count"][current_day] = \
            self.activity_log["daily_count"].get(current_day, 0) + 1

        # Registrar pregunta
        self.activity_log["questions_asked"].append({
            "timestamp": now.isoformat(),
            "question": question,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
        })

        self.activity_log["total_questions"] += 1
        self.activity_log["last_question_time"] = now.isoformat()

        # Limpiar contadores antiguos (más de 24 horas)
        self._cleanup_old_counters()

        self._save_activity_log()

    def _cleanup_old_counters(self):
        """Limpia contadores de hace más de 24 horas"""
        now = datetime.now()
        cutoff_hour = (now - timedelta(hours=24)).strftime("%Y-%m-%d-%H")
        cutoff_day = (now - timedelta(days=7)).strftime("%Y-%m-%d")

        # Limpiar horas antiguas
        old_hours = [h for h in self.activity_log.get("hourly_count", {}).keys() if h < cutoff_hour]
        for hour in old_hours:
            del self.activity_log["hourly_count"][hour]

        # Limpiar días antiguos
        old_days = [d for d in self.activity_log.get("daily_count", {}).keys() if d < cutoff_day]
        for day in old_days:
            del self.activity_log["daily_count"][day]

    def generate_question(self, cognitive_age: int = 0, topic: Optional[str] = None) -> Optional[str]:
        """
        Genera una pregunta apropiada para la edad cognitiva actual

        Args:
            cognitive_age: Edad cognitiva de THAU (0-15)
            topic: Tema específico (opcional, si no se proporciona se elige uno aleatorio)

        Returns:
            Pregunta generada o None si no se puede generar
        """
        # Seleccionar plantillas según edad
        age_bracket = min(cognitive_age, 3)  # Máximo nivel de complejidad 3
        templates = self.topic_templates.get(age_bracket, self.topic_templates[0])

        # Seleccionar concepto
        if not topic:
            topic = random.choice(self.exploration_concepts)

        # Generar pregunta usando plantilla
        template = random.choice(templates)

        # Si la plantilla necesita dos conceptos
        if "{concept1}" in template and "{concept2}" in template:
            concepts = random.sample(self.exploration_concepts, 2)
            question = template.format(concept1=concepts[0], concept2=concepts[1])
        else:
            question = template.format(concept=topic)

        return question

    def answer_question(self, question: str, timeout: int = 30) -> Optional[str]:
        """
        Responde una pregunta usando Ollama

        Args:
            question: Pregunta a responder
            timeout: Tiempo máximo de espera en segundos

        Returns:
            Respuesta generada o None si falla
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"Responde de forma clara y concisa en español:\n\n{question}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 300
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")

        return None

    def self_question_cycle(self, cognitive_age: int = 0) -> Optional[Dict]:
        """
        Ejecuta un ciclo completo de auto-cuestionamiento

        Returns:
            Diccionario con pregunta, respuesta y metadatos o None si no se puede ejecutar
        """
        # Verificar límites de seguridad
        can_ask, reason = self._can_ask_question()
        if not can_ask:
            print(f"⚠️  No se puede hacer pregunta: {reason}")
            return None

        print(f"\n💭 THAU se está haciendo una pregunta (Edad: {cognitive_age} años)...")

        # Generar pregunta
        question = self.generate_question(cognitive_age)
        if not question:
            print("❌ No se pudo generar pregunta")
            return None

        print(f"❓ Pregunta: {question}")

        # Responder pregunta
        answer = self.answer_question(question)
        if not answer:
            print("❌ No se pudo generar respuesta")
            return None

        print(f"✅ Respuesta generada ({len(answer)} caracteres)")

        # Registrar actividad
        self._record_question(question, answer)

        # Evaluar calidad de respuesta (simple heurística)
        confidence = self._evaluate_response_quality(answer)

        result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "cognitive_age": cognitive_age,
            "self_generated": True
        }

        # Guardar para análisis
        self._save_question_answer(result)

        return result

    def _evaluate_response_quality(self, answer: str) -> float:
        """
        Evalúa la calidad de una respuesta (heurística simple)

        Returns:
            Confianza entre 0.0 y 1.0
        """
        if not answer:
            return 0.0

        # Factores de calidad
        length_ok = 50 <= len(answer) <= 500  # Longitud razonable
        has_structure = any(word in answer.lower() for word in ["es", "permite", "sirve", "utiliza"])
        not_too_short = len(answer.split()) >= 10

        # Calcular confianza
        confidence = 0.0
        if length_ok:
            confidence += 0.4
        if has_structure:
            confidence += 0.4
        if not_too_short:
            confidence += 0.2

        return min(confidence, 1.0)

    def _save_question_answer(self, qa_data: Dict):
        """Guarda pregunta y respuesta para análisis"""
        qa_file = self.data_dir / f"qa_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_data, ensure_ascii=False) + '\n')

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema de auto-cuestionamiento"""
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        current_day = now.strftime("%Y-%m-%d")

        return {
            "total_questions": self.activity_log.get("total_questions", 0),
            "questions_this_hour": self.activity_log.get("hourly_count", {}).get(current_hour, 0),
            "questions_today": self.activity_log.get("daily_count", {}).get(current_day, 0),
            "max_per_hour": self.max_questions_per_hour,
            "max_per_day": self.max_questions_per_day,
            "last_question_time": self.activity_log.get("last_question_time"),
            "can_ask_now": self._can_ask_question()[0]
        }


# Testing
if __name__ == "__main__":
    print("🧠 Probando Sistema de Auto-Cuestionamiento\n")

    system = SelfQuestioningSystem(
        max_questions_per_hour=5,
        max_questions_per_day=20
    )

    # Simular 3 ciclos de auto-cuestionamiento
    for i in range(3):
        print(f"\n{'='*70}")
        print(f"Ciclo {i+1}/3")
        print('='*70)

        result = system.self_question_cycle(cognitive_age=0)

        if result:
            print(f"\n📊 Resultado:")
            print(f"   Pregunta: {result['question']}")
            print(f"   Respuesta: {result['answer'][:200]}...")
            print(f"   Confianza: {result['confidence']:.2f}")
        else:
            print("\n⚠️  No se pudo completar el ciclo")

        # Esperar un poco entre preguntas
        if i < 2:
            print("\n⏳ Esperando 30 segundos...")
            time.sleep(30)

    # Mostrar estadísticas
    print(f"\n{'='*70}")
    print("📈 Estadísticas Finales")
    print('='*70)
    stats = system.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
