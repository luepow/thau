"""
Entrenador Integrado de THAU
Combina desarrollo cognitivo, auto-aprendizaje, memoria vectorizada y multilingüismo
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading

from thau_trainer.cognitive_development import CognitiveDevelopmentManager, COGNITIVE_STAGES
from thau_trainer.self_learning import SelfLearningManager, DatasetGenerator
from thau_trainer.vector_memory import EfficientVectorMemory
from thau_trainer.language_learning import MultilingualLearningManager
from thau_trainer.training_service import ThauTrainingService
from thau_trainer.data_manager import DataManager
from thau_trainer.self_questioning import SelfQuestioningSystem
from thau_trainer.own_model_manager import ThauOwnModelManager
from thau_reasoning import ChainOfThought, TreeOfThoughts, TaskPlanner, SelfReflection


class IntegratedTHAUTrainer:
    """
    Entrenador integrado que combina todos los sistemas:
    - Desarrollo cognitivo por edades
    - Auto-generación de datasets
    - Memoria vectorizada para recuperación
    - Aprendizaje multilingüe
    - Entrenamiento automático
    """

    def __init__(self, auto_train_enabled: bool = True):
        print("🚀 Inicializando THAU Integrado...")

        # Componentes principales
        self.cognitive_manager = CognitiveDevelopmentManager()
        self.self_learning = SelfLearningManager(self.cognitive_manager)
        self.vector_memory = EfficientVectorMemory()
        self.language_manager = MultilingualLearningManager()
        self.data_manager = DataManager()
        self.self_questioning = SelfQuestioningSystem(
            max_questions_per_hour=10,
            max_questions_per_day=100
        )

        # Modelo LLM propio de THAU
        self.own_model = ThauOwnModelManager()
        self.own_model.initialize_model(cognitive_age=self.cognitive_manager.current_age)

        # Sistemas de razonamiento avanzado
        self.chain_of_thought = ChainOfThought(llm_client=self.own_model)
        self.tree_of_thoughts = TreeOfThoughts(llm_client=self.own_model)
        self.task_planner = TaskPlanner(llm_client=self.own_model)
        self.self_reflection = SelfReflection(llm_client=self.own_model)

        # Estado
        self.auto_train_enabled = auto_train_enabled
        self.stats = self._load_stats()

        # Background thread para auto-mejora
        self.improvement_thread = None
        self.running = False

        print("✅ THAU Integrado inicializado")
        self._print_status()

    def _load_stats(self) -> Dict:
        """Carga estadísticas generales"""
        stats_file = Path("./data/logs/integrated_stats.json")

        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)

        return {
            "total_interactions": 0,
            "total_trainings": 0,
            "age_progressions": 0,
            "datasets_generated": 0,
            "languages_learned": 1,  # español por defecto
            "memory_entries": 0,
            "start_date": datetime.now().isoformat()
        }

    def _save_stats(self):
        """Guarda estadísticas"""
        stats_file = Path("./data/logs/integrated_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def _print_status(self):
        """Muestra estado actual"""
        cognitive_status = self.cognitive_manager.get_status()
        memory_stats = self.vector_memory.get_stats()
        learning_stats = self.self_learning.get_stats()

        print("\n" + "="*70)
        print("📊 ESTADO DE THAU")
        print("="*70)
        print(f"\n🧠 Desarrollo Cognitivo:")
        print(f"   Edad: {cognitive_status['current_age']} años - {cognitive_status['stage_name']}")
        print(f"   Progreso: {cognitive_status['progress']['progress_pct']:.1f}%")
        print(f"   Ejemplos en edad: {cognitive_status['progress']['examples_at_age']}/{cognitive_status['progress']['examples_needed']}")
        print(f"   Puede avanzar: {'✅ Sí' if cognitive_status['progress']['can_advance'] else '❌ No'}")

        print(f"\n💾 Memoria Vectorizada:")
        print(f"   Vectores activos: {memory_stats['active_vectors']}")
        print(f"   Tamaño: {memory_stats['memory_size_mb']:.2f} MB")
        print(f"   Tipo de índice: {memory_stats['index_type']}")

        print(f"\n📚 Auto-Aprendizaje:")
        print(f"   Brechas detectadas: {learning_stats['total_gaps_detected']}")
        print(f"   Datasets generados: {learning_stats['total_datasets_generated']}")
        print(f"   Ejemplos generados: {learning_stats['total_examples_generated']}")

        print(f"\n🌍 Idiomas:")
        print(f"   Activos: {', '.join(self.language_manager.active_languages)}")

        print(f"\n📈 Estadísticas Generales:")
        print(f"   Interacciones totales: {self.stats['total_interactions']}")
        print(f"   Entrenamientos: {self.stats['total_trainings']}")
        print(f"   Progresiones de edad: {self.stats['age_progressions']}")

        print("="*70 + "\n")

    def process_interaction(
        self,
        question: str,
        answer: str,
        confidence: float = None,
        store_in_memory: bool = True,
        detect_language: bool = True
    ) -> Dict:
        """
        Procesa una interacción completa

        Returns:
            Diccionario con análisis y acciones tomadas
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "actions_taken": []
        }

        # 1. Detectar brechas de conocimiento
        gap = self.self_learning.process_interaction(question, answer, confidence)

        if gap:
            result["knowledge_gap_detected"] = True
            result["gap_topic"] = gap["topic"]
            result["actions_taken"].append("Gap de conocimiento registrado")
        else:
            result["knowledge_gap_detected"] = False

        # 2. Almacenar en memoria vectorizada
        if store_in_memory:
            memory_id = self.vector_memory.add(
                text=f"Q: {question}\nA: {answer}",
                metadata={
                    "question": question,
                    "answer": answer,
                    "confidence": confidence,
                    "age": self.cognitive_manager.current_age
                }
            )
            result["memory_id"] = memory_id
            result["actions_taken"].append("Almacenado en memoria vectorizada")
            self.stats["memory_entries"] += 1

        # 3. Detectar y aprender nuevo vocabulario (simplificado)
        if detect_language:
            # Extraer palabras nuevas
            words = set(question.lower().split() + answer.lower().split())

            for word in words:
                if len(word) > 4 and not self.language_manager.vocab_builder.get_word(word, "es"):
                    # Palabra potencialmente nueva
                    # En producción, aquí se haría validación más sofisticada
                    pass

        # 4. Añadir a cola de entrenamiento
        self.data_manager.add_example(
            instruction=question,
            output=answer,
            metadata={
                "confidence": confidence,
                "age": self.cognitive_manager.current_age,
                "auto_generated": False
            }
        )
        result["actions_taken"].append("Añadido a cola de entrenamiento")

        # Actualizar estadísticas
        self.stats["total_interactions"] += 1
        self._save_stats()

        return result

    def recall_from_memory(self, query: str, k: int = 3) -> List[Dict]:
        """
        Recupera información relevante de la memoria

        Returns:
            Lista de interacciones similares previas
        """
        results = self.vector_memory.search(query, k=k, min_score=0.5)

        return results

    def auto_improve_knowledge(self, min_gaps: int = 3):
        """
        Mejora automáticamente conocimiento detectando brechas
        y generando datasets
        """
        print("\n🔄 Ejecutando ciclo de auto-mejora...")

        # 1. Generar datasets para brechas
        generation_result = self.self_learning.auto_generate_missing_knowledge(min_gaps=min_gaps)

        if generation_result["generated"] > 0:
            self.stats["datasets_generated"] += generation_result["datasets_generated"]
            self._save_stats()

            print(f"✅ {generation_result['datasets_generated']} datasets generados")

            # 2. Importar datasets generados a cola de entrenamiento
            self._import_generated_datasets()

        return generation_result

    def _import_generated_datasets(self):
        """Importa datasets auto-generados a cola de entrenamiento"""
        generated_dir = Path("./data/datasets/auto_generated")

        if not generated_dir.exists():
            return

        # Buscar nuevos datasets
        for dataset_file in generated_dir.glob("*.jsonl"):
            # Leer y añadir ejemplos
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        example = json.loads(line)
                        self.data_manager.add_example(
                            instruction=example.get("instruction", ""),
                            output=example.get("output", ""),
                            input_context=example.get("input", ""),
                            metadata={
                                "auto_generated": True,
                                "source_file": dataset_file.name
                            }
                        )
                    except:
                        pass

            # Marcar como importado (renombrar)
            imported_file = dataset_file.parent / f"imported_{dataset_file.name}"
            dataset_file.rename(imported_file)

        print("✅ Datasets auto-generados importados")

    def check_and_advance_age(self) -> bool:
        """
        Verifica si puede avanzar de edad cognitiva
        y lo hace si es posible
        """
        if self.cognitive_manager.check_advancement():
            print("\n🎉 THAU puede avanzar de edad!")

            old_age = self.cognitive_manager.current_age
            advanced = self.cognitive_manager.advance_age()

            if advanced:
                self.stats["age_progressions"] += 1
                self._save_stats()

                new_age = self.cognitive_manager.current_age

                print(f"📈 Progresión: {old_age} años → {new_age} años")
                print(f"🎯 Nuevas capacidades: {', '.join(self.cognitive_manager.stage.capabilities)}")

                # CRUCIAL: Avanzar el modelo también
                print(f"\n🧠 Haciendo crecer el modelo propio...")
                self.own_model.advance_age(new_age)

                return True

        return False

    def train_now(self, force: bool = False) -> Dict:
        """
        Ejecuta entrenamiento inmediato

        Args:
            force: Forzar incluso sin ejemplos nuevos

        Returns:
            Resultado del entrenamiento
        """
        new_examples = self.data_manager.get_new_examples()

        if not new_examples and not force:
            return {"status": "skipped", "reason": "No new examples"}

        print(f"\n🎓 Iniciando entrenamiento REAL del modelo propio con {len(new_examples)} ejemplos...")

        # Obtener parámetros según edad cognitiva
        learning_params = self.cognitive_manager.get_learning_params()

        print(f"   Learning rate: {learning_params['learning_rate']}")
        print(f"   Context length: {learning_params['context_length']}")

        # Preparar textos para entrenar
        train_texts = []
        for example in new_examples:
            # Formato de entrenamiento: instrucción -> respuesta
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            input_context = example.get("input", "")

            if input_context:
                text = f"{instruction}\nContexto: {input_context}\nRespuesta: {output}"
            else:
                text = f"{instruction}\nRespuesta: {output}"

            train_texts.append(text)

        # Entrenar el modelo propio
        try:
            training_result = self.own_model.train_step(
                texts=train_texts,
                learning_rate=learning_params['learning_rate'],
                gradient_accumulation_steps=4
            )

            loss = training_result["loss"]
            perplexity = training_result["perplexity"]

            print(f"   Loss: {loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Tokens procesados: {training_result['tokens_processed']}")

            # Estimar accuracy desde perplexity (aproximación)
            # accuracy ~ 1 / (1 + log(perplexity))
            import math
            accuracy = 1.0 / (1.0 + math.log(max(perplexity, 1.1)))

        except Exception as e:
            print(f"⚠️  Error durante entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            accuracy = 0.0
            loss = float('inf')

        # Marcar ejemplos como entrenados
        self.data_manager.mark_as_trained(new_examples)

        # Registrar entrenamiento en cognitive manager
        self.cognitive_manager.record_training(len(new_examples), accuracy)

        self.stats["total_trainings"] += 1
        self._save_stats()

        # Guardar checkpoint del modelo cada ciertos entrenamientos
        if self.stats["total_trainings"] % 10 == 0:
            self.own_model.save_checkpoint()

        print(f"✅ Entrenamiento completado (accuracy estimada: {accuracy:.2%})")

        # Verificar si puede avanzar de edad
        self.check_and_advance_age()

        return {
            "status": "completed",
            "examples_trained": len(new_examples),
            "accuracy": accuracy,
            "age": self.cognitive_manager.current_age
        }

    def start_auto_improvement_loop(self, interval_hours: int = 6):
        """
        Inicia loop de auto-mejora en background

        Args:
            interval_hours: Horas entre ciclos de mejora
        """
        if self.running:
            print("⚠️  Loop de auto-mejora ya está corriendo")
            return

        self.running = True

        def improvement_loop():
            print(f"\n🔁 Loop de auto-mejora iniciado")
            print(f"   - Auto-cuestionamiento: cada 5 minutos")
            print(f"   - Generación de datasets: cada {interval_hours}h")
            print(f"   - Entrenamiento: cada {interval_hours}h\n")

            cycle_count = 0
            last_full_cycle = time.time()

            while self.running:
                try:
                    cycle_count += 1

                    # Auto-cuestionamiento cada 5 minutos (300 segundos)
                    time.sleep(300)

                    if not self.running:
                        break

                    print(f"\n🤔 Ciclo de auto-cuestionamiento #{cycle_count}")

                    # Ejecutar auto-cuestionamiento
                    qa_result = self.self_questioning.self_question_cycle(
                        cognitive_age=self.cognitive_manager.current_age
                    )

                    if qa_result:
                        # Procesar como interacción normal
                        self.process_interaction(
                            question=qa_result["question"],
                            answer=qa_result["answer"],
                            confidence=qa_result["confidence"],
                            store_in_memory=True
                        )
                        print(f"✅ Auto-aprendizaje completado (confianza: {qa_result['confidence']:.2f})")

                    # Ciclo completo cada N horas (generación de datasets y entrenamiento)
                    time_since_last = time.time() - last_full_cycle
                    if time_since_last >= (interval_hours * 3600):
                        print(f"\n⏰ Ejecutando ciclo completo de auto-mejora")

                        # 1. Auto-generar conocimiento para brechas
                        self.auto_improve_knowledge(min_gaps=5)

                        # 2. Entrenar si hay ejemplos nuevos
                        self.train_now()

                        # 3. Limpiar memoria si es necesario
                        memory_stats = self.vector_memory.get_stats()
                        if memory_stats["active_vectors"] > 10000:
                            self.vector_memory.cleanup(max_vectors=8000)
                            self.vector_memory.save()

                        last_full_cycle = time.time()

                except Exception as e:
                    print(f"❌ Error en loop de auto-mejora: {e}")
                    import traceback
                    traceback.print_exc()

        self.improvement_thread = threading.Thread(target=improvement_loop, daemon=True)
        self.improvement_thread.start()

    def stop_auto_improvement_loop(self):
        """Detiene loop de auto-mejora"""
        if self.running:
            self.running = False
            print("🛑 Loop de auto-mejora detenido")

    def save_all(self):
        """Guarda todo el estado"""
        self.vector_memory.save()
        self._save_stats()
        print("💾 Estado guardado")

    # ============== MÉTODOS DE RAZONAMIENTO AVANZADO ==============

    def answer_with_reasoning(self, question: str, max_steps: int = 5) -> Dict:
        """
        Responde una pregunta usando razonamiento Chain-of-Thought

        Args:
            question: Pregunta a responder
            max_steps: Máximo número de pasos de razonamiento

        Returns:
            Dict con pasos de razonamiento y respuesta
        """
        print(f"\n🧠 Razonando sobre: {question}")
        result = self.chain_of_thought.think_step_by_step(question, max_steps)

        # Almacenar en memoria
        self.process_interaction(
            question=question,
            answer=result["final_answer"],
            confidence=result["confidence"]
        )

        return result

    def explore_answers(self, question: str) -> Dict:
        """
        Explora múltiples caminos de razonamiento para una pregunta

        Args:
            question: Pregunta a analizar

        Returns:
            Dict con árbol de pensamientos y mejor respuesta
        """
        print(f"\n🌲 Explorando respuestas para: {question}")
        result = self.tree_of_thoughts.explore_thoughts(question)

        # Almacenar mejor respuesta
        self.process_interaction(
            question=question,
            answer=result["final_answer"],
            confidence=result["best_score"]
        )

        return result

    def plan_complex_task(self, goal: str, context: Optional[Dict] = None) -> Dict:
        """
        Planifica cómo lograr un objetivo complejo

        Args:
            goal: Objetivo a lograr
            context: Contexto adicional

        Returns:
            Plan estructurado con tareas
        """
        print(f"\n📋 Planificando: {goal}")
        plan = self.task_planner.create_plan(goal, context)
        return plan

    def improve_response(self, question: str, initial_response: str, criteria: Optional[List[str]] = None) -> Dict:
        """
        Mejora una respuesta usando auto-reflexión

        Args:
            question: Pregunta original
            initial_response: Respuesta inicial
            criteria: Criterios de evaluación

        Returns:
            Análisis y respuesta mejorada
        """
        print(f"\n🔍 Mejorando respuesta para: {question}")

        # Reflexionar
        reflection = self.self_reflection.reflect_on_response(question, initial_response, criteria)

        # Generar versión mejorada
        improved = self.self_reflection.improve_response(reflection)

        # Almacenar versión mejorada
        if reflection["overall_score"] >= 0.7:
            self.process_interaction(
                question=question,
                answer=improved,
                confidence=reflection["overall_score"]
            )

        return {
            "original": initial_response,
            "improved": improved,
            "reflection": reflection
        }

    def get_full_status(self) -> Dict:
        """Obtiene estado completo del sistema"""
        return {
            "cognitive": self.cognitive_manager.get_status(),
            "memory": self.vector_memory.get_stats(),
            "self_learning": self.self_learning.get_stats(),
            "self_questioning": self.self_questioning.get_stats(),
            "own_model": self.own_model.get_stats(),
            "reasoning": {
                "chain_of_thought": {
                    "history_size": len(self.chain_of_thought.reasoning_history),
                    "recent_confidence_avg": sum(
                        r["confidence"] for r in self.chain_of_thought.reasoning_history[-10:]
                    ) / len(self.chain_of_thought.reasoning_history[-10:])
                    if self.chain_of_thought.reasoning_history else 0.0
                },
                "tree_of_thoughts": {
                    "explorations": len(self.tree_of_thoughts.thought_trees)
                },
                "planning": {
                    "plans_created": len(self.task_planner.plans)
                },
                "reflection": {
                    "reflections_done": len(self.self_reflection.reflection_history),
                    "avg_improvement": sum(
                        r.get("overall_score", 0) for r in self.self_reflection.reflection_history[-10:]
                    ) / len(self.self_reflection.reflection_history[-10:])
                    if self.self_reflection.reflection_history else 0.0
                }
            },
            "general": self.stats,
            "auto_improvement_running": self.running
        }


# CLI para testing
if __name__ == "__main__":
    print("🤖 THAU - Sistema Integrado de Entrenamiento Autónomo\n")

    trainer = IntegratedTHAUTrainer(auto_train_enabled=True)

    # Simular algunas interacciones
    print("\n📝 Simulando interacciones...")

    interactions = [
        ("¿Qué es Python?", "Python es un lenguaje de programación interpretado de alto nivel", 0.9),
        ("¿Cómo funciona un bucle for?", "Un bucle for itera sobre una secuencia de elementos", 0.85),
        ("¿Qué es machine learning?", "No estoy muy seguro", 0.3),  # Brecha de conocimiento
    ]

    for question, answer, conf in interactions:
        result = trainer.process_interaction(question, answer, conf)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Acciones: {', '.join(result['actions_taken'])}")
        if result.get('knowledge_gap_detected'):
            print(f"⚠️  Brecha detectada en tópico: {result['gap_topic']}")

    # Auto-mejorar
    print("\n" + "="*70)
    trainer.auto_improve_knowledge(min_gaps=1)

    # Entrenar
    print("\n" + "="*70)
    trainer.train_now()

    # Recall de memoria
    print("\n" + "="*70)
    print("\n🔍 Buscando en memoria: 'programación'")
    results = trainer.recall_from_memory("programación", k=2)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Q: {result['question']}")
        print(f"   A: {result['answer'][:100]}...")

    # Estado final
    print("\n" + "="*70)
    trainer._print_status()

    # Guardar todo
    trainer.save_all()
