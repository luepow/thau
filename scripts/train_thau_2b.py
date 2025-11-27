#!/usr/bin/env python3
"""
THAU-2B Progressive Training Script
Entrena el modelo THAU desde cero con crecimiento progresivo hasta 2B parámetros
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add thau_trainer to path
sys.path.insert(0, str(Path(__file__).parent))

from thau_trainer.own_model_manager import ThauOwnModelManager
from thau_trainer.self_questioning import SelfQuestioningSystem
from thau_trainer.self_learning import SelfLearningManager


class ProgressiveTrainer:
    """
    Entrenador progresivo de THAU
    Empieza con modelo pequeño y va creciendo gradualmente
    """

    def __init__(self):
        print("="*80)
        print("🚀 THAU-2B Progressive Training System")
        print("="*80)

        # Inicializar componentes
        self.model_manager = ThauOwnModelManager()
        self.self_learning = SelfLearningManager()
        self.self_questioning = SelfQuestioningSystem()

        # Configuración de entrenamiento
        self.age_milestones = [0, 1, 3, 6, 12, 15]  # Edades objetivo
        self.current_age = 0

        # Datos de bootstrap (conceptos fundamentales)
        self.bootstrap_data = self._create_bootstrap_dataset()

    def _create_bootstrap_dataset(self) -> list:
        """Crea dataset inicial para bootstrap del modelo"""
        return [
            # Nivel 0: Conceptos básicos
            "Hola",
            "Adiós",
            "Sí",
            "No",
            "Gracias",
            "Por favor",
            "Ayuda",
            "Python",
            "Código",
            "Variable",

            # Nivel 1: Frases simples
            "Hola, ¿cómo estás?",
            "Me llamo THAU",
            "Soy un modelo de lenguaje",
            "Estoy aprendiendo",
            "Python es un lenguaje de programación",
            "Una variable almacena datos",
            "Una función realiza una tarea",
            "El código se ejecuta línea por línea",

            # Nivel 2: Explicaciones básicas
            "Python es un lenguaje de programación interpretado y de alto nivel",
            "Las variables en Python se declaran sin especificar tipo",
            "Una función es un bloque de código reutilizable",
            "Los bucles permiten repetir código múltiples veces",
            "Las listas en Python almacenan colecciones de elementos",
            "Los diccionarios almacenan pares clave-valor",
            "Las clases definen objetos con propiedades y métodos",

            # Nivel 3: Conceptos intermedios
            "La programación orientada a objetos organiza el código en clases y objetos que representan entidades del mundo real",
            "Las API REST permiten la comunicación entre sistemas usando HTTP con métodos GET, POST, PUT y DELETE",
            "Una base de datos relacional organiza los datos en tablas con relaciones entre ellas",
            "El control de versiones con Git permite rastrear cambios en el código a lo largo del tiempo",
            "Los algoritmos de ordenamiento organizan datos según criterios específicos como QuickSort o MergeSort",
        ]

    def train_age_phase(self, target_age: int, num_steps: int = 100):
        """
        Entrena el modelo para una fase de edad específica

        Args:
            target_age: Edad objetivo
            num_steps: Número de pasos de entrenamiento
        """
        print(f"\n{'='*80}")
        print(f"📚 FASE DE ENTRENAMIENTO - Edad {target_age}")
        print(f"{'='*80}\n")

        # Inicializar o avanzar modelo
        if self.model_manager.model is None:
            self.model_manager.initialize_model(cognitive_age=target_age)
        elif target_age > self.current_age:
            self.model_manager.advance_age(target_age)

        self.current_age = target_age

        # Estadísticas de la fase
        phase_stats = {
            "age": target_age,
            "start_time": datetime.now().isoformat(),
            "steps_completed": 0,
            "avg_loss": 0.0,
            "questions_asked": 0,
            "datasets_generated": 0
        }

        # Training loop
        for step in range(num_steps):
            print(f"\n🎯 Step {step + 1}/{num_steps}")

            # 1. Entrenamiento con bootstrap data
            if step < num_steps // 2:  # Primera mitad usa bootstrap
                batch_size = min(4, len(self.bootstrap_data))
                batch = self.bootstrap_data[
                    (step * batch_size) % len(self.bootstrap_data):
                    ((step + 1) * batch_size) % len(self.bootstrap_data)
                ]

                print(f"   📖 Training on bootstrap data ({len(batch)} texts)")
            else:
                # 2. Auto-generación de preguntas
                print(f"   💭 Generating self-question...")
                qa_result = self.self_questioning.self_question_cycle(
                    cognitive_age=target_age
                )

                if qa_result:
                    # Usar pregunta y respuesta para entrenar
                    batch = [
                        qa_result['question'],
                        qa_result['answer']
                    ]
                    phase_stats["questions_asked"] += 1

                    # Detectar brechas de conocimiento
                    gap = self.self_learning.process_interaction(
                        qa_result['question'],
                        qa_result['answer'],
                        qa_result.get('confidence', 0.5)
                    )
                else:
                    # Fallback a bootstrap
                    batch = self.bootstrap_data[:2]

            # 3. Ejecutar paso de entrenamiento
            training_result = self.model_manager.train_step(
                texts=batch,
                learning_rate=1e-4 / (target_age + 1),  # LR decrece con edad
                gradient_accumulation_steps=2
            )

            phase_stats["avg_loss"] = (
                (phase_stats["avg_loss"] * step + training_result['loss']) / (step + 1)
            )
            phase_stats["steps_completed"] += 1

            print(f"   ✅ Loss: {training_result['loss']:.4f}, "
                  f"Perplexity: {training_result['perplexity']:.2f}")

            # 4. Guardar checkpoint cada 25 steps
            if (step + 1) % 25 == 0:
                self.model_manager.save_checkpoint()
                print(f"   💾 Checkpoint saved")

            # 5. Auto-generar datasets cada 50 steps
            if (step + 1) % 50 == 0:
                print(f"\n   🧠 Auto-generating missing knowledge...")
                gen_result = self.self_learning.auto_generate_missing_knowledge(
                    min_gaps=2
                )
                phase_stats["datasets_generated"] += gen_result.get('datasets_generated', 0)

            # Pequeña pausa entre steps
            time.sleep(1)

        # Guardar estadísticas de la fase
        phase_stats["end_time"] = datetime.now().isoformat()
        self._save_phase_stats(target_age, phase_stats)

        # Guardar checkpoint final de la fase
        self.model_manager.save_checkpoint(f"age_{target_age}_final")

        # Probar generación
        print(f"\n{'='*80}")
        print(f"🧪 TEST DE GENERACIÓN - Edad {target_age}")
        print(f"{'='*80}")

        test_prompts = [
            "Hola",
            "¿Qué es Python?",
            "Explica una variable"
        ]

        for prompt in test_prompts:
            generated = self.model_manager.generate_text(
                prompt=prompt,
                max_new_tokens=30,
                temperature=0.7
            )
            print(f"\n📝 Prompt: {prompt}")
            print(f"💬 Generated: {generated}")

        print(f"\n{'='*80}")
        print(f"✅ Fase {target_age} completada")
        print(f"{'='*80}\n")

        return phase_stats

    def _save_phase_stats(self, age: int, stats: dict):
        """Guarda estadísticas de la fase"""
        stats_dir = Path("./data/training_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats_file = stats_dir / f"age_{age}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n📊 Estadísticas guardadas: {stats_file}")

    def train_progressive(self, target_final_age: int = 15):
        """
        Entrena progresivamente hasta la edad objetivo final

        Args:
            target_final_age: Edad final objetivo (15 = THAU-2B)
        """
        print(f"\n🌱 Iniciando entrenamiento progresivo hasta edad {target_final_age}")

        # Filtrar milestones hasta edad objetivo
        milestones = [age for age in self.age_milestones if age <= target_final_age]

        all_stats = []

        for age in milestones:
            # Calcular steps según edad (más steps en edades avanzadas)
            steps = 50 + (age * 10)  # 50, 60, 80, 110, 170, 200 steps

            phase_stats = self.train_age_phase(age, num_steps=steps)
            all_stats.append(phase_stats)

        # Resumen final
        print(f"\n{'='*80}")
        print(f"🎉 ENTRENAMIENTO COMPLETADO")
        print(f"{'='*80}\n")

        print("📊 Resumen de todas las fases:\n")
        for stats in all_stats:
            print(f"  Edad {stats['age']}:")
            print(f"    Steps: {stats['steps_completed']}")
            print(f"    Avg Loss: {stats['avg_loss']:.4f}")
            print(f"    Preguntas auto-generadas: {stats['questions_asked']}")
            print(f"    Datasets generados: {stats['datasets_generated']}\n")

        # Mostrar estadísticas finales del modelo
        final_stats = self.model_manager.get_stats()
        print(f"\n🤖 Estadísticas finales del modelo:")
        print(json.dumps(final_stats, indent=2))

        return all_stats


def main():
    """Punto de entrada principal"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrena THAU progresivamente hasta 2B parámetros"
    )
    parser.add_argument(
        "--target-age",
        type=int,
        default=15,
        choices=[0, 1, 3, 6, 12, 15],
        help="Edad cognitiva objetivo (15 = THAU-2B con 2B parámetros)"
    )
    parser.add_argument(
        "--steps-per-age",
        type=int,
        help="Número de steps por fase (opcional, se calcula automáticamente)"
    )

    args = parser.parse_args()

    # Crear trainer
    trainer = ProgressiveTrainer()

    # Ejecutar entrenamiento progresivo
    trainer.train_progressive(target_final_age=args.target_age)

    print("\n✨ Entrenamiento finalizado exitosamente!")
    print("\n📁 Para usar el modelo entrenado:")
    print("   - Checkpoints: ./data/model_checkpoints/")
    print("   - Estadísticas: ./data/training_stats/")
    print("\n🚀 Próximos pasos:")
    print("   1. Exportar a GGUF: python export/export_to_gguf.py")
    print("   2. Importar a Ollama: ollama create thau-2b -f export/Modelfile-thau")
    print("   3. Continuar entrenamiento: python train_thau_2b.py --target-age 15")


if __name__ == "__main__":
    main()
