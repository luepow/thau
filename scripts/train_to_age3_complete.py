#!/usr/bin/env python3
"""
THAU Training to Age 3 - Complete Pipeline
Este script lleva a THAU desde Age 0 hasta Age 3 con entrenamiento completo
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from thau_trainer.own_model_manager import ThauOwnModelManager
from thau_trainer.self_learning import SelfLearningManager
from thau_trainer.self_questioning import SelfQuestioningSystem
from thau_trainer.cognitive_development import CognitiveDevelopmentManager


class Age3TrainingPipeline:
    """
    Pipeline completo de entrenamiento hasta Age 3
    """

    def __init__(self, resume_from_checkpoint: bool = True):
        print("=" * 80)
        print("🚀 THAU Age 3 Complete Training Pipeline")
        print("=" * 80)
        print()

        # Inicializar componentes
        self.model_manager = ThauOwnModelManager()
        self.cognitive_manager = CognitiveDevelopmentManager()
        self.self_learning = SelfLearningManager(self.cognitive_manager)
        self.self_questioning = SelfQuestioningSystem()

        # Configuración de entrenamiento
        self.ages_to_train = [0, 1, 3]
        self.resume = resume_from_checkpoint

        # Dataset expandido para mejor entrenamiento
        self.training_data = self._create_comprehensive_dataset()

        print(f"📊 Dataset size: {len(self.training_data)} examples")
        print(f"🔄 Resume from checkpoint: {self.resume}")
        print()

    def _create_comprehensive_dataset(self) -> list:
        """Crea un dataset completo con múltiples niveles de complejidad"""

        data = []

        # === NIVEL 0: Conceptos básicos (1-3 palabras) ===
        basic_concepts = [
            "Hola", "Adiós", "Sí", "No", "Gracias", "Por favor",
            "Python", "Código", "Variable", "Función", "Clase",
            "Objeto", "Lista", "Diccionario", "Bucle", "Condición",
            "Archivo", "Módulo", "Paquete", "Biblioteca", "API",
            "Base de datos", "Servidor", "Cliente", "Red", "HTTP",
        ]
        data.extend(basic_concepts)

        # === NIVEL 1: Frases simples (5-10 palabras) ===
        simple_phrases = [
            "Hola, ¿cómo estás?",
            "Me llamo THAU",
            "Soy un modelo de lenguaje",
            "Estoy aprendiendo a programar",
            "Python es un lenguaje de programación",
            "Una variable almacena datos",
            "Una función realiza una tarea específica",
            "Las clases definen objetos",
            "Los bucles repiten código",
            "Las condiciones evalúan expresiones",
            "Los archivos almacenan información",
            "Los módulos organizan el código",
            "Las APIs conectan sistemas",
            "Las bases de datos guardan datos",
            "Los servidores procesan solicitudes",
        ]
        data.extend(simple_phrases)

        # === NIVEL 2: Oraciones completas (10-20 palabras) ===
        complete_sentences = [
            "Python es un lenguaje de programación interpretado y de alto nivel muy popular.",
            "Las variables en Python se declaran sin necesidad de especificar el tipo de dato.",
            "Una función es un bloque de código reutilizable que realiza una tarea específica.",
            "Los bucles permiten repetir un bloque de código múltiples veces de forma automática.",
            "Las listas en Python son colecciones ordenadas que pueden almacenar múltiples elementos.",
            "Los diccionarios almacenan pares clave-valor para acceso rápido a los datos.",
            "Las clases son plantillas que definen objetos con propiedades y métodos específicos.",
            "La herencia permite que una clase herede atributos y métodos de otra clase.",
            "Los decoradores en Python modifican el comportamiento de funciones o métodos existentes.",
            "Las excepciones permiten manejar errores de forma controlada durante la ejecución.",
        ]
        data.extend(complete_sentences)

        # === NIVEL 3: Explicaciones técnicas (20-40 palabras) ===
        technical_explanations = [
            "La programación orientada a objetos es un paradigma que organiza el código en clases y objetos que representan entidades del mundo real con sus atributos y comportamientos, facilitando la reutilización y el mantenimiento del código.",
            "Las API REST son interfaces de programación que permiten la comunicación entre sistemas mediante el protocolo HTTP, utilizando métodos estándar como GET, POST, PUT y DELETE para realizar operaciones CRUD sobre recursos.",
            "Una base de datos relacional organiza los datos en tablas con filas y columnas, donde las relaciones entre tablas se establecen mediante claves primarias y foráneas, garantizando la integridad referencial de la información.",
            "El control de versiones con Git permite rastrear cambios en el código fuente a lo largo del tiempo, facilitando la colaboración entre desarrolladores mediante ramas, commits y merges que documentan la evolución del proyecto.",
            "Los algoritmos de ordenamiento como QuickSort y MergeSort organizan colecciones de datos según criterios específicos, con diferentes características de complejidad temporal y espacial que los hacen más o menos adecuados según el contexto.",
            "La arquitectura de microservicios descompone una aplicación en servicios pequeños e independientes que se comunican entre sí mediante APIs, permitiendo escalabilidad, despliegue independiente y mayor resiliencia ante fallos.",
            "El machine learning es una rama de la inteligencia artificial que permite a los sistemas aprender patrones de los datos sin ser programados explícitamente, utilizando algoritmos que mejoran su rendimiento con la experiencia.",
            "Los contenedores Docker encapsulan aplicaciones con todas sus dependencias en unidades portables que se ejecutan de forma consistente en cualquier entorno, facilitando el desarrollo, testing y despliegue de software.",
            "El testing automatizado valida que el código funcione correctamente mediante pruebas unitarias, de integración y end-to-end, asegurando que los cambios no introduzcan regresiones en funcionalidades existentes.",
            "La seguridad en aplicaciones web requiere implementar múltiples capas de protección contra vulnerabilidades comunes como inyección SQL, XSS y CSRF, utilizando validación de entrada, sanitización de datos y autenticación robusta.",
        ]
        data.extend(technical_explanations)

        # === NIVEL 4: Conversaciones Q&A ===
        qa_pairs = [
            "Pregunta: ¿Qué es Python?\nRespuesta: Python es un lenguaje de programación interpretado, de alto nivel y propósito general, conocido por su sintaxis clara y legible.",
            "Pregunta: ¿Cómo funciona una función?\nRespuesta: Una función es un bloque de código que toma entradas (parámetros), ejecuta operaciones y puede devolver un resultado.",
            "Pregunta: ¿Para qué sirven las clases?\nRespuesta: Las clases son plantillas para crear objetos que encapsulan datos y comportamientos relacionados en una estructura cohesiva.",
            "Pregunta: ¿Qué es una API REST?\nRespuesta: Una API REST es una interfaz que permite la comunicación entre sistemas usando HTTP con métodos estándar y recursos identificados por URLs.",
            "Pregunta: ¿Cómo se manejan errores en Python?\nRespuesta: Los errores se manejan con bloques try-except que capturan excepciones y permiten ejecutar código alternativo cuando ocurren fallos.",
        ]
        data.extend(qa_pairs)

        # === NIVEL 5: Instrucciones de código ===
        code_instructions = [
            "Para crear una lista en Python usa corchetes: mi_lista = [1, 2, 3]",
            "Define una función con def nombre_funcion(parametros): y luego el código indentado.",
            "Los bucles for iteran sobre secuencias: for item in lista: print(item)",
            "Las condiciones if evalúan expresiones: if condicion: ejecutar_esto()",
            "Importa módulos con import nombre_modulo o from modulo import funcion",
        ]
        data.extend(code_instructions)

        # === NIVEL 6: Conceptos avanzados ===
        advanced_concepts = [
            "El pattern matching en Python 3.10+ permite comparar estructuras de datos complejas de forma declarativa mediante la expresión match-case, similar a switch pero más poderoso.",
            "Los generadores son funciones que usan yield para producir valores bajo demanda sin crear toda la secuencia en memoria, siendo eficientes para procesar grandes volúmenes de datos.",
            "Los context managers con with garantizan la correcta adquisición y liberación de recursos como archivos y conexiones, incluso cuando ocurren excepciones durante su uso.",
            "La metaprogramación permite que el código se modifique a sí mismo durante la ejecución usando metaclases, descriptores y decoradores para crear abstracciones poderosas.",
            "El async/await en Python permite programación asíncrona para operaciones I/O concurrentes sin bloquear el event loop, mejorando el rendimiento en aplicaciones de red.",
        ]
        data.extend(advanced_concepts)

        return data

    def train_age(self, age: int, steps: int, batch_size: int = 4):
        """
        Entrena el modelo para una edad específica

        Args:
            age: Edad cognitiva objetivo
            steps: Número de pasos de entrenamiento
            batch_size: Tamaño del batch
        """
        print(f"\n{'=' * 80}")
        print(f"🧠 TRAINING AGE {age}")
        print(f"{'=' * 80}\n")

        # Verificar checkpoint existente
        checkpoint_name = f"thau_age_{age}_step_50"

        if self.resume:
            checkpoint_file = Path(f"data/model_checkpoints/{checkpoint_name}.pt")
            if checkpoint_file.exists():
                print(f"📂 Loading existing checkpoint: {checkpoint_file}")
                try:
                    self.model_manager.load_checkpoint(checkpoint_name)
                    print(f"✅ Checkpoint loaded successfully")
                    return
                except Exception as e:
                    print(f"⚠️  Error loading checkpoint: {e}")
                    print(f"   Continuing with training from scratch...")

        # Inicializar modelo para esta edad
        print(f"🔧 Initializing model for age {age}...")
        self.model_manager.initialize_model(cognitive_age=age)

        # Seleccionar datos según complejidad de la edad
        if age == 0:
            # Age 0: Conceptos básicos + frases simples
            training_subset = self.training_data[:40]
            learning_rate = 5e-4
        elif age == 1:
            # Age 1: Todo hasta oraciones completas
            training_subset = self.training_data[:60]
            learning_rate = 3e-4
        else:  # age >= 3
            # Age 3+: Dataset completo
            training_subset = self.training_data
            learning_rate = 1e-4

        print(f"📚 Training examples: {len(training_subset)}")
        print(f"📊 Steps: {steps}")
        print(f"🎯 Batch size: {batch_size}")
        print(f"📈 Learning rate: {learning_rate}")
        print()

        # Entrenar
        start_time = time.time()

        for step in range(steps):
            # Seleccionar batch aleatorio
            import random
            batch = random.sample(training_subset, min(batch_size, len(training_subset)))

            # Entrenar
            try:
                result = self.model_manager.train_step(
                    texts=batch,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=4
                )

                # Mostrar progreso cada 10 steps
                if (step + 1) % 10 == 0 or step == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                    print(f"Step {step + 1}/{steps} | "
                          f"Loss: {result.get('loss', 0):.4f} | "
                          f"Tokens: {result.get('tokens_processed', 0)} | "
                          f"Speed: {steps_per_sec:.2f} steps/s")

                # Guardar checkpoint cada 50 steps
                if (step + 1) % 50 == 0 or step == steps - 1:
                    checkpoint_name = f"thau_age_{age}_step_{step + 1}"
                    self.model_manager.save_checkpoint(
                        checkpoint_name=checkpoint_name
                    )
                    print(f"💾 Checkpoint saved: {checkpoint_name}.pt")

            except Exception as e:
                print(f"❌ Error at step {step + 1}: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed:.2f}s")
        print(f"   Average: {steps / elapsed:.2f} steps/s")

        # Guardar checkpoint final
        final_checkpoint_name = f"age_{age}_final"
        self.model_manager.save_checkpoint(checkpoint_name=final_checkpoint_name)
        print(f"💾 Final checkpoint saved: {final_checkpoint_name}.pt\n")

    def advance_to_age(self, target_age: int):
        """Avanza el modelo a una nueva edad cognitiva"""
        print(f"\n{'=' * 80}")
        print(f"🔄 ADVANCING TO AGE {target_age}")
        print(f"{'=' * 80}\n")

        current_age = self.model_manager.current_age
        print(f"Current age: {current_age} → Target age: {target_age}")

        if target_age <= current_age:
            print(f"⚠️  Target age must be higher than current age")
            return

        # Avanzar edad
        try:
            self.model_manager.advance_age(target_age)
            print(f"✅ Successfully advanced to age {target_age}")
            success = True
        except Exception as e:
            print(f"❌ Error advancing to age {target_age}: {e}")
            success = False

        if success:

            # Guardar estado
            checkpoint_name = f"age_{target_age}_initialized"
            self.model_manager.save_checkpoint(
                checkpoint_name=checkpoint_name
            )
            print(f"💾 Checkpoint saved: {checkpoint_name}.pt\n")
        else:
            print(f"❌ Failed to advance to age {target_age}\n")

    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo hasta Age 3"""
        print(f"\n{'=' * 80}")
        print(f"🚀 STARTING COMPLETE TRAINING PIPELINE")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        # === AGE 0: Foundation ===
        print("🎯 PHASE 1: Age 0 Foundation")
        print("   Teaching basic concepts and simple phrases\n")
        self.train_age(age=0, steps=200, batch_size=4)

        # Auto-questioning para Age 0
        print("\n🤔 Running self-questioning for Age 0...")
        for i in range(5):
            question = self.self_questioning.generate_question(cognitive_age=0)
            if question:
                print(f"  Q{i+1}: {question}")

        # === AGE 1: Growth ===
        print("\n🎯 PHASE 2: Age 1 Growth")
        print("   Expanding to complete sentences\n")
        self.advance_to_age(target_age=1)
        self.train_age(age=1, steps=300, batch_size=6)

        # Auto-questioning para Age 1
        print("\n🤔 Running self-questioning for Age 1...")
        for i in range(5):
            question = self.self_questioning.generate_question(cognitive_age=1)
            if question:
                print(f"  Q{i+1}: {question}")

        # === AGE 3: Maturity ===
        print("\n🎯 PHASE 3: Age 3 Maturity")
        print("   Mastering technical explanations and complex concepts\n")
        self.advance_to_age(target_age=3)
        self.train_age(age=3, steps=500, batch_size=8)

        # Auto-questioning para Age 3
        print("\n🤔 Running self-questioning for Age 3...")
        for i in range(10):
            question = self.self_questioning.generate_question(cognitive_age=3)
            if question:
                print(f"  Q{i+1}: {question}")

        # === FINAL REPORT ===
        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"✅ PIPELINE COMPLETED")
        print(f"{'=' * 80}\n")
        print(f"⏱️  Total time: {elapsed / 60:.1f} minutes")
        print(f"🧠 Final age: {self.model_manager.current_age}")
        print(f"📊 Total training steps: 1000")
        print(f"💾 Checkpoints saved in: data/model_checkpoints/")
        print()
        print(f"{'=' * 80}")
        print(f"📋 NEXT STEPS")
        print(f"{'=' * 80}\n")
        print(f"1. Evaluate model quality:")
        print(f"   python evaluate_model.py --checkpoint data/model_checkpoints/age_3_final.pt")
        print()
        print(f"2. Export to GGUF for Ollama:")
        print(f"   python export_to_gguf.py --checkpoint data/model_checkpoints/age_3_final.pt --output thau_age_3.gguf --age 3")
        print()
        print(f"3. Import to Ollama:")
        print(f"   ollama create thau-age-3 -f thau_age_3.Modelfile")
        print()
        print(f"4. Test in Ollama:")
        print(f"   ollama run thau-age-3 '¿Qué es Python?'")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Train THAU to Age 3 with complete pipeline'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from existing checkpoints'
    )
    parser.add_argument(
        '--age-only',
        type=int,
        choices=[0, 1, 3],
        help='Train only specific age (skip pipeline)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps for single age training'
    )

    args = parser.parse_args()

    # Inicializar pipeline
    pipeline = Age3TrainingPipeline(resume_from_checkpoint=not args.no_resume)

    # Entrenar solo una edad específica
    if args.age_only is not None:
        steps_map = {0: 200, 1: 300, 3: 500}
        steps = args.steps if args.steps else steps_map[args.age_only]

        print(f"🎯 Training only Age {args.age_only} for {steps} steps")
        pipeline.train_age(age=args.age_only, steps=steps)
    else:
        # Pipeline completo
        pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
