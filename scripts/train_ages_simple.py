#!/usr/bin/env python3
"""
THAU Simple Age Training - Train each age independently
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thau_trainer.own_model_manager import ThauOwnModelManager
import random


def train_specific_age(age: int, steps: int, dataset: list):
    """Train a specific age"""
    print(f"\n{'='*80}")
    print(f"🧠 TRAINING AGE {age}")
    print(f"{'='*80}\n")

    # Initialize model for this age
    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=age)

    # Select appropriate data
    if age == 0:
        training_data = dataset[:40]
        batch_size = 4
        lr = 5e-4
    elif age == 1:
        training_data = dataset[:60]
        batch_size = 6
        lr = 3e-4
    else:  # age 3
        training_data = dataset
        batch_size = 8
        lr = 1e-4

    print(f"📚 Examples: {len(training_data)}")
    print(f"📊 Steps: {steps}")
    print(f"🎯 Batch: {batch_size}")
    print(f"📈 LR: {lr}\n")

    start = time.time()

    for step in range(steps):
        batch = random.sample(training_data, min(batch_size, len(training_data)))

        try:
            result = manager.train_step(
                texts=batch,
                learning_rate=lr,
                gradient_accumulation_steps=4
            )

            if (step + 1) % 10 == 0 or step == 0:
                elapsed = time.time() - start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step+1}/{steps} | "
                      f"Loss: {result.get('loss', 0):.4f} | "
                      f"Tokens: {result.get('tokens_processed', 0)} | "
                      f"Speed: {speed:.2f} steps/s")

            if (step + 1) % 50 == 0 or step == steps - 1:
                ckpt = f"thau_age_{age}_step_{step+1}"
                manager.save_checkpoint(checkpoint_name=ckpt)
                print(f"💾 Saved: {ckpt}.pt")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    elapsed = time.time() - start
    print(f"\n✅ Completed in {elapsed:.1f}s ({steps/elapsed:.2f} steps/s)")

    # Save final
    final = f"age_{age}_final"
    manager.save_checkpoint(checkpoint_name=final)
    print(f"💾 Final: {final}.pt\n")


def main():
    # Dataset
    dataset = [
        # Basics
        "Hola", "Python", "Código", "Variable", "Función",
        # Simple phrases
        "Hola, ¿cómo estás?",
        "Python es un lenguaje de programación",
        "Una variable almacena datos",
        "Una función realiza una tarea",
        "Las clases definen objetos",
        "Los bucles repiten código",
        "Las condiciones evalúan expresiones",
        # Complete sentences
        "Python es un lenguaje interpretado y de alto nivel muy popular.",
        "Las variables en Python no requieren declaración de tipo.",
        "Una función es un bloque de código reutilizable.",
        "Los bucles permiten repetir código múltiples veces.",
        "Las listas almacenan colecciones ordenadas de elementos.",
        "Los diccionarios usan pares clave-valor para datos.",
        "Las clases son plantillas para crear objetos.",
        # Technical
        "La programación orientada a objetos organiza código en clases y objetos que representan entidades del mundo real con atributos y comportamientos específicos.",
        "Las APIs REST permiten comunicación entre sistemas usando HTTP con métodos GET, POST, PUT y DELETE para operaciones CRUD.",
        "Una base de datos relacional organiza datos en tablas con relaciones mediante claves primarias y foráneas.",
        "Git permite control de versiones rastreando cambios en código mediante commits, ramas y merges.",
        "Los algoritmos de ordenamiento como QuickSort organizan datos con diferentes características de complejidad.",
        # Q&A
        "Pregunta: ¿Qué es Python?\nRespuesta: Python es un lenguaje interpretado de alto nivel con sintaxis clara.",
        "Pregunta: ¿Cómo funciona una función?\nRespuesta: Una función toma parámetros, ejecuta operaciones y devuelve resultados.",
        # Code
        "Para crear una lista usa: mi_lista = [1, 2, 3]",
        "Define función con: def nombre(params): codigo",
        "Bucle for: for item in lista: print(item)",
        # Advanced
        "Los generadores usan yield para producir valores bajo demanda sin crear secuencias completas en memoria.",
        "El async/await permite programación asíncrona para operaciones I/O concurrentes sin bloquear.",
    ]

    print("="*80)
    print("🚀 THAU Simple Age Training")
    print("="*80)
    print(f"📊 Dataset: {len(dataset)} examples\n")

    # Train each age
    train_specific_age(age=0, steps=200, dataset=dataset)
    train_specific_age(age=1, steps=300, dataset=dataset)
    train_specific_age(age=3, steps=500, dataset=dataset)

    print("="*80)
    print("✅ ALL AGES TRAINED")
    print("="*80)
    print("\n📋 Next Steps:")
    print("1. Export to GGUF:")
    print("   python export_to_gguf.py --checkpoint data/model_checkpoints/age_3_final.pt --output thau_age_3.gguf --age 3")
    print("\n2. Create Ollama model:")
    print("   ollama create thau-age-3 -f thau_age_3.Modelfile")
    print("\n3. Test:")
    print("   ollama run thau-age-3 '¿Qué es Python?'\n")


if __name__ == "__main__":
    main()
