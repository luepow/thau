#!/usr/bin/env python3
"""
Script de prueba rápida del modelo entrenado
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def test_architecture_questions():
    """Prueba el modelo con preguntas de arquitectura"""

    print("=" * 80)
    print("🧪 Test del Modelo de Arquitectura de Software")
    print("=" * 80)

    # Cargar modelo entrenado
    model_path = "./data/checkpoints/architecture-expert"

    print(f"\n📂 Cargando modelo desde {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Crear pipeline de generación
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        print("✅ Modelo cargado correctamente\n")

        # Preguntas de prueba
        test_questions = [
            "¿Qué es el patrón Repository?",
            "Explica Clean Architecture",
            "¿Cuándo usar microservicios?",
            "¿Qué es SOLID?",
            "Explica el patrón Factory",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'─' * 80}")
            print(f"Pregunta {i}: {question}")
            print(f"{'─' * 80}")

            # Formatear prompt
            prompt = f"<|user|>\n{question}</s>\n<|assistant|>\n"

            # Generar respuesta
            response = generator(prompt, num_return_sequences=1)[0]["generated_text"]

            # Extraer solo la respuesta del asistente
            answer = response.split("<|assistant|>\n")[-1].replace("</s>", "").strip()

            print(f"\n{answer}\n")

        print("=" * 80)
        print("✅ Test completado exitosamente")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Asegúrate de haber entrenado el modelo primero:")
        print("   python scripts/train_architecture.py")


if __name__ == "__main__":
    test_architecture_questions()
