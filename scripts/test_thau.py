"""Test THAU v2.0 with trained checkpoints."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def chat_with_thau():
    """Interactive chat with THAU using pipeline."""

    print("=" * 60)
    print("  THAU v3.0 - Asistente de Programacion")
    print("  Escribe 'salir' para terminar")
    print("=" * 60)

    # Cargar modelo con LoRA
    checkpoint = "data/checkpoints/incremental/specialized/thau_v3_20251202_211505"
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nCargando modelo: {base_model}")
    print(f"Checkpoint: {checkpoint}")

    # Cargar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Aplicar LoRA
    try:
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()  # Merge for faster inference
        print("Checkpoint LoRA aplicado!")
    except Exception as e:
        print(f"Usando modelo base: {e}")

    # Crear pipeline
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("\n" + "-" * 60)

    while True:
        try:
            user_input = input("\nTu: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("\nHasta luego!")
                break

            # Formato de mensajes
            messages = [
                {"role": "system", "content": "Eres THAU, un asistente experto en programacion, UX/UI y desarrollo web. Responde de forma clara y concisa en espanol."},
                {"role": "user", "content": user_input},
            ]

            # Generar respuesta
            outputs = pipe(
                messages,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
            )

            response = outputs[0]["generated_text"][-1]["content"]
            print(f"\nTHAU: {response}")

        except KeyboardInterrupt:
            print("\n\nHasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def quick_test():
    """Run quick tests on THAU using pipeline."""

    print("=" * 60)
    print("  THAU v3.0 - Test Rapido")
    print("=" * 60)

    # Cargar modelo con LoRA
    checkpoint = "data/checkpoints/incremental/specialized/thau_v3_20251202_211505"
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nCargando modelo: {base_model}")

    # Cargar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    # Aplicar LoRA
    try:
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        print("Checkpoint LoRA aplicado!")
    except Exception as e:
        print(f"Usando modelo base: {e}")

    # Crear pipeline
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Preguntas de prueba
    tests = [
        "Que es una lista en Python?",
        "Como se crea una funcion en Python?",
        "Explica que es un decorador en Python",
        "Que es SQL y para que sirve?",
        "Cual es la diferencia entre una clase y un objeto?",
    ]

    for i, question in enumerate(tests, 1):
        print(f"\n[Test {i}] {question}")
        print("-" * 50)

        messages = [
            {"role": "system", "content": "Eres THAU, un asistente experto en programacion. Responde de forma clara y concisa en espanol."},
            {"role": "user", "content": question},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )

        response = outputs[0]["generated_text"][-1]["content"]
        print(f"Respuesta: {response[:600]}")

    print("\n" + "=" * 60)
    print("Test completado!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test THAU v2.0")
    parser.add_argument("--chat", action="store_true", help="Modo chat interactivo")
    parser.add_argument("--test", action="store_true", help="Test rápido automático")
    args = parser.parse_args()

    if args.chat:
        chat_with_thau()
    elif args.test:
        quick_test()
    else:
        print("Uso:")
        print("  python scripts/test_thau.py --chat   # Chat interactivo")
        print("  python scripts/test_thau.py --test   # Test automático")
