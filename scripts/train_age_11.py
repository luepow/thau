#!/usr/bin/env python3
"""
THAU Age 11 Training - Advanced Model Training
Entrena el modelo THAU a edad cognitiva 11 (usa config de 12 - ~400M params)
"""

import sys
import time
import json
import random
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from thau_trainer.own_model_manager import ThauOwnModelManager


def load_all_qa_data(data_dir: Path) -> list:
    """Carga todos los datos de Q&A de self-questioning"""
    qa_data = []

    qa_dir = data_dir / "self_questioning"
    if not qa_dir.exists():
        print(f"Directorio no encontrado: {qa_dir}")
        return qa_data

    for qa_file in sorted(qa_dir.glob("qa_*.jsonl")):
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Formatear como texto de entrenamiento
                    question = entry.get('question', '')
                    answer = entry.get('answer', '')
                    if question and answer:
                        # Formato Q&A para entrenamiento
                        text = f"Pregunta: {question}\nRespuesta: {answer}"
                        qa_data.append(text)
                except json.JSONDecodeError:
                    continue

    print(f"Cargados {len(qa_data)} pares Q&A")
    return qa_data


def load_base_dataset() -> list:
    """Dataset base de conocimientos fundamentales"""
    return [
        # Conceptos basicos
        "Python es un lenguaje de programacion interpretado de alto nivel.",
        "Una variable almacena datos en memoria durante la ejecucion.",
        "Las funciones son bloques de codigo reutilizable.",
        "Los bucles permiten repetir instrucciones.",
        "Las condiciones evaluan expresiones booleanas.",

        # Programacion avanzada
        "La programacion orientada a objetos organiza codigo en clases y objetos que encapsulan datos y comportamientos.",
        "Los patrones de diseno son soluciones probadas para problemas comunes en desarrollo de software.",
        "Las APIs REST usan HTTP para comunicacion entre sistemas con verbos GET, POST, PUT, DELETE.",
        "Git permite control de versiones mediante commits, ramas y merges.",
        "Docker containeriza aplicaciones para despliegue consistente.",

        # Estructuras de datos
        "Las listas almacenan colecciones ordenadas y son mutables.",
        "Los diccionarios usan pares clave-valor para acceso rapido O(1).",
        "Los conjuntos almacenan elementos unicos sin orden.",
        "Las pilas siguen LIFO: Last In First Out.",
        "Las colas siguen FIFO: First In First Out.",
        "Los arboles binarios tienen maximo dos hijos por nodo.",
        "Los grafos representan relaciones entre nodos mediante aristas.",

        # Algoritmos
        "La complejidad O(n) significa crecimiento lineal con el input.",
        "O(log n) es muy eficiente, como busqueda binaria.",
        "O(n^2) es cuadratica, comun en algoritmos de ordenamiento simples.",
        "QuickSort usa particionamiento con complejidad promedio O(n log n).",
        "BFS explora grafos por niveles usando cola.",
        "DFS explora grafos en profundidad usando pila o recursion.",

        # Bases de datos
        "SQL usa SELECT, INSERT, UPDATE, DELETE para manipular datos.",
        "Las relaciones se establecen con claves primarias y foraneas.",
        "Los indices aceleran consultas pero ralentizan escrituras.",
        "ACID garantiza atomicidad, consistencia, aislamiento, durabilidad.",
        "NoSQL ofrece flexibilidad de esquema y escalabilidad horizontal.",

        # Matematicas
        "La derivada mide la tasa de cambio instantanea de una funcion.",
        "La integral calcula el area bajo una curva.",
        "Una matriz es un arreglo bidimensional de numeros.",
        "Los vectores tienen magnitud y direccion.",
        "El teorema de Pitagoras: a^2 + b^2 = c^2 en triangulos rectangulos.",
        "El logaritmo es la operacion inversa de la potencia.",
        "Los numeros primos solo son divisibles por 1 y si mismos.",

        # Fisica
        "F = ma describe la segunda ley de Newton.",
        "La energia se conserva pero se transforma.",
        "La velocidad es desplazamiento sobre tiempo.",
        "La aceleracion es cambio de velocidad sobre tiempo.",
        "La gravedad acelera objetos a 9.8 m/s^2 en la Tierra.",

        # Logica
        "En logica, AND es verdadero solo si ambos operandos son verdaderos.",
        "OR es verdadero si al menos un operando es verdadero.",
        "NOT invierte el valor de verdad.",
        "Una tautologia es siempre verdadera.",
        "Una contradiccion es siempre falsa.",
        "Modus ponens: si P implica Q y P es verdadero, entonces Q es verdadero.",

        # IA/ML
        "El aprendizaje supervisado usa datos etiquetados.",
        "El aprendizaje no supervisado encuentra patrones sin etiquetas.",
        "Las redes neuronales aprenden representaciones en capas.",
        "El backpropagation ajusta pesos para minimizar error.",
        "Los transformers usan atencion para procesar secuencias.",

        # Arquitectura
        "Los microservicios dividen aplicaciones en servicios independientes.",
        "El patron MVC separa modelo, vista y controlador.",
        "SOLID son principios de diseno orientado a objetos.",
        "La escalabilidad horizontal agrega mas maquinas.",
        "El caching reduce latencia almacenando datos frecuentes.",
    ]


def train_age_11(steps: int = 1000, batch_size: int = 8):
    """Entrena THAU a edad 11"""
    print("="*80)
    print("THAU AGE 11 TRAINING")
    print("~400M Parameters Model")
    print("="*80)

    # Cargar datos
    data_dir = Path("./data")

    # Dataset combinado
    base_data = load_base_dataset()
    qa_data = load_all_qa_data(data_dir)

    # Combinar y mezclar
    all_data = base_data + qa_data
    random.shuffle(all_data)

    print(f"\nDataset total: {len(all_data)} ejemplos")
    print(f"  - Base: {len(base_data)}")
    print(f"  - Q&A: {len(qa_data)}")

    # Inicializar manager
    manager = ThauOwnModelManager()

    # Cargar checkpoint de Age 6 si existe (para transferir conocimiento)
    age6_checkpoint = data_dir / "model_checkpoints" / "age_6_final.pt"
    if age6_checkpoint.exists():
        print(f"\nCargando checkpoint Age 6 para transferencia...")
        manager.load_checkpoint(age6_checkpoint)
        print("Avanzando a Age 11...")
        manager.advance_age(new_age=11)
    else:
        print("\nInicializando modelo Age 11 desde cero...")
        manager.initialize_model(cognitive_age=11)

    # Mostrar info del modelo
    stats = manager.get_stats()
    print(f"\nModelo Age 11:")
    print(f"  Parametros: {stats['total_parameters']:,}")
    print(f"  Dimension: {stats['config']['d_model']}")
    print(f"  Capas: {stats['config']['n_layers']}")
    print(f"  Cabezas: {stats['config']['n_heads']}")

    # Parametros de entrenamiento para Age 11
    lr = 5e-5  # Learning rate mas bajo para modelo grande
    warmup_steps = 100

    print(f"\nEntrenamiento:")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup: {warmup_steps} steps")

    # Training loop
    print("\n" + "-"*80)
    print("INICIANDO ENTRENAMIENTO")
    print("-"*80 + "\n")

    start_time = time.time()
    total_loss = 0

    for step in range(steps):
        # Calcular learning rate con warmup
        if step < warmup_steps:
            current_lr = lr * (step + 1) / warmup_steps
        else:
            current_lr = lr

        # Seleccionar batch
        batch = random.sample(all_data, min(batch_size, len(all_data)))

        try:
            result = manager.train_step(
                texts=batch,
                learning_rate=current_lr,
                gradient_accumulation_steps=4
            )

            loss = result.get('loss', 0)
            total_loss += loss

            # Log cada 25 steps
            if (step + 1) % 25 == 0 or step == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                speed = (step + 1) / elapsed if elapsed > 0 else 0

                print(f"Step {step+1:4d}/{steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Speed: {speed:.2f} steps/s")

            # Guardar checkpoint cada 200 steps
            if (step + 1) % 200 == 0:
                ckpt_name = f"age_11_step_{step+1}"
                manager.save_checkpoint(checkpoint_name=ckpt_name)

        except Exception as e:
            print(f"Error en step {step+1}: {e}")
            continue

    # Final
    elapsed = time.time() - start_time
    final_avg_loss = total_loss / steps

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"  Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Loss promedio final: {final_avg_loss:.4f}")
    print(f"  Velocidad: {steps/elapsed:.2f} steps/s")

    # Guardar checkpoint final
    manager.save_checkpoint(checkpoint_name="age_11_final")

    # Guardar estadisticas
    stats_file = data_dir / "training_stats" / "age_11_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    final_stats = {
        "age": 11,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_loss": final_avg_loss,
        "training_time_seconds": elapsed,
        "dataset_size": len(all_data),
        "model_params": stats['total_parameters'],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)

    print(f"\nEstadisticas guardadas en: {stats_file}")

    # Test de generacion
    print("\n" + "-"*80)
    print("TEST DE GENERACION")
    print("-"*80)

    test_prompts = [
        "Python es",
        "Una funcion",
        "El aprendizaje automatico",
        "La derivada de",
    ]

    for prompt in test_prompts:
        try:
            generated = manager.generate_text(prompt, max_new_tokens=30, temperature=0.7)
            print(f"\nPrompt: {prompt}")
            print(f"Generado: {generated[:100]}...")
        except Exception as e:
            print(f"Error generando para '{prompt}': {e}")

    print("\n" + "="*80)
    print("Age 11 Training Complete!")
    print("="*80)

    return final_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train THAU to Age 11")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")

    args = parser.parse_args()

    train_age_11(steps=args.steps, batch_size=args.batch)
