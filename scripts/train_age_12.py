#!/usr/bin/env python3
"""
THAU Age 12 Training - Complete Model (~400M Parameters)
Entrena el modelo THAU a edad cognitiva 12 - Modelo completo
Este es el modelo maximo que cabe en memoria MPS sin errores OOM
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
                    question = entry.get('question', '')
                    answer = entry.get('answer', '')
                    if question and answer:
                        text = f"Pregunta: {question}\nRespuesta: {answer}"
                        qa_data.append(text)
                except json.JSONDecodeError:
                    continue

    print(f"Cargados {len(qa_data)} pares Q&A")
    return qa_data


def load_advanced_dataset() -> list:
    """Dataset avanzado para modelo completo"""
    return [
        # Programacion avanzada
        "Python es un lenguaje de programacion interpretado de alto nivel con tipado dinamico.",
        "La programacion orientada a objetos se basa en encapsulacion, herencia y polimorfismo.",
        "Los patrones de diseno como Singleton, Factory y Observer resuelven problemas de arquitectura.",
        "Los generadores usan yield para crear iteradores eficientes en memoria.",
        "La programacion asincrona con async/await permite operaciones no bloqueantes.",
        "Los decoradores modifican comportamiento de funciones sin alterar su codigo.",
        "Los context managers con with garantizan limpieza de recursos.",

        # Algoritmos
        "La complejidad Big-O describe el crecimiento asintotico del tiempo de ejecucion.",
        "QuickSort usa particionamiento con pivote para ordenar en O(n log n) promedio.",
        "Los algoritmos de grafos como Dijkstra encuentran caminos minimos.",
        "La programacion dinamica descompone problemas en subproblemas memorizando resultados.",
        "Las tablas hash proveen acceso O(1) promedio mediante funciones de dispersion.",

        # Machine Learning
        "El aprendizaje supervisado entrena modelos con datos etiquetados.",
        "Las redes neuronales profundas aprenden representaciones jerarquicas.",
        "El backpropagation calcula gradientes mediante la regla de la cadena.",
        "Los transformers usan atencion multi-cabeza para dependencias a larga distancia.",
        "El fine-tuning adapta modelos preentrenados a tareas especificas.",

        # Matematicas
        "El calculo diferencial estudia tasas de cambio mediante derivadas.",
        "La integral definida calcula areas bajo curvas.",
        "El algebra lineal maneja espacios vectoriales y transformaciones.",
        "La probabilidad bayesiana actualiza creencias con nueva evidencia.",

        # Fisica
        "Las leyes de Newton describen movimiento: inercia, F=ma, accion-reaccion.",
        "La termodinamica estudia energia, calor, trabajo y entropia.",
        "Las ecuaciones de Maxwell unifican electricidad, magnetismo y luz.",

        # Arquitectura
        "Los microservicios desacoplan funcionalidades en servicios independientes.",
        "Event sourcing almacena estado como secuencia de eventos inmutables.",
        "Los message brokers como Kafka permiten comunicacion asincrona.",
        "Kubernetes orquesta contenedores para escalado y recuperacion.",
        "Las APIs REST usan HTTP con recursos y metodos estandarizados.",

        # Bases de datos
        "Las transacciones ACID garantizan atomicidad, consistencia, aislamiento, durabilidad.",
        "Los indices B-tree aceleran busquedas manteniendo datos ordenados.",
        "El sharding distribuye datos horizontalmente entre servidores.",

        # Seguridad
        "La criptografia asimetrica usa pares de claves publica/privada.",
        "JWT codifica claims firmados para autenticacion stateless.",
        "OAuth2 delega autorizacion sin compartir credenciales.",
    ]


def train_age_12(steps: int = 600, batch_size: int = 4):
    """Entrena THAU a edad 12 - Modelo completo ~400M"""
    print("="*80)
    print("THAU AGE 12 TRAINING - COMPLETE MODEL")
    print("~400M Parameters - Maximum stable training")
    print("="*80)

    # Cargar datos
    data_dir = Path("./data")

    # Dataset combinado
    advanced_data = load_advanced_dataset()
    qa_data = load_all_qa_data(data_dir)

    # Combinar y mezclar
    all_data = advanced_data + qa_data
    random.shuffle(all_data)

    print(f"\nDataset total: {len(all_data)} ejemplos")
    print(f"  - Advanced: {len(advanced_data)}")
    print(f"  - Q&A: {len(qa_data)}")

    # Inicializar manager
    manager = ThauOwnModelManager()

    # Cargar checkpoint de Age 11 si existe
    age11_checkpoint = data_dir / "model_checkpoints" / "age_11_final.pt"
    if age11_checkpoint.exists():
        print(f"\nCargando checkpoint Age 11 para transferencia...")
        manager.load_checkpoint(age11_checkpoint)
        print("Avanzando a Age 12...")
        manager.advance_age(new_age=12)
    else:
        print("\nInicializando modelo Age 12 desde cero...")
        manager.initialize_model(cognitive_age=12)

    # Mostrar info del modelo
    stats = manager.get_stats()
    print(f"\nModelo Age 12 (COMPLETO):")
    print(f"  Parametros: {stats['total_parameters']:,}")
    print(f"  Dimension: {stats['config']['d_model']}")
    print(f"  Capas: {stats['config']['n_layers']}")
    print(f"  Cabezas: {stats['config']['n_heads']}")

    # Parametros de entrenamiento
    lr = 3e-5
    warmup_steps = 100

    print(f"\nEntrenamiento:")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup: {warmup_steps} steps")

    # Training loop
    print("\n" + "-"*80)
    print("INICIANDO ENTRENAMIENTO - MODELO COMPLETO")
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
                gradient_accumulation_steps=8
            )

            loss = result.get('loss', 0)
            total_loss += loss

            # Log cada 25 steps
            if (step + 1) % 25 == 0 or step == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                eta = (steps - step - 1) / speed if speed > 0 else 0

                print(f"Step {step+1:4d}/{steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Speed: {speed:.2f} s/s | "
                      f"ETA: {eta/60:.1f}min")

            # Guardar checkpoint cada 150 steps
            if (step + 1) % 150 == 0:
                ckpt_name = f"age_12_step_{step+1}"
                manager.save_checkpoint(checkpoint_name=ckpt_name)

        except Exception as e:
            print(f"Error en step {step+1}: {e}")
            continue

    # Final
    elapsed = time.time() - start_time
    final_avg_loss = total_loss / steps

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO - THAU COMPLETO")
    print("="*80)
    print(f"  Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Loss promedio final: {final_avg_loss:.4f}")
    print(f"  Velocidad: {steps/elapsed:.2f} steps/s")

    # Guardar checkpoint final
    manager.save_checkpoint(checkpoint_name="age_12_final")

    # Guardar estadisticas
    stats_file = data_dir / "training_stats" / "age_12_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    final_stats = {
        "age": 12,
        "model_type": "COMPLETE_400M",
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
    print("TEST DE GENERACION - MODELO COMPLETO")
    print("-"*80)

    test_prompts = [
        "Python es",
        "Las redes neuronales",
        "La derivada",
        "Los microservicios",
    ]

    for prompt in test_prompts:
        try:
            generated = manager.generate_text(prompt, max_new_tokens=40, temperature=0.7)
            print(f"\nPrompt: {prompt}")
            print(f"Generado: {generated[:120]}...")
        except Exception as e:
            print(f"Error generando para '{prompt}': {e}")

    print("\n" + "="*80)
    print("THAU AGE 12 - COMPLETE MODEL DONE!")
    print("="*80)

    return final_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train THAU to Age 12 (Complete Model)")
    parser.add_argument("--steps", type=int, default=600, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    train_age_12(steps=args.steps, batch_size=args.batch)
