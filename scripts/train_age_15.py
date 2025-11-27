#!/usr/bin/env python3
"""
THAU Age 15 Training - Adult Model (2B Parameters)
Entrena el modelo THAU a edad cognitiva 15 - MODELO ADULTO COMPLETO
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
    """Dataset avanzado para modelo adulto"""
    return [
        # Programacion avanzada
        "Python es un lenguaje de programacion interpretado de alto nivel con tipado dinamico y gestion automatica de memoria.",
        "La programacion orientada a objetos se basa en encapsulacion, herencia, polimorfismo y abstraccion.",
        "Los patrones de diseno como Singleton, Factory y Observer resuelven problemas comunes de arquitectura.",
        "Las funciones lambda son funciones anonimas de una sola expresion utiles para callbacks.",
        "Los generadores usan yield para crear iteradores sin almacenar toda la secuencia en memoria.",
        "La programacion asincrona con async/await permite operaciones no bloqueantes en I/O.",
        "Los decoradores modifican el comportamiento de funciones sin alterar su codigo.",
        "Las metaclases permiten modificar la creacion de clases en tiempo de ejecucion.",
        "Los context managers con with garantizan limpieza de recursos incluso con excepciones.",
        "Type hints mejoran la legibilidad y permiten analisis estatico del codigo.",

        # Algoritmos avanzados
        "La complejidad temporal Big-O describe el crecimiento asintotico del tiempo de ejecucion.",
        "QuickSort usa particionamiento con pivote para ordenar en O(n log n) promedio.",
        "MergeSort garantiza O(n log n) mediante divide y conquista pero requiere O(n) espacio.",
        "Los algoritmos de grafos como Dijkstra encuentran caminos minimos en grafos ponderados.",
        "La programacion dinamica descompone problemas en subproblemas solapados memorizando resultados.",
        "Los arboles B+ optimizan busquedas en disco con nodos de alto factor de ramificacion.",
        "Las tablas hash proveen acceso O(1) promedio mediante funciones de dispersion.",
        "El algoritmo A* combina heuristicas con busqueda de costo uniforme para pathfinding optimo.",
        "Los algoritmos de consenso como Raft mantienen consistencia en sistemas distribuidos.",

        # Machine Learning
        "El aprendizaje supervisado entrena modelos con datos etiquetados para prediccion.",
        "Las redes neuronales profundas aprenden representaciones jerarquicas de caracteristicas.",
        "El backpropagation calcula gradientes mediante la regla de la cadena para optimizacion.",
        "Los transformers usan atencion multi-cabeza para capturar dependencias a larga distancia.",
        "El fine-tuning adapta modelos preentrenados a tareas especificas con menos datos.",
        "La normalizacion por lotes estabiliza el entrenamiento normalizando activaciones.",
        "El dropout previene sobreajuste desactivando neuronas aleatoriamente durante entrenamiento.",
        "Los embeddings representan datos discretos como vectores densos en espacio continuo.",
        "La funcion de perdida cross-entropy mide la divergencia entre distribucion predicha y real.",
        "El descenso de gradiente estocastico actualiza pesos usando mini-batches.",

        # Matematicas
        "El calculo diferencial estudia tasas de cambio instantaneas mediante derivadas.",
        "La integral definida calcula areas y acumulaciones bajo curvas.",
        "El algebra lineal maneja espacios vectoriales, matrices y transformaciones lineales.",
        "Los autovalores y autovectores caracterizan transformaciones lineales.",
        "La probabilidad bayesiana actualiza creencias con nueva evidencia via teorema de Bayes.",
        "La optimizacion convexa garantiza optimos globales en funciones convexas.",
        "Las series de Fourier descomponen senales periodicas en suma de sinusoides.",
        "El teorema central del limite explica por que muchas distribuciones convergen a normal.",
        "Los numeros complejos extienden los reales con la unidad imaginaria i = sqrt(-1).",

        # Fisica
        "Las leyes de Newton describen movimiento: inercia, F=ma, accion-reaccion.",
        "La termodinamica estudia energia, calor, trabajo y entropia en sistemas.",
        "Las ecuaciones de Maxwell unifican electricidad, magnetismo y luz.",
        "La relatividad especial relaciona espacio y tiempo con la velocidad de la luz.",
        "La mecanica cuantica describe comportamiento probabilistico a escala atomica.",
        "El principio de incertidumbre limita precision simultanea de posicion y momento.",
        "La conservacion de energia establece que la energia total se mantiene constante.",
        "Las ondas electromagneticas transportan energia a la velocidad de la luz.",

        # Arquitectura de software
        "Los microservicios desacoplan funcionalidades en servicios independientes escalables.",
        "La arquitectura hexagonal separa dominio de infraestructura mediante puertos y adaptadores.",
        "Event sourcing almacena estado como secuencia de eventos inmutables.",
        "CQRS separa lecturas de escrituras para optimizar cada camino independientemente.",
        "Los message brokers como Kafka permiten comunicacion asincrona desacoplada.",
        "Los contenedores Docker empaquetan aplicaciones con sus dependencias.",
        "Kubernetes orquesta contenedores para escalado, recuperacion y despliegue.",
        "Las APIs REST usan HTTP con recursos, metodos y representaciones estandarizadas.",
        "GraphQL permite consultas flexibles definidas por el cliente.",
        "El circuit breaker previene cascada de fallos aislando servicios degradados.",

        # Bases de datos
        "Las transacciones ACID garantizan atomicidad, consistencia, aislamiento, durabilidad.",
        "Los indices B-tree aceleran busquedas manteniendo datos ordenados en disco.",
        "La normalizacion elimina redundancia dividiendo datos en tablas relacionadas.",
        "El sharding distribuye datos horizontalmente entre multiples servidores.",
        "La replicacion copia datos entre nodos para disponibilidad y lectura escalada.",
        "El teorema CAP establece que sistemas distribuidos no pueden garantizar todo.",
        "Los indices invertidos mapean terminos a documentos para busqueda de texto.",
        "Las bases de datos columnares optimizan consultas analiticas sobre columnas.",

        # Seguridad
        "La criptografia asimetrica usa pares de claves publica/privada.",
        "Los hash criptograficos producen digests fijos irreversibles de datos.",
        "JWT codifica claims firmados para autenticacion stateless.",
        "OAuth2 delega autorizacion sin compartir credenciales.",
        "El principio de minimo privilegio limita acceso a lo estrictamente necesario.",
        "La inyeccion SQL explota consultas mal construidas para ejecutar codigo.",
        "HTTPS usa TLS para cifrar comunicacion cliente-servidor.",
        "Los certificados X.509 verifican identidad mediante cadena de confianza.",
    ]


def train_age_15(steps: int = 1000, batch_size: int = 4):
    """Entrena THAU a edad 15 - Modelo adulto 2B"""
    print("="*80)
    print("THAU AGE 15 TRAINING - ADULT MODEL")
    print("~2B Parameters - FINAL COGNITIVE STAGE")
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
        print("Avanzando a Age 15 (Modelo Adulto)...")
        manager.advance_age(new_age=15)
    else:
        print("\nInicializando modelo Age 15 desde cero...")
        manager.initialize_model(cognitive_age=15)

    # Mostrar info del modelo
    stats = manager.get_stats()
    print(f"\nModelo Age 15 (ADULTO):")
    print(f"  Parametros: {stats['total_parameters']:,}")
    print(f"  Dimension: {stats['config']['d_model']}")
    print(f"  Capas: {stats['config']['n_layers']}")
    print(f"  Cabezas: {stats['config']['n_heads']}")

    # Parametros de entrenamiento para modelo grande
    lr = 2e-5  # Learning rate muy bajo para modelo 2B
    warmup_steps = 150

    print(f"\nEntrenamiento:")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup: {warmup_steps} steps")

    # Training loop
    print("\n" + "-"*80)
    print("INICIANDO ENTRENAMIENTO - MODELO ADULTO")
    print("-"*80 + "\n")

    start_time = time.time()
    total_loss = 0

    for step in range(steps):
        # Calcular learning rate con warmup
        if step < warmup_steps:
            current_lr = lr * (step + 1) / warmup_steps
        else:
            # Decay suave despues de warmup
            decay = 0.99 ** ((step - warmup_steps) / 100)
            current_lr = lr * decay

        # Seleccionar batch
        batch = random.sample(all_data, min(batch_size, len(all_data)))

        try:
            result = manager.train_step(
                texts=batch,
                learning_rate=current_lr,
                gradient_accumulation_steps=8  # Mas acumulacion para modelo grande
            )

            loss = result.get('loss', 0)
            total_loss += loss

            # Log cada 20 steps
            if (step + 1) % 20 == 0 or step == 0:
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

            # Guardar checkpoint cada 100 steps
            if (step + 1) % 100 == 0:
                ckpt_name = f"age_15_step_{step+1}"
                manager.save_checkpoint(checkpoint_name=ckpt_name)

        except Exception as e:
            print(f"Error en step {step+1}: {e}")
            continue

    # Final
    elapsed = time.time() - start_time
    final_avg_loss = total_loss / steps

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO - THAU ADULTO")
    print("="*80)
    print(f"  Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Loss promedio final: {final_avg_loss:.4f}")
    print(f"  Velocidad: {steps/elapsed:.2f} steps/s")

    # Guardar checkpoint final
    manager.save_checkpoint(checkpoint_name="age_15_final")

    # Guardar estadisticas
    stats_file = data_dir / "training_stats" / "age_15_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    final_stats = {
        "age": 15,
        "model_type": "ADULT_2B",
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
    print("TEST DE GENERACION - MODELO ADULTO")
    print("-"*80)

    test_prompts = [
        "El aprendizaje automatico",
        "Las redes neuronales",
        "La programacion funcional",
        "Los microservicios",
        "El calculo diferencial",
    ]

    for prompt in test_prompts:
        try:
            generated = manager.generate_text(prompt, max_new_tokens=50, temperature=0.7)
            print(f"\nPrompt: {prompt}")
            print(f"Generado: {generated[:150]}...")
        except Exception as e:
            print(f"Error generando para '{prompt}': {e}")

    print("\n" + "="*80)
    print("THAU AGE 15 - ADULT MODEL COMPLETE!")
    print("="*80)
    print("\nTHAU ha alcanzado madurez cognitiva completa.")
    print("Modelo: 2B parametros, 24 capas, 32 cabezas de atencion")

    return final_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train THAU to Age 15 (Adult Model)")
    parser.add_argument("--steps", type=int, default=800, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    train_age_15(steps=args.steps, batch_size=args.batch)
