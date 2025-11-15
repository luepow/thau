#!/usr/bin/env python3
"""
CLI para THAU - Comandos para gestionar el modelo
"""

import click
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer import get_service, get_config, set_model_size, DataManager
import json


@click.group()
def cli():
    """THAU - Sistema de entrenamiento autónomo para modelos de IA"""
    pass


@cli.command()
@click.option('--size', type=click.Choice(['1.5b', '3b', '7b', '13b']), default='1.5b')
def init(size):
    """Inicializa THAU con un tamaño de modelo específico"""

    config = set_model_size(size)

    click.echo(f"🤖 Inicializando THAU-{size}")
    click.echo(f"Modelo base: {config.get_base_model()}")
    click.echo(f"Versión inicial: v1")
    click.echo("✅ Configuración guardada")


@cli.command()
def start():
    """Inicia el servicio de entrenamiento autónomo"""

    service = get_service()
    config = get_config()

    click.echo("=" * 80)
    click.echo(f"🤖 Iniciando THAU {config.get_model_identifier()}")
    click.echo("=" * 80)

    service.start()

    click.echo("\n✅ Servicio iniciado")
    click.echo(f"💡 El modelo se entrenará automáticamente cada {config.auto_train_interval_hours} horas")
    click.echo(f"📊 Mínimo de ejemplos para entrenar: {config.min_new_examples}")
    click.echo("\n🌐 Usa el API (puerto 8000) para agregar datos de entrenamiento")
    click.echo("   python api/thau_api.py")

    # Mantener vivo
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n\n🛑 Deteniendo servicio...")
        service.stop()


@cli.command()
def status():
    """Muestra el estado del modelo y servicio"""

    service = get_service()
    config = get_config()
    data_manager = DataManager()

    stats = service.get_stats()
    data_stats = data_manager.get_stats()

    click.echo("=" * 80)
    click.echo(f"🤖 THAU Status")
    click.echo("=" * 80)
    click.echo(f"\nModelo: {config.get_model_identifier()}")
    click.echo(f"Tamaño: {config.model_size}")
    click.echo(f"Versión: v{config.current_version}")
    click.echo(f"\nServicio: {'🟢 Running' if service.is_running else '🔴 Stopped'}")
    click.echo(f"Entrenamiento en progreso: {'Sí' if stats.get('training_in_progress') else 'No'}")
    click.echo(f"\nEjemplos pendientes: {data_stats['pending_examples']}")
    click.echo(f"Total entrenado: {data_stats['total_trained']}")
    click.echo(f"Entrenamientos completados: {stats['total_trainings']}")

    if stats.get('last_training'):
        click.echo(f"Último entrenamiento: {stats['last_training']}")

    click.echo("=" * 80)


@cli.command()
@click.argument('instruction')
@click.argument('output')
def add(instruction, output):
    """Agrega un ejemplo de entrenamiento"""

    data_manager = DataManager()

    result = data_manager.add_example(instruction, output)

    if result == "duplicate":
        click.echo("⚠️  Ejemplo duplicado, no se agregó")
    else:
        click.echo(f"✅ Ejemplo agregado: {result}")
        click.echo(f"📊 Pendientes: {len(data_manager.get_new_examples())}")


@cli.command()
@click.argument('file', type=click.Path(exists=True))
def import_data(file):
    """Importa ejemplos desde un archivo JSONL"""

    data_manager = DataManager()

    examples = []
    with open(file, 'r') as f:
        for line in f:
            try:
                ex = json.loads(line)
                examples.append(ex)
            except:
                continue

    results = data_manager.add_batch(examples)

    click.echo(f"✅ Agregados: {results['added']}")
    click.echo(f"⏭️  Duplicados: {results['duplicates']}")
    click.echo(f"❌ Errores: {results['errors']}")


@cli.command()
def train():
    """Fuerza un entrenamiento inmediato"""

    service = get_service()

    click.echo("⚡ Forzando entrenamiento...")
    service.force_train()


@cli.command()
def version():
    """Muestra la versión del modelo"""

    config = get_config()

    click.echo(f"{config.get_model_identifier()}")


if __name__ == '__main__':
    cli()
