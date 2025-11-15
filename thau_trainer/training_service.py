"""
Servicio de entrenamiento autónomo para THAU
Se ejecuta en background y entrena automáticamente cuando hay nuevos datos
"""

import time
import schedule
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import threading
import logging

from .config import get_config
from .trainer import ThauTrainer
from .data_manager import DataManager


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./data/logs/thau_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ThauService')


class ThauTrainingService:
    """Servicio de entrenamiento automático"""

    def __init__(self):
        self.config = get_config()
        self.data_manager = DataManager()
        self.trainer = ThauTrainer()
        self.is_running = False
        self.training_thread = None

        # Estadísticas
        self.stats = {
            "total_trainings": 0,
            "total_examples_trained": 0,
            "last_training": None,
            "current_version": self.config.current_version,
            "training_in_progress": False
        }

        self._load_stats()

    def _load_stats(self):
        """Carga estadísticas guardadas"""
        stats_file = Path("./data/logs/training_stats.json")
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats.update(json.load(f))
                logger.info(f"📊 Estadísticas cargadas: {self.stats}")

    def _save_stats(self):
        """Guarda estadísticas"""
        stats_file = Path("./data/logs/training_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)

    def check_and_train(self):
        """Verifica si hay datos nuevos y entrena si es necesario"""

        if self.stats["training_in_progress"]:
            logger.info("⏳ Entrenamiento ya en progreso, saltando...")
            return

        try:
            # Obtener nuevos ejemplos
            new_examples = self.data_manager.get_new_examples()

            if len(new_examples) < self.config.min_new_examples:
                logger.info(
                    f"📊 Sólo {len(new_examples)} nuevos ejemplos. "
                    f"Mínimo requerido: {self.config.min_new_examples}"
                )
                return

            logger.info(f"🎯 {len(new_examples)} nuevos ejemplos encontrados!")
            logger.info("🚀 Iniciando entrenamiento automático...")

            self.stats["training_in_progress"] = True
            self._save_stats()

            # Entrenar
            result = self.trainer.train(
                examples=new_examples,
                epochs=self.config.epochs_per_training,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate
            )

            # Incrementar versión
            new_version = self.config.increment_version()

            # Guardar nuevo modelo
            checkpoint_path = self.config.get_checkpoint_path()
            self.trainer.save(checkpoint_path)

            # Actualizar estadísticas
            self.stats["total_trainings"] += 1
            self.stats["total_examples_trained"] += len(new_examples)
            self.stats["last_training"] = datetime.now()
            self.stats["current_version"] = new_version
            self.stats["training_in_progress"] = False

            self._save_stats()

            # Marcar ejemplos como entrenados
            self.data_manager.mark_as_trained(new_examples)

            logger.info(f"✅ Entrenamiento completado!")
            logger.info(f"📦 Nueva versión: {self.config.get_model_identifier()}")
            logger.info(f"💾 Guardado en: {checkpoint_path}")

            # Actualizar modelo en Ollama
            self._update_ollama_model(checkpoint_path, new_version)

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento automático: {e}", exc_info=True)
            self.stats["training_in_progress"] = False
            self._save_stats()

    def _update_ollama_model(self, checkpoint_path: str, version: int):
        """Actualiza el modelo en Ollama"""
        try:
            import subprocess

            model_identifier = self.config.get_model_identifier()

            logger.info(f"🔄 Actualizando modelo en Ollama: {model_identifier}")

            # Crear Modelfile
            modelfile_content = f"""FROM {checkpoint_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER num_ctx 4096

SYSTEM \"\"\"Soy THAU ({model_identifier}), un asistente especializado en arquitectura de software,
razonamiento técnico y uso de herramientas. Puedo:
- Analizar y diseñar arquitecturas de software
- Razonar paso a paso sobre problemas complejos
- Usar herramientas para búsqueda web, ejecución de código y análisis
- Proporcionar respuestas técnicas precisas

Versión: {version}
Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}
\"\"\"
"""

            modelfile_path = Path(f"./Modelfile-thau-v{version}")
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Crear modelo en Ollama
            cmd = f"ollama create {model_identifier} -f {modelfile_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ Modelo {model_identifier} actualizado en Ollama")
            else:
                logger.error(f"❌ Error actualizando Ollama: {result.stderr}")

        except Exception as e:
            logger.error(f"❌ Error actualizando Ollama: {e}")

    def start(self):
        """Inicia el servicio de entrenamiento"""
        if self.is_running:
            logger.warning("⚠️  Servicio ya está corriendo")
            return

        self.is_running = True

        logger.info("=" * 80)
        logger.info("🤖 THAU Training Service - Iniciando")
        logger.info("=" * 80)
        logger.info(f"Modelo: {self.config.get_model_identifier()}")
        logger.info(f"Intervalo de entrenamiento: {self.config.auto_train_interval_hours} horas")
        logger.info(f"Mínimo de ejemplos: {self.config.min_new_examples}")
        logger.info(f"Auto-train: {self.config.auto_train_enabled}")
        logger.info("=" * 80)

        if self.config.auto_train_enabled:
            # Programar entrenamiento periódico
            schedule.every(self.config.auto_train_interval_hours).hours.do(self.check_and_train)

            # También permitir entrenamiento manual cada hora para chequear
            schedule.every(1).hours.do(self.check_and_train)

            # Ejecutar chequeo inicial después de 1 minuto
            schedule.every(1).minutes.do(self.check_and_train).tag('initial')

            logger.info("⏰ Scheduler configurado")

            # Thread del scheduler
            def run_scheduler():
                while self.is_running:
                    schedule.run_pending()
                    time.sleep(60)  # Chequear cada minuto

            self.training_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.training_thread.start()

            logger.info("✅ Servicio iniciado en background")
            logger.info("💡 Usa el API para agregar datos de entrenamiento")
        else:
            logger.info("⚠️  Auto-training deshabilitado")

    def stop(self):
        """Detiene el servicio"""
        logger.info("🛑 Deteniendo servicio...")
        self.is_running = False
        schedule.clear()
        if self.training_thread:
            self.training_thread.join(timeout=5)
        logger.info("✅ Servicio detenido")

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del servicio"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "model_identifier": self.config.get_model_identifier(),
            "new_examples_pending": len(self.data_manager.get_new_examples())
        }

    def force_train(self):
        """Fuerza un entrenamiento inmediato"""
        logger.info("⚡ Entrenamiento forzado solicitado")
        self.check_and_train()


# Singleton del servicio
_service = None


def get_service() -> ThauTrainingService:
    """Obtiene la instancia del servicio"""
    global _service
    if _service is None:
        _service = ThauTrainingService()
    return _service
