#!/usr/bin/env python3
"""
Monitor THAU's learning progress over time
Displays periodic status updates
"""

import requests
import time
from datetime import datetime
import sys

BASE_URL = "http://localhost:8000"
CHECK_INTERVAL = 600  # 10 minutes

def get_status():
    """Get current THAU status"""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_self_learning_stats():
    """Get self-learning statistics"""
    try:
        response = requests.get(f"{BASE_URL}/stats/self-learning", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def format_status(status):
    """Format status for display"""
    if not status:
        return "❌ No se pudo obtener estado de THAU"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get additional stats
    self_learning = get_self_learning_stats()

    output = f"\n{'='*70}\n"
    output += f"📊 THAU STATUS - {timestamp}\n"
    output += f"{'='*70}\n"
    output += f"🧠 Edad Cognitiva: {status['cognitive_age']} años ({status['stage_name']})\n"
    output += f"📈 Progreso: {status['progress_pct']:.1f}%\n"
    output += f"💬 Interacciones: {status['total_interactions']}\n"
    output += f"💾 Vectores en Memoria: {status['memory_vectors']}\n"

    if self_learning:
        output += f"📚 Datasets Generados: {self_learning.get('total_datasets_generated', 0)}\n"
        output += f"💡 Brechas Detectadas: {self_learning.get('total_gaps_detected', 0)}\n"

    output += f"🔄 Auto-mejora: {'✅ Activa' if status['auto_learning_active'] else '❌ Inactiva'}\n"
    output += f"🌍 Idiomas: {', '.join(status['languages'])}\n"
    output += f"{'='*70}\n"

    return output

def main():
    print("\n🤖 Iniciando monitor de THAU...")
    print(f"⏱️  Revisando estado cada {CHECK_INTERVAL // 60} minutos")
    print("📍 Presiona Ctrl+C para detener\n")

    iteration = 0
    start_time = datetime.now()

    try:
        while True:
            iteration += 1
            elapsed = datetime.now() - start_time

            print(f"\n🔍 Iteración #{iteration} - Tiempo transcurrido: {elapsed}")

            status = get_status()
            print(format_status(status))

            if status:
                # Detectar cambios interesantes
                if status['progress_pct'] > 0:
                    print(f"🎯 ¡Progreso detectado! THAU está aprendiendo...")

                self_learning = get_self_learning_stats()
                if self_learning and self_learning.get('total_datasets_generated', 0) > 0:
                    print(f"📚 THAU ha generado {self_learning['total_datasets_generated']} datasets automáticamente")

                if status['cognitive_age'] > 0:
                    print(f"🚀 ¡THAU ha avanzado de edad! Ahora tiene {status['cognitive_age']} años")

            print(f"\n⏳ Próxima revisión en {CHECK_INTERVAL // 60} minutos...")
            print(f"💡 Para ver estado en vivo: curl -s {BASE_URL}/status | jq")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n✋ Monitor detenido por el usuario")
        final_status = get_status()
        print("\n📊 Estado final de THAU:")
        print(format_status(final_status))
        elapsed = datetime.now() - start_time
        print(f"\n⏱️  Tiempo total monitoreado: {elapsed}")
        print("\n✅ Monitor finalizado")

if __name__ == "__main__":
    main()
