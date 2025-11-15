#!/usr/bin/env python3
"""
Script para iniciar THAU y sembrar datos iniciales
Luego lo deja corriendo para que aprenda automáticamente
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Datos de semilla - 20 interacciones de alta calidad
SEED_INTERACTIONS = [
    {
        "question": "¿Qué es Python?",
        "answer": "Python es un lenguaje de programación interpretado, de alto nivel y de propósito general. Fue creado por Guido van Rossum en 1991. Es conocido por su sintaxis clara y legible, lo que lo hace ideal para principiantes.",
        "confidence": 0.95
    },
    {
        "question": "¿Cómo funciona un bucle for en Python?",
        "answer": "Un bucle for en Python itera sobre una secuencia (lista, tupla, string, etc.). Sintaxis: for item in secuencia: <código>. Ejemplo: for i in range(5): print(i) imprimirá números del 0 al 4.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es Git?",
        "answer": "Git es un sistema de control de versiones distribuido, creado por Linus Torvalds. Permite rastrear cambios en código fuente, colaborar con otros desarrolladores, y mantener historial completo del proyecto.",
        "confidence": 0.95
    },
    {
        "question": "¿Qué es una API REST?",
        "answer": "API REST (Representational State Transfer) es un estilo arquitectónico para servicios web. Usa métodos HTTP (GET, POST, PUT, DELETE) para operaciones CRUD. Es stateless, cacheable y tiene interfaz uniforme.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es Docker?",
        "answer": "Docker es una plataforma de contenedorización que empaqueta aplicaciones y sus dependencias en contenedores ligeros y portables. Garantiza que la app funcione igual en cualquier ambiente: desarrollo, staging, producción.",
        "confidence": 0.92
    },
    {
        "question": "¿Qué es PostgreSQL?",
        "answer": "PostgreSQL es un sistema de gestión de bases de datos relacional (RDBMS) de código abierto. Es conocido por su robustez, extensibilidad y cumplimiento de estándares SQL. Soporta transacciones ACID, JSON, y funciones avanzadas.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es async/await en JavaScript?",
        "answer": "async/await es sintaxis para manejar operaciones asíncronas en JavaScript. async declara una función asíncrona que retorna una Promise. await pausa la ejecución hasta que la Promise se resuelva. Hace código asíncrono más legible.",
        "confidence": 0.88
    },
    {
        "question": "¿Qué son los microservicios?",
        "answer": "Microservicios es un estilo arquitectónico donde la aplicación se divide en servicios pequeños e independientes. Cada servicio tiene su propia base de datos, se despliega independientemente, y se comunica vía APIs. Mejora escalabilidad y mantenibilidad.",
        "confidence": 0.85
    },
    {
        "question": "¿Qué es JWT?",
        "answer": "JWT (JSON Web Token) es un estándar para transmitir información de forma segura entre partes como JSON. Se usa para autenticación. Tiene 3 partes: header, payload, signature. Es stateless y permite Single Sign-On.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es React?",
        "answer": "React es una biblioteca de JavaScript para construir interfaces de usuario, creada por Facebook. Usa componentes reutilizables, Virtual DOM para optimizar actualizaciones, y un flujo de datos unidireccional. Es declarativa y eficiente.",
        "confidence": 0.93
    },
    {
        "question": "¿Qué es Kubernetes?",
        "answer": "Kubernetes (K8s) es un sistema de orquestación de contenedores de código abierto. Automatiza despliegue, escalado y gestión de aplicaciones containerizadas. Agrupa contenedores en pods, maneja service discovery, load balancing y self-healing.",
        "confidence": 0.87
    },
    {
        "question": "¿Qué es CI/CD?",
        "answer": "CI/CD es Integración Continua y Despliegue Continuo. CI: integrar cambios frecuentemente con pruebas automáticas. CD: desplegar automáticamente a producción. Herramientas: Jenkins, GitHub Actions, GitLab CI. Reduce errores y acelera desarrollo.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es Clean Architecture?",
        "answer": "Clean Architecture es un patrón arquitectónico propuesto por Robert C. Martin. Separa el código en capas con dependencias que apuntan hacia adentro: Entidades, Casos de Uso, Interfaces, Frameworks. El core es independiente de frameworks y UI.",
        "confidence": 0.85
    },
    {
        "question": "¿Qué es el patrón Repository?",
        "answer": "Repository es un patrón de diseño que media entre el dominio y la capa de datos. Encapsula lógica de acceso a datos, provee interfaz de colección de objetos del dominio. Permite cambiar la persistencia sin afectar la lógica de negocio.",
        "confidence": 0.88
    },
    {
        "question": "¿Qué es GraphQL?",
        "answer": "GraphQL es un lenguaje de consulta para APIs, creado por Facebook. El cliente especifica exactamente qué datos necesita. Ventajas: evita over-fetching/under-fetching, un solo endpoint, tipado fuerte. Alternativa a REST.",
        "confidence": 0.86
    },
    {
        "question": "¿Qué es Redis?",
        "answer": "Redis es un almacén de datos en memoria (in-memory) de código abierto. Soporta estructuras: strings, hashes, lists, sets, sorted sets. Usado como cache, message broker, y session store. Muy rápido (microsegundos de latencia).",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es TDD?",
        "answer": "TDD (Test-Driven Development) es una metodología donde escribes tests antes que el código. Ciclo: Red (test falla), Green (código mínimo para pasar), Refactor. Beneficios: mejor diseño, menos bugs, documentación viva.",
        "confidence": 0.87
    },
    {
        "question": "¿Qué es SOLID?",
        "answer": "SOLID son 5 principios de diseño OOP: S-Single Responsibility, O-Open/Closed, L-Liskov Substitution, I-Interface Segregation, D-Dependency Inversion. Propuestos por Robert C. Martin. Hacen código más mantenible, escalable y testeable.",
        "confidence": 0.9
    },
    {
        "question": "¿Qué es NoSQL?",
        "answer": "NoSQL son bases de datos no relacionales. Tipos: Document (MongoDB), Key-Value (Redis), Column-family (Cassandra), Graph (Neo4j). Ventajas: escalabilidad horizontal, esquema flexible, alta performance. Usan consistencia eventual vs ACID.",
        "confidence": 0.85
    },
    {
        "question": "¿Qué es Serverless?",
        "answer": "Serverless es un modelo donde el proveedor cloud gestiona la infraestructura. Tú solo despliegas funciones (FaaS - Function as a Service). Ejemplos: AWS Lambda, Azure Functions. Paga por uso, escala automáticamente. Ideal para eventos y APIs.",
        "confidence": 0.88
    }
]

def wait_for_api(max_retries=30, delay=2):
    """Espera hasta que la API esté lista"""
    print("🔄 Esperando que THAU API esté lista...")

    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ THAU API está lista!")
                return True
        except:
            pass

        print(f"   Intento {i+1}/{max_retries}...")
        time.sleep(delay)

    return False

def seed_interactions():
    """Siembra interacciones iniciales"""
    print(f"\n📚 Sembrando {len(SEED_INTERACTIONS)} interacciones iniciales...\n")

    for i, interaction in enumerate(SEED_INTERACTIONS, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/interact",
                json=interaction,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ {i}/{len(SEED_INTERACTIONS)}: {interaction['question'][:50]}...")

                if result.get("knowledge_gap_detected"):
                    print(f"   ⚠️  Brecha detectada: {result.get('gap_topic')}")
            else:
                print(f"❌ {i}/{len(SEED_INTERACTIONS)}: Error {response.status_code}")

        except Exception as e:
            print(f"❌ {i}/{len(SEED_INTERACTIONS)}: Error - {e}")

        time.sleep(0.5)  # No saturar

    print("\n✅ Siembra completada!\n")

def check_status():
    """Verifica el estado de THAU"""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("\n" + "="*70)
            print("📊 ESTADO ACTUAL DE THAU")
            print("="*70)
            print(f"🧠 Edad cognitiva: {status['cognitive_age']} años")
            print(f"📝 Etapa: {status['stage_name']}")
            print(f"📈 Progreso: {status['progress_pct']:.1f}%")
            print(f"💬 Interacciones totales: {status['total_interactions']}")
            print(f"💾 Vectores en memoria: {status['memory_vectors']}")
            print(f"🔄 Auto-mejora activa: {'✅ Sí' if status['auto_learning_active'] else '❌ No'}")
            print(f"🌍 Idiomas: {', '.join(status['languages'])}")
            print("="*70 + "\n")
    except Exception as e:
        print(f"❌ Error obteniendo estado: {e}")

def trigger_auto_improve():
    """Dispara auto-mejora si hay brechas"""
    try:
        print("🔍 Verificando brechas de conocimiento...")
        response = requests.post(f"{BASE_URL}/auto-improve?min_gaps=1", timeout=60)

        if response.status_code == 200:
            result = response.json()
            if result.get("generated", 0) > 0:
                print(f"✅ Auto-mejora: {result['datasets_generated']} datasets generados!")
                print(f"   Total ejemplos: {result['examples_generated']}")
            else:
                print("ℹ️  No hay brechas suficientes para generar datasets aún")
        else:
            print(f"⚠️  Auto-mejora: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Error en auto-mejora: {e}")

def main():
    print("\n" + "="*70)
    print("🤖 THAU - Iniciando Sistema de Aprendizaje Autónomo")
    print("="*70 + "\n")

    # 1. Esperar que API esté lista
    if not wait_for_api():
        print("❌ No se pudo conectar con THAU API")
        print("Asegúrate de que el servidor esté corriendo:")
        print("  python api/thau_api_integrated.py")
        return

    # 2. Ver estado inicial
    print("\n📊 Estado inicial:")
    check_status()

    # 3. Sembrar interacciones
    seed_interactions()

    # 4. Ver estado después de siembra
    print("\n📊 Estado después de siembra:")
    check_status()

    # 5. Disparar auto-mejora
    print("\n🚀 Disparando primera auto-mejora...")
    trigger_auto_improve()

    # 6. Información final
    print("\n" + "="*70)
    print("✅ THAU está corriendo y aprendiendo!")
    print("="*70)
    print(f"\n📍 API: {BASE_URL}")
    print(f"📍 Documentación: {BASE_URL}/docs")
    print(f"📍 Estado: {BASE_URL}/status")
    print("\n🔄 THAU se auto-mejorará cada 6 horas automáticamente")
    print("💡 Puedes añadir más interacciones en cualquier momento")
    print("\n📝 Para ver progreso en tiempo real:")
    print(f"   watch -n 10 'curl -s {BASE_URL}/status | jq'")
    print("\n⏸️  Para detener: Ctrl+C en la terminal de la API\n")

if __name__ == "__main__":
    main()
