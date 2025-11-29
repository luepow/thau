#!/usr/bin/env python3
"""
Publica el modelo THAU en Ollama Registry

Pasos:
1. Crear cuenta en ollama.com
2. Crear Modelfile
3. ollama create luepow/thau -f Modelfile
4. ollama push luepow/thau

Uso:
    python scripts/publish_to_ollama.py --username luepow
"""

import argparse
import subprocess
import os
from pathlib import Path


def create_modelfile(gguf_path: str, output_path: str, model_name: str = "thau"):
    """Crea el Modelfile para Ollama"""

    modelfile_content = f'''# THAU - Self-Learning AI Model
# Modelo entrenado con sistema de edades cognitivas

FROM {gguf_path}

# Template de chat (formato TinyLlama/Llama)
TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}</s>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}</s>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}</s>
"""

# Parametros de generacion
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt por defecto
SYSTEM """Eres THAU, un asistente AI inteligente y servicial. Respondes en espanol de forma clara y concisa.

Tienes capacidad de usar herramientas cuando es necesario. Para usar una herramienta, responde con:
<tool_call>{{"name": "nombre_herramienta", "arguments": {{"param": "valor"}}}}</tool_call>

Herramientas disponibles:
- get_current_time: Obtiene la fecha y hora actual
- web_search: Busca informacion en internet
- execute_python: Ejecuta codigo Python
- generate_image: Genera una imagen a partir de un prompt

Si no necesitas herramientas, responde directamente al usuario."""

# Licencia
LICENSE """Apache 2.0 - THAU Project"""
'''

    with open(output_path, 'w') as f:
        f.write(modelfile_content)

    print(f"Modelfile creado: {output_path}")
    return output_path


def check_ollama():
    """Verifica que Ollama este instalado y corriendo"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print("Error: Ollama no esta respondiendo")
            return False
    except FileNotFoundError:
        print("Error: Ollama no esta instalado")
        print("Instala con: brew install ollama")
        return False


def create_local_model(modelfile_path: str, model_name: str):
    """Crea el modelo localmente"""
    print(f"\nCreando modelo local: {model_name}")

    cmd = ['ollama', 'create', model_name, '-f', modelfile_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Modelo creado exitosamente")
        return True
    else:
        print(f"  Error: {result.stderr}")
        return False


def push_to_registry(model_name: str):
    """Sube el modelo al registry de Ollama"""
    print(f"\nSubiendo modelo a ollama.com: {model_name}")
    print("  (Esto requiere estar autenticado con 'ollama login')")

    cmd = ['ollama', 'push', model_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Modelo subido exitosamente")
        return True
    else:
        print(f"  Error: {result.stderr}")
        if "unauthorized" in result.stderr.lower():
            print("\n  Necesitas autenticarte primero:")
            print("  1. Ve a https://ollama.com y crea una cuenta")
            print("  2. Ejecuta: ollama login")
        return False


def main():
    parser = argparse.ArgumentParser(description="Publicar THAU en Ollama")
    parser.add_argument("--username", type=str, required=True, help="Tu username de Ollama")
    parser.add_argument("--gguf", type=str, default="./export/models/thau-f16.gguf", help="Path al archivo GGUF")
    parser.add_argument("--model-name", type=str, default="thau", help="Nombre del modelo")
    parser.add_argument("--skip-push", action="store_true", help="Solo crear localmente, no subir")
    args = parser.parse_args()

    print("=" * 60)
    print("  THAU - Publicacion en Ollama")
    print("=" * 60)

    # Verificar Ollama
    if not check_ollama():
        return

    # Verificar GGUF
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"\nError: No se encontro el archivo GGUF: {gguf_path}")
        print("Ejecuta primero: python export_to_gguf.py")
        return

    print(f"\nArchivo GGUF: {gguf_path}")
    print(f"Tamano: {gguf_path.stat().st_size / 1e9:.2f} GB")

    # Crear Modelfile
    modelfile_path = Path("./Modelfile_ollama")
    create_modelfile(str(gguf_path.absolute()), str(modelfile_path), args.model_name)

    # Nombre completo del modelo
    full_model_name = f"{args.username}/{args.model_name}"

    # Crear modelo local
    if not create_local_model(str(modelfile_path), full_model_name):
        return

    # Verificar modelo
    print(f"\nVerificando modelo...")
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if full_model_name in result.stdout:
        print(f"  Modelo '{full_model_name}' disponible localmente")

    # Subir al registry
    if not args.skip_push:
        print("\n" + "-" * 60)
        print("IMPORTANTE: Para subir a ollama.com necesitas:")
        print("1. Crear cuenta en https://ollama.com")
        print("2. Ejecutar: ollama login")
        print("-" * 60)

        response = input("\nQuieres intentar subir ahora? (s/n): ")
        if response.lower() == 's':
            push_to_registry(full_model_name)

    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"\n  Modelo local: {full_model_name}")
    print(f"  Modelfile: {modelfile_path.absolute()}")
    print(f"\n  Para probar localmente:")
    print(f"    ollama run {full_model_name}")
    print(f"\n  Para subir manualmente:")
    print(f"    ollama login")
    print(f"    ollama push {full_model_name}")
    print(f"\n  URL despues de subir:")
    print(f"    https://ollama.com/{full_model_name}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
