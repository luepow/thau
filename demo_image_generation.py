#!/usr/bin/env python3
"""
Demo: THAU with Image Generation
Shows how THAU can generate images based on text requests
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from capabilities.vision.image_generator import ThauImageGenerator
from capabilities.tools.tool_registry import get_tool_registry
import webbrowser


def demo_direct_generation():
    """Demo 1: Direct image generation"""
    print("="*80)
    print("🎨 DEMO 1: Generación Directa de Imágenes")
    print("="*80)

    generator = ThauImageGenerator()

    prompts = [
        "a cute robot learning to code, digital art, vibrant colors",
        "a futuristic cityscape at sunset, cyberpunk style",
        "an AI brain made of circuits and neurons, glowing, artistic",
    ]

    print(f"\n📋 Generando {len(prompts)} imágenes de demostración...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")

        result = generator.generate_image(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
        )

        if result['success']:
            print(f"✅ Imagen generada: {result['path']}")

            # Try to open in browser
            try:
                file_url = f"file://{result['path'].absolute()}"
                webbrowser.open(file_url)
                print(f"   🌐 Abriendo en navegador...")
            except:
                print(f"   💡 Abre manualmente: {result['path']}")
        else:
            print(f"❌ Error: {result.get('error')}")

    stats = generator.get_stats()
    print(f"\n📊 Estadísticas:")
    print(f"   Total generadas: {stats['total_images_generated']}")
    print(f"   Guardadas en: {stats['output_directory']}")


def demo_conversation_with_tools():
    """Demo 2: THAU with tool calling"""
    print("\n\n" + "="*80)
    print("💬 DEMO 2: THAU con Generación de Imágenes (Simulado)")
    print("="*80)

    # Get tool registry
    registry = get_tool_registry()

    # Simulate conversation
    conversations = [
        "Hola THAU, ¿cómo estás?",
        "Genera una imagen de un gato astronauta en el espacio",
        "¿Qué es Python?",
        "Crea una imagen de un paisaje de montañas nevadas",
        "Gracias!",
    ]

    print("\n🤖 Simulando conversación con THAU:\n")

    for user_input in conversations:
        print(f"\n👤 Usuario: {user_input}")

        # Detect if tool is needed
        tool = registry.detect_tool_needed(user_input)

        if tool:
            print(f"🔧 THAU detectó: {tool.name}")

            # Extract parameters
            params = registry.extract_parameters(user_input, tool)
            print(f"📝 Parámetros extraídos: {params}")

            # Generate image
            generator = ThauImageGenerator()
            result = generator.generate_image(**params)

            if result['success']:
                thau_response = f"¡Listo! He generado la imagen: {result['path']}"

                # Try to open
                try:
                    file_url = f"file://{result['path'].absolute()}"
                    webbrowser.open(file_url)
                    thau_response += " (Abriendo en navegador...)"
                except:
                    pass

                print(f"🤖 THAU: {thau_response}")
            else:
                print(f"🤖 THAU: Lo siento, hubo un error generando la imagen: {result.get('error')}")

        else:
            # Normal conversation (would use LLM here)
            print(f"🤖 THAU: [Respuesta normal del modelo de lenguaje]")


def demo_batch_generation():
    """Demo 3: Batch image generation"""
    print("\n\n" + "="*80)
    print("📦 DEMO 3: Generación por Lotes")
    print("="*80)

    generator = ThauImageGenerator()

    # Theme: Coding concepts as art
    coding_themes = [
        "recursion visualized as mirrors within mirrors, abstract art",
        "a binary tree growing like a real tree, nature meets code",
        "loops represented as infinity symbols in motion, digital art",
        "variables as colorful containers holding different shapes",
    ]

    print(f"\n🎨 Generando {len(coding_themes)} imágenes sobre conceptos de programación...\n")

    results = generator.generate_batch(
        prompts=coding_themes,
        num_inference_steps=20,  # Faster for demo
        guidance_scale=7.0,
    )

    successful = [r for r in results if r['success']]

    print(f"\n✅ Generación completada:")
    print(f"   Exitosas: {len(successful)}/{len(results)}")

    if successful:
        print(f"\n📁 Imágenes generadas:")
        for i, result in enumerate(successful, 1):
            print(f"   {i}. {result['path'].name}")


def main():
    """Run all demos"""
    import argparse

    parser = argparse.ArgumentParser(description="Demo de generación de imágenes con THAU")

    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3],
        help="Número de demo a ejecutar (1=Direct, 2=Conversation, 3=Batch, default=all)"
    )

    args = parser.parse_args()

    print("\n" + "🎨"*40)
    print("THAU - Image Generation Demo")
    print("🎨"*40 + "\n")

    if args.demo is None or args.demo == 1:
        demo_direct_generation()

    if args.demo is None or args.demo == 2:
        demo_conversation_with_tools()

    if args.demo is None or args.demo == 3:
        demo_batch_generation()

    print("\n\n" + "="*80)
    print("✅ Demos completados!")
    print("="*80)
    print("\n📁 Todas las imágenes guardadas en: ./data/generated_images/")
    print("\n💡 Próximos pasos:")
    print("   1. Revisar las imágenes generadas")
    print("   2. Integrar con la API de THAU")
    print("   3. Entrenar THAU para decidir cuándo generar imágenes")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
