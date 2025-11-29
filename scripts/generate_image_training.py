#!/usr/bin/env python3
"""
Genera datos de entrenamiento sobre generación de imágenes para THAU
Usa Gemini para crear ejemplos de alta calidad
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("./data/self_questioning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preguntas sobre generación de imágenes
IMAGE_GENERATION_TOPICS = [
    # Conceptos básicos
    ("que es Stable Diffusion?", "Explica que es Stable Diffusion y como funciona para generar imagenes"),
    ("como funciona un modelo de difusion?", "Explica el proceso de difusion en modelos generativos de imagenes"),
    ("que es un prompt en generacion de imagenes?", "Explica que es un prompt para generar imagenes y como escribir buenos prompts"),
    ("que es un negative prompt?", "Explica que es un negative prompt y para que sirve"),
    ("que es CFG scale o guidance scale?", "Explica el parametro CFG scale en generacion de imagenes"),
    ("que son los steps de inferencia?", "Explica que son los pasos de inferencia en Stable Diffusion"),
    ("que es un sampler en Stable Diffusion?", "Explica los diferentes samplers como DPM++, Euler, DDIM"),
    ("que es un VAE en generacion de imagenes?", "Explica el rol del VAE en Stable Diffusion"),
    ("que es el espacio latente?", "Explica que es el espacio latente en modelos de difusion"),
    ("que es CLIP en generacion de imagenes?", "Explica como CLIP conecta texto con imagenes"),

    # Modelos específicos
    ("que es SDXL?", "Explica que es Stable Diffusion XL y sus mejoras"),
    ("que es Midjourney?", "Explica que es Midjourney y como se usa"),
    ("que es DALL-E?", "Explica que es DALL-E de OpenAI"),
    ("diferencias entre SD, Midjourney y DALL-E", "Compara los principales modelos de generacion de imagenes"),
    ("que es ControlNet?", "Explica que es ControlNet y para que sirve"),
    ("que es LoRA en Stable Diffusion?", "Explica que son los modelos LoRA para fine-tuning"),
    ("que es un checkpoint en SD?", "Explica que es un modelo checkpoint"),

    # Técnicas avanzadas
    ("que es img2img?", "Explica la tecnica de imagen a imagen"),
    ("que es inpainting?", "Explica que es inpainting en generacion de imagenes"),
    ("que es outpainting?", "Explica que es outpainting y como funciona"),
    ("que es upscaling de imagenes?", "Explica tecnicas de mejora de resolucion"),
    ("como hacer prompt engineering para imagenes?", "Da consejos para escribir mejores prompts"),
    ("que es el seed en generacion?", "Explica que es el seed y como usarlo para reproducir imagenes"),

    # Herramientas
    ("que es Automatic1111?", "Explica que es la WebUI de Automatic1111"),
    ("que es ComfyUI?", "Explica que es ComfyUI y sus ventajas"),
    ("como usar la API de Stable Diffusion?", "Explica como conectar programaticamente con SD"),
    ("que es Pollinations.ai?", "Explica el servicio gratuito de generacion de imagenes"),
    ("que es Replicate para imagenes?", "Explica la plataforma Replicate"),

    # Aspectos técnicos
    ("cuanta VRAM necesita Stable Diffusion?", "Explica los requisitos de hardware para SD"),
    ("que es la precision fp16 vs fp32?", "Explica las diferencias de precision en modelos"),
    ("como optimizar la generacion de imagenes?", "Da consejos de optimizacion de memoria y velocidad"),
]

# Ejemplos de tool calling para generación de imágenes
TOOL_CALLING_EXAMPLES = [
    {"user": "genera una imagen de un gato", "prompt": "a cute cat, high quality"},
    {"user": "crea una imagen de un atardecer", "prompt": "beautiful sunset over the ocean, vibrant colors"},
    {"user": "dibuja un robot", "prompt": "a friendly robot, digital art style"},
    {"user": "hazme una imagen de un bosque magico", "prompt": "magical forest with glowing lights, fantasy art"},
    {"user": "genera una foto de una ciudad futurista", "prompt": "futuristic city skyline, neon lights, cyberpunk style"},
    {"user": "crea una ilustracion de un dragon", "prompt": "majestic dragon flying, epic fantasy art"},
    {"user": "dibuja un paisaje de montañas nevadas", "prompt": "snowy mountain landscape, majestic peaks, photorealistic"},
    {"user": "genera una imagen abstracta", "prompt": "abstract colorful art, geometric shapes, modern design"},
    {"user": "crea un retrato artistico", "prompt": "artistic portrait, oil painting style, dramatic lighting"},
    {"user": "hazme una imagen de un jardin japones", "prompt": "japanese garden with cherry blossoms, serene atmosphere"},
]

def query_gemini(prompt: str) -> str:
    """Consulta a Gemini CLI"""
    try:
        result = subprocess.run(
            ["gemini", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        print(f"Error llamando a Gemini: {e}")
        return ""

def generate_conceptual_training():
    """Genera datos de entrenamiento sobre conceptos de generación de imágenes"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n📚 Generando conocimiento conceptual sobre generación de imágenes...")
    print("=" * 60)

    for i, (question, gemini_prompt) in enumerate(IMAGE_GENERATION_TOPICS):
        print(f"\n[{i+1}/{len(IMAGE_GENERATION_TOPICS)}] {question}")

        # Consultar a Gemini
        full_prompt = f"{gemini_prompt}. Responde de forma clara y concisa en español, maximo 150 palabras."
        answer = query_gemini(full_prompt)

        if answer:
            print(f"   ✅ Respuesta obtenida ({len(answer)} chars)")

            entry = {
                "timestamp": timestamp,
                "question": question,
                "answer": answer,
                "confidence": 0.9,
                "cognitive_age": 12,
                "category": "image_generation",
                "self_generated": True,
                "source": "gemini"
            }
            examples.append(entry)
        else:
            print(f"   ❌ Sin respuesta")

        time.sleep(1)  # Rate limiting

    return examples

def generate_tool_calling_training():
    """Genera datos de entrenamiento para tool calling de imágenes"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n🔧 Generando ejemplos de tool calling para imágenes...")
    print("=" * 60)

    for item in TOOL_CALLING_EXAMPLES:
        tool_call = json.dumps({
            "name": "generate_image",
            "arguments": {"prompt": item["prompt"]}
        }, ensure_ascii=False)

        entry = {
            "timestamp": timestamp,
            "question": item["user"],
            "answer": f"<tool_call>{tool_call}</tool_call>",
            "confidence": 0.95,
            "cognitive_age": 12,
            "category": "tool_calling",
            "self_generated": True,
            "tool_training": True
        }
        examples.append(entry)
        print(f"   ✅ {item['user'][:40]}...")

    # Variaciones con prefijos
    prefixes = ["por favor ", "puedes ", "me gustaria que ", "necesito que "]
    for prefix in prefixes:
        for item in TOOL_CALLING_EXAMPLES[:5]:  # Solo primeros 5
            tool_call = json.dumps({
                "name": "generate_image",
                "arguments": {"prompt": item["prompt"]}
            }, ensure_ascii=False)

            entry = {
                "timestamp": timestamp,
                "question": prefix + item["user"],
                "answer": f"<tool_call>{tool_call}</tool_call>",
                "confidence": 0.95,
                "cognitive_age": 12,
                "category": "tool_calling",
                "self_generated": True,
                "tool_training": True
            }
            examples.append(entry)

    print(f"   Total: {len(examples)} ejemplos de tool calling")
    return examples

def generate_prompt_engineering_tips():
    """Genera consejos de prompt engineering usando Gemini"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n💡 Generando consejos de prompt engineering...")

    tips_prompts = [
        "Dame 5 consejos para escribir buenos prompts para Stable Diffusion",
        "Cuales son los mejores modificadores de estilo para prompts de imagenes (ej: 'digital art', '4k', etc)",
        "Como estructurar un prompt efectivo para generacion de imagenes",
        "Que palabras evitar en prompts negativos comunes",
        "Como especificar iluminacion y composicion en prompts",
    ]

    for prompt in tips_prompts:
        answer = query_gemini(f"{prompt}. Responde en español, maximo 200 palabras.")
        if answer:
            entry = {
                "timestamp": timestamp,
                "question": prompt,
                "answer": answer,
                "confidence": 0.9,
                "cognitive_age": 12,
                "category": "prompt_engineering",
                "self_generated": True,
                "source": "gemini"
            }
            examples.append(entry)
            print(f"   ✅ {prompt[:50]}...")
        time.sleep(1)

    return examples

def save_training_data(examples, filename="image_generation_training.jsonl"):
    """Guarda datos de entrenamiento"""
    output_path = OUTPUT_DIR / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\n💾 Guardados {len(examples)} ejemplos en {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("  GENERADOR DE DATOS - GENERACIÓN DE IMÁGENES")
    print("=" * 60)

    all_examples = []

    # 1. Conceptos básicos con Gemini
    conceptual = generate_conceptual_training()
    all_examples.extend(conceptual)

    # 2. Tool calling examples
    tool_calling = generate_tool_calling_training()
    all_examples.extend(tool_calling)

    # 3. Prompt engineering tips
    tips = generate_prompt_engineering_tips()
    all_examples.extend(tips)

    # Guardar
    save_training_data(all_examples)

    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Conceptuales: {len(conceptual)}")
    print(f"  Tool calling: {len(tool_calling)}")
    print(f"  Prompt tips:  {len(tips)}")
    print(f"  TOTAL:        {len(all_examples)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
