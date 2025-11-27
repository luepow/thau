#!/usr/bin/env python3
"""
THAU Chat with Tool Calling
Integrates THAU in Ollama with image generation API

Usage:
    python thau_chat.py                    # Interactive chat
    python thau_chat.py --message "..."    # Single message
"""

import re
import json
import argparse
import subprocess
from typing import Dict, Optional, Tuple
import requests
from pathlib import Path


class ThauChat:
    """Chat interface for THAU with tool calling capabilities"""

    def __init__(
        self,
        ollama_model: str = "thau:latest",
        api_base_url: str = "http://localhost:8000",
    ):
        self.ollama_model = ollama_model
        self.api_base_url = api_base_url
        self.conversation_history = []

        print("🤖 THAU Chat iniciado")
        print(f"   Modelo: {ollama_model}")
        print(f"   API: {api_base_url}")
        print()

    def _call_ollama(self, prompt: str) -> str:
        """
        Call THAU via Ollama

        Args:
            prompt: User prompt

        Returns:
            THAU's response
        """
        try:
            # Build conversation context
            context = "\n".join([
                f"Usuario: {msg['user']}\nAsistente: {msg['assistant']}"
                for msg in self.conversation_history[-3:]  # Last 3 exchanges
            ])

            if context:
                full_prompt = f"{context}\n\nUsuario: {prompt}\nAsistente:"
            else:
                full_prompt = f"Usuario: {prompt}\nAsistente:"

            # Call Ollama
            result = subprocess.run(
                ["ollama", "run", self.ollama_model, full_prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                print(f"❌ Error llamando a Ollama: {result.stderr}")
                return "Lo siento, hubo un error procesando tu mensaje."

            response = result.stdout.strip()
            return response

        except subprocess.TimeoutExpired:
            print("⏱️ Timeout esperando respuesta de Ollama")
            return "Lo siento, la generación tomó demasiado tiempo."
        except FileNotFoundError:
            print("❌ Ollama no encontrado. ¿Está instalado?")
            return "Error: Ollama no está disponible."
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return "Lo siento, ocurrió un error inesperado."

    def _parse_tool_call(self, response: str) -> Tuple[str, Optional[Dict]]:
        """
        Parse tool call from THAU's response

        Args:
            response: THAU's raw response

        Returns:
            Tuple of (clean_response, tool_params)
            where tool_params is None if no tool was called
        """
        # Pattern: <TOOL:generate_image>{"prompt": "..."}</TOOL>
        pattern = r'<TOOL:generate_image>(.*?)</TOOL>'

        match = re.search(pattern, response, re.DOTALL)

        if not match:
            # No tool call
            return response, None

        # Extract tool parameters
        try:
            tool_json = match.group(1).strip()
            tool_params = json.loads(tool_json)

            # Remove tool call from response
            clean_response = re.sub(pattern, '', response, flags=re.DOTALL).strip()

            return clean_response, tool_params

        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando tool call JSON: {e}")
            return response, None

    def _generate_image(self, params: Dict) -> Optional[Dict]:
        """
        Call the vision API to generate an image

        Args:
            params: Tool parameters with 'prompt' key

        Returns:
            API response or None on error
        """
        try:
            url = f"{self.api_base_url}/vision/generate"

            # Prepare request
            payload = {
                "prompt": params.get("prompt", ""),
                "num_inference_steps": params.get("num_inference_steps", 30),
                "guidance_scale": params.get("guidance_scale", 7.5),
                "width": params.get("width", 512),
                "height": params.get("height", 512),
            }

            print(f"\n🎨 Generando imagen: '{payload['prompt'][:60]}...'")

            # Call API
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            result = response.json()

            if result.get("success"):
                print(f"✅ Imagen generada: {result.get('image_url')}")
                return result
            else:
                print(f"❌ Error generando imagen: {result.get('error')}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"❌ No se puede conectar a la API en {self.api_base_url}")
            print("   Asegúrate de que la API esté corriendo: python api/main.py")
            return None
        except requests.exceptions.Timeout:
            print("⏱️ Timeout esperando generación de imagen")
            return None
        except Exception as e:
            print(f"❌ Error llamando a vision API: {e}")
            return None

    def _open_image(self, image_path: str):
        """Open image in default viewer"""
        try:
            import platform
            system = platform.system()

            if system == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", image_path])
            elif system == "Windows":
                subprocess.run(["start", image_path], shell=True)

        except Exception as e:
            print(f"⚠️ No se pudo abrir la imagen automáticamente: {e}")

    def send_message(self, user_message: str) -> str:
        """
        Send a message to THAU and handle tool calling

        Args:
            user_message: User's message

        Returns:
            Final response to display to user
        """
        # Get response from THAU
        print("\n🤔 THAU pensando...", end="", flush=True)
        thau_response = self._call_ollama(user_message)
        print("\r" + " " * 30 + "\r", end="")  # Clear "thinking" message

        # Parse for tool calls
        clean_response, tool_params = self._parse_tool_call(thau_response)

        # Display THAU's text response
        print(f"🤖 THAU: {clean_response}")

        # Execute tool if needed
        if tool_params:
            image_result = self._generate_image(tool_params)

            if image_result:
                # Show image URL
                image_url = f"{self.api_base_url}{image_result['image_url']}"
                image_path = image_result.get('image_path')

                print(f"\n🖼️  Imagen disponible en: {image_url}")

                if image_path and Path(image_path).exists():
                    print(f"📁 Guardada en: {image_path}")

                    # Try to open image
                    try:
                        self._open_image(image_path)
                        print("   (Abriendo imagen...)")
                    except:
                        pass

        # Save to history
        self.conversation_history.append({
            "user": user_message,
            "assistant": clean_response,
        })

        return clean_response

    def interactive_chat(self):
        """Start interactive chat session"""
        print("="*80)
        print("💬 THAU Chat Interactivo")
        print("="*80)
        print("Comandos:")
        print("  - Escribe tu mensaje y presiona Enter")
        print("  - 'salir' o 'exit' para terminar")
        print("  - 'limpiar' para borrar historial")
        print("="*80)
        print()

        while True:
            try:
                # Get user input
                user_input = input("👤 Tú: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ["salir", "exit", "quit"]:
                    print("\n👋 ¡Hasta luego!")
                    break

                if user_input.lower() in ["limpiar", "clear"]:
                    self.conversation_history = []
                    print("🗑️  Historial limpiado")
                    print()
                    continue

                # Send message
                self.send_message(user_input)
                print()

            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print()


def main():
    """Main CLI"""
    parser = argparse.ArgumentParser(
        description="THAU Chat with tool calling for image generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="thau:latest",
        help="Ollama model name (default: thau:latest)"
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Send a single message (non-interactive)"
    )

    args = parser.parse_args()

    # Initialize chat
    chat = ThauChat(
        ollama_model=args.model,
        api_base_url=args.api,
    )

    if args.message:
        # Single message mode
        chat.send_message(args.message)
    else:
        # Interactive mode
        chat.interactive_chat()


if __name__ == "__main__":
    main()
