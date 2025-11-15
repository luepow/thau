"""
Test script for exported THAU model in Ollama
Tests model functionality and performance
"""

import time
import json
import requests
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger


class OllamaModelTester:
    """Tests THAU model exported to Ollama"""

    def __init__(self, model_name: str = "thau", base_url: str = "http://localhost:11434"):
        """
        Initialize tester

        Args:
            model_name: Name of model in Ollama
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.test_results = []

        logger.info(f"🧪 Ollama Model Tester initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   API: {base_url}")

    def check_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def check_model_exists(self) -> bool:
        """Check if model exists in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(m['name'].startswith(self.model_name) for m in models)
            return False
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, **kwargs) -> Dict:
        """
        Generate response from model

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dict with response and metadata
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        start_time = time.time()

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            elapsed = time.time() - start_time

            return {
                "success": True,
                "prompt": prompt,
                "response": result.get("response", ""),
                "elapsed_time": elapsed,
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
                "tokens_per_second": result.get("eval_count", 0) / (result.get("eval_duration", 1) / 1e9),
                "raw": result
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }

    def run_test(self, test_name: str, prompt: str, validation_fn: Optional[callable] = None) -> Dict:
        """
        Run a single test

        Args:
            test_name: Name of the test
            prompt: Prompt to test
            validation_fn: Optional function to validate response

        Returns:
            Test result dict
        """
        logger.info(f"\n🔍 Test: {test_name}")
        logger.info(f"   Prompt: {prompt[:100]}...")

        result = self.generate(prompt)

        if result["success"]:
            logger.info(f"   ✅ Response ({result['elapsed_time']:.2f}s, {result['tokens_per_second']:.1f} tok/s)")
            logger.info(f"   Response: {result['response'][:200]}...")

            # Validate if function provided
            if validation_fn:
                is_valid = validation_fn(result["response"])
                result["validation_passed"] = is_valid
                logger.info(f"   Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
            else:
                result["validation_passed"] = None
        else:
            logger.error(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")

        result["test_name"] = test_name
        result["timestamp"] = datetime.now().isoformat()

        self.test_results.append(result)
        return result

    def run_test_suite(self):
        """Run complete test suite"""
        logger.info("="*60)
        logger.info("🚀 Starting THAU Model Test Suite")
        logger.info("="*60)

        # Test 1: Ollama running
        logger.info("\n📡 Checking Ollama status...")
        if not self.check_ollama_running():
            logger.error("❌ Ollama is not running!")
            logger.error("   Start with: ollama serve")
            return False

        logger.info("✅ Ollama is running")

        # Test 2: Model exists
        logger.info("\n🔍 Checking if model exists...")
        if not self.check_model_exists():
            logger.error(f"❌ Model '{self.model_name}' not found!")
            logger.error(f"   Create with: ollama create {self.model_name} -f export/Modelfile")
            return False

        logger.info(f"✅ Model '{self.model_name}' found")

        # Test 3: Basic generation
        self.run_test(
            "basic_generation",
            "Hola, ¿quién eres?",
            validation_fn=lambda r: "thau" in r.lower() or "asistente" in r.lower()
        )

        # Test 4: Spanish fluency
        self.run_test(
            "spanish_fluency",
            "Explica qué es la inteligencia artificial en una oración.",
            validation_fn=lambda r: len(r.split()) > 5
        )

        # Test 5: Reasoning
        self.run_test(
            "math_reasoning",
            "¿Cuánto es 15 * 24? Piensa paso a paso.",
            validation_fn=lambda r: "360" in r
        )

        # Test 6: Programming
        self.run_test(
            "programming",
            "Escribe una función Python que calcule el factorial de un número.",
            validation_fn=lambda r: "def" in r and "factorial" in r.lower()
        )

        # Test 7: Instructions following
        self.run_test(
            "instructions",
            "Lista exactamente 3 características de un buen código. Usa viñetas.",
            validation_fn=lambda r: r.count("-") >= 2 or r.count("•") >= 2 or r.count("1") >= 1
        )

        # Test 8: Creativity
        self.run_test(
            "creativity",
            "Escribe un título creativo para una historia sobre un robot que aprende a sentir emociones.",
            validation_fn=lambda r: len(r) > 10 and len(r) < 200
        )

        # Test 9: Context understanding
        self.run_test(
            "context",
            "Si un tren viaja a 80 km/h durante 2.5 horas, ¿qué distancia recorre?",
            validation_fn=lambda r: "200" in r
        )

        # Test 10: Long-form generation
        self.run_test(
            "long_form",
            "Explica brevemente las diferencias entre aprendizaje supervisado y no supervisado.",
            validation_fn=lambda r: "supervisado" in r.lower() and len(r.split()) > 30
        )

        # Generate report
        self.generate_report()

        return True

    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("📊 TEST REPORT")
        logger.info("="*60)

        successful = [r for r in self.test_results if r.get("success")]
        failed = [r for r in self.test_results if not r.get("success")]

        logger.info(f"\n✅ Successful: {len(successful)}/{len(self.test_results)}")
        logger.info(f"❌ Failed: {len(failed)}/{len(self.test_results)}")

        if successful:
            avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
            avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)

            logger.info(f"\n⚡ Performance:")
            logger.info(f"   Average response time: {avg_time:.2f}s")
            logger.info(f"   Average tokens/sec: {avg_tps:.1f}")

        # Validation results
        validated = [r for r in self.test_results if r.get("validation_passed") is not None]
        if validated:
            passed = [r for r in validated if r["validation_passed"]]
            logger.info(f"\n✓ Validation:")
            logger.info(f"   Passed: {len(passed)}/{len(validated)}")

        # Save to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n💾 Full report saved to: {report_file}")

        # Summary
        logger.info("\n" + "="*60)
        if len(successful) == len(self.test_results):
            logger.info("🎉 ALL TESTS PASSED!")
        elif len(successful) > len(self.test_results) / 2:
            logger.info("⚠️  MOST TESTS PASSED (some issues)")
        else:
            logger.info("❌ MULTIPLE FAILURES - Check configuration")
        logger.info("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test THAU model in Ollama")

    parser.add_argument(
        "--model",
        type=str,
        default="thau",
        help="Model name in Ollama"
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to test (skip test suite)"
    )

    args = parser.parse_args()

    tester = OllamaModelTester(model_name=args.model, base_url=args.url)

    if args.prompt:
        # Single test
        result = tester.generate(args.prompt)
        if result["success"]:
            print(f"\n{result['response']}\n")
            print(f"⏱️  {result['elapsed_time']:.2f}s | {result['tokens_per_second']:.1f} tok/s")
        else:
            print(f"❌ Error: {result.get('error')}")
    else:
        # Full test suite
        tester.run_test_suite()


if __name__ == "__main__":
    main()
