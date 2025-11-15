"""Benchmark script for model performance."""

import time
import torch
from loguru import logger
import json

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from core.inference.generator import TextGenerator


def benchmark_generation(generator: TextGenerator, prompts: list, iterations: int = 5):
    """Benchmark text generation."""
    logger.info("Benchmarking text generation...")

    total_time = 0
    total_tokens = 0

    for prompt in prompts:
        for _ in range(iterations):
            start_time = time.time()

            generated = generator.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            # Estimate tokens
            tokens = len(generated[0].split()) * 1.3
            total_tokens += tokens

    avg_time = total_time / (len(prompts) * iterations)
    tokens_per_second = total_tokens / total_time

    return {
        "avg_generation_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "total_iterations": len(prompts) * iterations,
    }


def benchmark_inference(model, tokenizer, device_manager, batch_sizes: list = [1, 4, 8]):
    """Benchmark inference throughput."""
    logger.info("Benchmarking inference throughput...")

    results = {}

    test_input = "This is a test sentence for benchmarking inference performance."

    for batch_size in batch_sizes:
        # Prepare batch
        texts = [test_input] * batch_size
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = device_manager.to_device(encodings["input_ids"])

        # Warmup
        with torch.no_grad():
            model(input_ids)

        # Benchmark
        iterations = 10
        start_time = time.time()

        with torch.no_grad():
            for _ in range(iterations):
                model(input_ids)

        elapsed = time.time() - start_time
        throughput = (batch_size * iterations) / elapsed

        results[f"batch_size_{batch_size}"] = {
            "throughput_samples_per_sec": throughput,
            "avg_time_per_batch": elapsed / iterations,
        }

    return results


def main():
    """Run benchmarks."""
    logger.info("Starting performance benchmarking...")

    # Load config
    config = get_config()

    # Initialize model
    logger.info("Loading model...")
    model_adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=config.USE_QUANTIZATION,
    )
    model_adapter.load_model()
    model_adapter.load_tokenizer()

    # Initialize generator
    generator = TextGenerator(model_adapter=model_adapter)

    # Test prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "How do neural networks work?",
    ]

    # Run benchmarks
    results = {
        "device_info": model_adapter.device_manager.get_device_info(),
        "model_info": model_adapter.get_model_info(),
    }

    # Generation benchmark
    gen_results = benchmark_generation(generator, prompts)
    results["generation_benchmark"] = gen_results

    # Inference benchmark
    inf_results = benchmark_inference(
        model_adapter.model,
        model_adapter.tokenizer,
        model_adapter.device_manager,
    )
    results["inference_benchmark"] = inf_results

    # Save results
    output_file = "./data/logs/benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*50)
    logger.info(f"Device: {results['device_info']['device']}")
    logger.info(f"Model: {results['model_info']['model_name']}")
    logger.info(f"\nGeneration:")
    logger.info(f"  Avg Time: {gen_results['avg_generation_time']:.3f}s")
    logger.info(f"  Tokens/sec: {gen_results['tokens_per_second']:.1f}")
    logger.info(f"\nInference Throughput:")
    for batch_size, metrics in inf_results.items():
        logger.info(f"  {batch_size}: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
    logger.info("="*50)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
