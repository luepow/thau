"""Test THAU Code Agent capabilities."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adapters.model_adapter import ModelAdapter
from config.base_config import get_config


def generate_response(adapter, prompt, max_tokens=300):
    """Generate a response from the model."""
    inputs = adapter.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(adapter.model.device) for k, v in inputs.items()}

    outputs = adapter.model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        pad_token_id=adapter.tokenizer.eos_token_id,
    )

    response = adapter.tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def test_code_agent():
    """Test Code Agent generation with tool calling."""

    print("=" * 60)
    print("THAU Code Agent Test")
    print("=" * 60)

    config = get_config()

    print("\n[1] Loading base model...")
    adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=config.USE_QUANTIZATION,
    )
    adapter.load_model()
    adapter.load_tokenizer()

    # Load LoRA adapters
    print("\n[2] Loading LoRA adapters...")
    agent_checkpoint = "data/checkpoints/incremental/specialized/agent_20251202_091217"

    try:
        from peft import PeftModel
        adapter.model = PeftModel.from_pretrained(
            adapter.model,
            agent_checkpoint,
            is_trainable=False,
        )
        print("    LoRA adapters loaded!")
    except Exception as e:
        print(f"    Error loading LoRA: {e}")
        return

    # Test using instruction format (matching training data)
    print("\n[3] Testing with instruction format (matching training)...")
    print("-" * 60)

    tests = [
        {
            "name": "Create Flask App",
            "instruction": "Crea una aplicación web simple con Flask que tenga un endpoint /hello"
        },
        {
            "name": "Search Files",
            "instruction": "Encuentra todos los archivos Python que contienen 'import torch'"
        },
        {
            "name": "Fix Bug",
            "instruction": "Revisa el archivo calculator.py y corrige el bug en la función divide"
        },
    ]

    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test['name']}")

        # Use instruction format like in training data
        prompt = f"### Instruction:\n{test['instruction']}\n\n### Response:\n"

        print(f"Prompt: {test['instruction'][:80]}...")
        response = generate_response(adapter, prompt, max_tokens=400)
        print(f"Response:\n{response[:600]}")
        print("-" * 40)

    # Also test simple completion
    print("\n[4] Testing simple code completion...")
    code_prompt = "def fibonacci(n):\n    "
    response = generate_response(adapter, code_prompt, max_tokens=150)
    print(f"Prompt: def fibonacci(n):")
    print(f"Response:\n{response[:300]}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_code_agent()
