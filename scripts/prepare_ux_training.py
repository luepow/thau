"""Prepare UX/CSS dataset for training."""

import json
from pathlib import Path


def convert_and_combine():
    """Convert UX dataset to chat format and combine with existing."""

    print("=" * 60)
    print("  Preparing UX/CSS Dataset for Training")
    print("=" * 60)

    # Load UX dataset
    ux_file = "data/datasets/ux_css_frameworks.jsonl"
    ux_data = []

    print(f"\n[1] Loading UX/CSS dataset: {ux_file}")
    with open(ux_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ux_data.append(json.loads(line))

    print(f"    Loaded {len(ux_data)} UX/CSS examples")

    # Load existing programming dataset
    prog_file = "data/datasets/programming_chat_format.jsonl"
    prog_data = []

    print(f"\n[2] Loading existing programming dataset: {prog_file}")
    try:
        with open(prog_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    prog_data.append(json.loads(line))
        print(f"    Loaded {len(prog_data)} programming examples")
    except FileNotFoundError:
        print("    No existing programming dataset found")

    # Convert UX to chat format
    print("\n[3] Converting UX/CSS to TinyLlama chat format...")
    ux_chat = []

    for item in ux_data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        chat_text = f"""<|system|>
Eres THAU, un asistente experto en programacion, UX/UI y desarrollo web. Responde de forma clara, concisa y en espanol.</s>
<|user|>
{instruction}</s>
<|assistant|>
{output}</s>"""

        ux_chat.append({
            'instruction': instruction,
            'output': output,
            'text': chat_text,
        })

    print(f"    Converted {len(ux_chat)} examples")

    # Combine datasets
    print("\n[4] Combining datasets...")
    combined = prog_data + ux_chat
    print(f"    Total: {len(combined)} examples")
    print(f"    - Programming: {len(prog_data)}")
    print(f"    - UX/CSS: {len(ux_chat)}")

    # Save combined dataset
    output_file = "data/datasets/thau_v2_combined.jsonl"
    print(f"\n[5] Saving combined dataset: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"    Saved {len(combined)} examples")

    # Show sample
    print("\n[6] Sample from UX/CSS dataset:")
    print("-" * 50)
    if ux_chat:
        sample = ux_chat[0]
        print(f"Q: {sample['instruction'][:80]}...")
        print(f"A: {sample['output'][:200]}...")

    print("\n" + "=" * 60)
    print(f"  Dataset ready: {output_file}")
    print(f"  Total examples: {len(combined)}")
    print("=" * 60)

    return output_file, len(combined)


if __name__ == "__main__":
    convert_and_combine()
