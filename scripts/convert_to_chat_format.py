"""Convert instruction/output dataset to TinyLlama chat format."""

import json
from pathlib import Path

def convert_to_chat_format(input_file: str, output_file: str):
    """Convert dataset to TinyLlama chat format.

    TinyLlama format:
    <|system|>
    System message</s>
    <|user|>
    User message</s>
    <|assistant|>
    Assistant response</s>
    """

    converted = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)

                instruction = item.get('instruction', item.get('prompt', ''))
                output = item.get('output', item.get('response', ''))

                # Format as TinyLlama chat
                chat_text = f"""<|system|>
Eres THAU, un asistente experto en programación. Responde de forma clara, concisa y en español.</s>
<|user|>
{instruction}</s>
<|assistant|>
{output}</s>"""

                converted.append({
                    'instruction': instruction,
                    'output': output,
                    'text': chat_text,  # Full chat format for training
                })

    # Save converted dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted)} examples")
    print(f"Saved to: {output_file}")

    # Show example
    if converted:
        print("\n--- Example ---")
        print(converted[0]['text'][:500])

    return len(converted)


if __name__ == "__main__":
    input_file = "data/datasets/programming_combined_20251202.jsonl"
    output_file = "data/datasets/programming_chat_format.jsonl"

    convert_to_chat_format(input_file, output_file)
