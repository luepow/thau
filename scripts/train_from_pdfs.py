#!/usr/bin/env python3
"""
Script para entrenar THAU desde PDFs de documentación

Extrae texto de PDFs, genera pares Q&A, y entrena el modelo.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  Error extrayendo {pdf_path}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Limpia el texto extraído"""
    # Remover múltiples espacios y líneas en blanco
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remover caracteres especiales problemáticos
    text = text.replace('\x00', '')
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Divide el texto en chunks manejables"""
    chunks = []
    words = text.split()

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def generate_qa_pairs(chunks: List[str], source_name: str, category: str) -> List[Dict]:
    """Genera pares de pregunta-respuesta para entrenamiento"""
    qa_pairs = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:  # Skip chunks muy pequeños
            continue

        # Determinar el tipo de contenido basado en keywords
        chunk_lower = chunk.lower()

        # Generar diferentes tipos de prompts según el contenido
        if 'agent' in chunk_lower or 'multi-agent' in chunk_lower:
            questions = [
                f"Explica sobre agentes AI según {source_name}",
                f"¿Cómo funcionan los sistemas multi-agente?",
                f"Describe la arquitectura de agentes en {source_name}"
            ]
        elif 'tool' in chunk_lower or 'function' in chunk_lower:
            questions = [
                f"¿Cómo se usan las herramientas en {source_name}?",
                f"Explica el tool calling según {source_name}",
                f"¿Qué herramientas están disponibles?"
            ]
        elif 'model' in chunk_lower or 'llm' in chunk_lower:
            questions = [
                f"¿Qué modelos se mencionan en {source_name}?",
                f"Explica sobre modelos de lenguaje según {source_name}",
                f"¿Cómo se configuran los modelos?"
            ]
        elif 'rag' in chunk_lower or 'retrieval' in chunk_lower:
            questions = [
                f"Explica RAG según {source_name}",
                f"¿Cómo funciona el retrieval augmented generation?",
                f"¿Qué es Agentic RAG?"
            ]
        elif 'server' in chunk_lower or 'api' in chunk_lower:
            questions = [
                f"¿Cómo se configura el servidor según {source_name}?",
                f"Explica la API de {source_name}",
                f"¿Cuáles son los endpoints disponibles?"
            ]
        else:
            questions = [
                f"Explica este concepto de {source_name}: {chunk[:50]}...",
                f"¿Qué información importante hay sobre {source_name}?",
                f"Resume la documentación de {source_name}"
            ]

        # Usar la primera pregunta más relevante
        question = questions[i % len(questions)]

        qa_pairs.append({
            "instruction": question,
            "input": "",
            "output": chunk,
            "category": category,
            "source": source_name
        })

    return qa_pairs


def convert_to_chat_format(qa_pairs: List[Dict]) -> List[Dict]:
    """Convierte a formato de chat para entrenamiento"""
    chat_data = []

    for pair in qa_pairs:
        # Formato TinyLlama chat
        text = f"<|system|>\nEres THAU, un asistente AI experto en {pair['category']}.</s>\n"
        text += f"<|user|>\n{pair['instruction']}</s>\n"
        text += f"<|assistant|>\n{pair['output']}</s>"

        chat_data.append({
            "text": text,
            "category": pair["category"],
            "source": pair["source"]
        })

    return chat_data


def process_pdfs(pdf_folders: List[str]) -> Tuple[List[Dict], Dict]:
    """Procesa todos los PDFs de las carpetas"""
    all_qa_pairs = []
    stats = {"total_pdfs": 0, "total_chunks": 0, "by_category": {}}

    category_map = {
        "hugginface": "huggingface_agents",
        "Autogen": "autogen_agents",
        "llama": "llama_cpp"
    }

    for folder in pdf_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Carpeta no encontrada: {folder}")
            continue

        # Determinar categoría
        folder_name = folder_path.name
        category = category_map.get(folder_name, folder_name.lower())

        print(f"\n📁 Procesando: {folder_name}")

        for pdf_file in folder_path.glob("*.pdf"):
            print(f"  📄 {pdf_file.name}")
            stats["total_pdfs"] += 1

            # Extraer texto
            text = extract_text_from_pdf(str(pdf_file))
            if not text:
                continue

            text = clean_text(text)

            # Dividir en chunks
            chunks = split_into_chunks(text)
            stats["total_chunks"] += len(chunks)

            # Generar Q&A
            source_name = pdf_file.stem.replace("_", " ").replace("-", " ")
            qa_pairs = generate_qa_pairs(chunks, source_name, category)
            all_qa_pairs.extend(qa_pairs)

            # Actualizar stats
            if category not in stats["by_category"]:
                stats["by_category"][category] = 0
            stats["by_category"][category] += len(qa_pairs)

            print(f"    → {len(chunks)} chunks, {len(qa_pairs)} pares Q&A")

    return all_qa_pairs, stats


def save_dataset(qa_pairs: List[Dict], output_path: str):
    """Guarda el dataset en formato JSONL"""
    chat_data = convert_to_chat_format(qa_pairs)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return len(chat_data)


def train_model(dataset_path: str, epochs: int = 3):
    """Entrena el modelo con el dataset generado"""
    print("\n🧠 Iniciando entrenamiento...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import load_dataset
        import torch

        # Cargar dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        print(f"  Dataset cargado: {len(dataset)} ejemplos")

        # Cargar modelo y tokenizer
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Intentar cargar checkpoint existente
        checkpoint_path = project_root / "data" / "checkpoints" / "specialized_training" / "final"
        if checkpoint_path.exists():
            print(f"  Cargando checkpoint: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        else:
            print(f"  Cargando modelo base: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Habilitar gradientes para entrenamiento
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        tokenizer.pad_token = tokenizer.eos_token

        # Tokenizar con labels para language modeling
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            # Para causal LM, labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        # Data collator para language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, no masked LM
        )

        # Configurar entrenamiento
        output_dir = project_root / "data" / "checkpoints" / "pdf_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            fp16=False,  # MPS no soporta fp16
            report_to="none",
            use_cpu=False,
        )

        # Crear trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Entrenar
        print("  Entrenando...")
        trainer.train()

        # Guardar modelo final
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        print(f"\n✅ Modelo guardado en: {final_path}")
        return True

    except Exception as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  THAU - Entrenamiento desde PDFs")
    print("=" * 60)

    # Carpetas de PDFs
    pdf_folders = [
        "/Users/lperez/Library/CloudStorage/Dropbox/Books/hugginface",
        "/Users/lperez/Library/CloudStorage/Dropbox/Books/Autogen",
        "/Users/lperez/Library/CloudStorage/Dropbox/Books/llama"
    ]

    # Procesar PDFs
    print("\n📚 Extrayendo texto de PDFs...")
    qa_pairs, stats = process_pdfs(pdf_folders)

    print(f"\n📊 Estadísticas:")
    print(f"  PDFs procesados: {stats['total_pdfs']}")
    print(f"  Chunks totales: {stats['total_chunks']}")
    print(f"  Pares Q&A generados: {len(qa_pairs)}")
    print(f"  Por categoría:")
    for cat, count in stats["by_category"].items():
        print(f"    - {cat}: {count}")

    if not qa_pairs:
        print("\n❌ No se generaron datos de entrenamiento")
        return

    # Guardar dataset
    output_path = project_root / "data" / "datasets" / "pdf_training_agents.jsonl"
    saved_count = save_dataset(qa_pairs, str(output_path))
    print(f"\n💾 Dataset guardado: {output_path}")
    print(f"  {saved_count} ejemplos de entrenamiento")

    # Entrenar modelo
    train_model(str(output_path), epochs=3)

    print("\n" + "=" * 60)
    print("  Entrenamiento completado!")
    print("=" * 60)


if __name__ == "__main__":
    main()
