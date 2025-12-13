#!/usr/bin/env python3
"""
Script para entrenar THAU con manuales contables

Extrae texto de PDFs de contabilidad bancaria y genera dataset de entrenamiento.
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


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Divide el texto en chunks manejables"""
    chunks = []
    words = text.split()

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip() and len(chunk) > 50:
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def generate_accounting_qa(chunks: List[str], source_name: str, category: str) -> List[Dict]:
    """Genera pares de pregunta-respuesta para contabilidad"""
    qa_pairs = []

    # Keywords para detectar tipo de contenido contable
    accounting_keywords = {
        'activo': ['activo', 'activos', 'disponibilidades', 'inversiones', 'cartera'],
        'pasivo': ['pasivo', 'pasivos', 'obligaciones', 'depositos', 'captaciones'],
        'patrimonio': ['patrimonio', 'capital', 'reservas', 'utilidades'],
        'cuenta': ['cuenta', 'cuentas', 'subcuenta', 'codigo'],
        'registro': ['registro', 'asiento', 'debito', 'credito', 'debe', 'haber'],
        'operacion': ['operacion', 'transaccion', 'movimiento'],
        'balance': ['balance', 'estado', 'situacion', 'financiero'],
        'garantia': ['garantia', 'fianza', 'aval', 'reciproca'],
        'cambio': ['cambio', 'divisa', 'moneda', 'tipo de cambio'],
        'banco': ['banco', 'bancario', 'institucion', 'financiera'],
    }

    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:
            continue

        chunk_lower = chunk.lower()

        # Detectar tipo de contenido
        detected_topics = []
        for topic, keywords in accounting_keywords.items():
            if any(kw in chunk_lower for kw in keywords):
                detected_topics.append(topic)

        # Generar preguntas según contenido
        questions = []

        if 'cuenta' in detected_topics or 'codigo' in chunk_lower:
            questions.extend([
                f"¿Cómo se codifican las cuentas contables según {source_name}?",
                f"Explica la estructura del catálogo de cuentas en {source_name}",
                f"¿Qué significa este código contable según el manual?"
            ])

        if 'activo' in detected_topics:
            questions.extend([
                f"¿Cómo se registran los activos según {source_name}?",
                f"Explica las cuentas de activo en {source_name}",
                f"¿Qué tipos de activos contempla el manual contable?"
            ])

        if 'pasivo' in detected_topics:
            questions.extend([
                f"¿Cómo se registran los pasivos según {source_name}?",
                f"Explica las obligaciones financieras en {source_name}",
                f"¿Qué cuentas de pasivo menciona el manual?"
            ])

        if 'registro' in detected_topics or 'asiento' in chunk_lower:
            questions.extend([
                f"¿Cómo se realiza el registro contable según {source_name}?",
                f"Explica la dinámica de los asientos contables",
                f"¿Cuál es el procedimiento para registrar esta operación?"
            ])

        if 'garantia' in detected_topics:
            questions.extend([
                f"¿Cómo se contabilizan las garantías según {source_name}?",
                f"Explica el sistema de garantías recíprocas",
                f"¿Qué cuentas se usan para registrar garantías?"
            ])

        if 'cambio' in detected_topics or 'divisa' in chunk_lower:
            questions.extend([
                f"¿Cómo se registran las operaciones de cambio?",
                f"Explica la contabilidad de divisas según {source_name}",
                f"¿Cómo se contabilizan las diferencias de cambio?"
            ])

        if 'banco' in detected_topics:
            questions.extend([
                f"¿Cuáles son las normas contables para instituciones bancarias?",
                f"Explica los requerimientos contables del manual bancario",
                f"¿Cómo se aplica la normativa a instituciones financieras?"
            ])

        # Preguntas genéricas si no se detectaron temas específicos
        if not questions:
            questions = [
                f"Explica este concepto contable de {source_name}",
                f"¿Qué establece el manual contable sobre este tema?",
                f"Resume esta sección del manual {source_name}"
            ]

        # Usar pregunta rotativa
        question = questions[i % len(questions)]

        qa_pairs.append({
            "instruction": question,
            "input": "",
            "output": chunk,
            "category": category,
            "source": source_name,
            "topics": detected_topics
        })

    return qa_pairs


def convert_to_chat_format(qa_pairs: List[Dict]) -> List[Dict]:
    """Convierte a formato de chat para entrenamiento"""
    chat_data = []

    for pair in qa_pairs:
        # System prompt especializado en contabilidad
        system_prompt = """Eres THAU, un asistente experto en contabilidad bancaria y financiera.

Tienes conocimiento de:
- Manuales contables de SUDEBAN (Venezuela)
- Plan de cuentas para instituciones bancarias
- Casas de cambio y operaciones en divisas
- Sistema de garantías recíprocas para PYMES
- Normas de reexpresión monetaria
- Registros contables y asientos
- Estados financieros

Responde de forma técnica pero clara, citando las normas cuando sea relevante."""

        text = f"<|system|>\n{system_prompt}</s>\n"
        text += f"<|user|>\n{pair['instruction']}</s>\n"
        text += f"<|assistant|>\n{pair['output']}</s>"

        chat_data.append({
            "text": text,
            "category": pair["category"],
            "source": pair["source"]
        })

    return chat_data


def process_contable_pdfs(base_folder: str) -> Tuple[List[Dict], Dict]:
    """Procesa todos los PDFs contables"""
    all_qa_pairs = []
    stats = {"total_pdfs": 0, "total_chunks": 0, "by_category": {}}

    base_path = Path(base_folder)

    # Mapeo de carpetas a categorías
    category_map = {
        "SISTEMA-NACIONAL-GARANTIAS-RECIPROCAS-PEQUENA-Y-MEDIANA-EMPRESA": "garantias_pyme",
        "CASAS-DE-CAMBIO": "casas_cambio",
        "INSTITUCIONES-BANCARIAS": "banca"
    }

    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        category = category_map.get(folder_name, folder_name.lower().replace("-", "_"))

        print(f"\n📁 Procesando: {folder_name}")

        pdf_files = list(folder.glob("*.pdf"))
        print(f"  Encontrados: {len(pdf_files)} PDFs")

        for pdf_file in pdf_files:
            print(f"  📄 {pdf_file.name}")
            stats["total_pdfs"] += 1

            # Extraer texto
            text = extract_text_from_pdf(str(pdf_file))
            if not text:
                print(f"    ⚠️ Sin texto extraíble")
                continue

            text = clean_text(text)

            # Dividir en chunks
            chunks = split_into_chunks(text)
            stats["total_chunks"] += len(chunks)

            # Generar Q&A
            source_name = pdf_file.stem.replace("-", " ").replace("_", " ")
            qa_pairs = generate_accounting_qa(chunks, source_name, category)
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


def train_model(dataset_path: str, epochs: int = 5):
    """Entrena el modelo con el dataset contable"""
    print("\n🧠 Iniciando entrenamiento con datos contables...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import load_dataset
        import torch

        # Cargar dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        print(f"  Dataset cargado: {len(dataset)} ejemplos")

        # Cargar modelo base (el último checkpoint)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        checkpoint_path = project_root / "data" / "checkpoints" / "pdf_training" / "final"

        if not checkpoint_path.exists():
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

        # Habilitar gradientes
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        tokenizer.pad_token = tokenizer.eos_token

        # Tokenizar
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Configurar entrenamiento
        output_dir = project_root / "data" / "checkpoints" / "contable_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,  # Learning rate más bajo para fine-tuning
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=False,
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Entrenar
        print("  Entrenando...")
        trainer.train()

        # Guardar
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
    print("  THAU - Entrenamiento Contable")
    print("=" * 60)

    # Carpeta de manuales contables
    contable_folder = "/Users/lperez/Library/CloudStorage/Dropbox/Books/Manuales Contables"

    # Procesar PDFs
    print("\n📚 Extrayendo texto de manuales contables...")
    qa_pairs, stats = process_contable_pdfs(contable_folder)

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
    output_path = project_root / "data" / "datasets" / "contable_training.jsonl"
    saved_count = save_dataset(qa_pairs, str(output_path))
    print(f"\n💾 Dataset guardado: {output_path}")
    print(f"  {saved_count} ejemplos de entrenamiento")

    # Entrenar
    train_model(str(output_path), epochs=5)

    print("\n" + "=" * 60)
    print("  Entrenamiento contable completado!")
    print("=" * 60)


if __name__ == "__main__":
    main()
