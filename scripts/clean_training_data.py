#!/usr/bin/env python3
"""
Limpia los datos de entrenamiento de THAU eliminando registros de baja calidad.

Criterios de filtrado:
1. Respuestas que empiezan con otra pregunta
2. Respuestas muy cortas (< 100 caracteres)
3. Baja confianza (< 0.7)
4. Contenido incoherente (hablar de mercados en matemáticas)
5. Preguntas sin sentido ("futuro de polígono")
6. Respuestas con repetición excesiva
7. Respuestas que mencionan "sitio web" sin contexto
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


def is_good_quality(data: Dict) -> Tuple[bool, str]:
    """
    Verifica si un registro es de buena calidad.

    Returns:
        Tuple[bool, str]: (es_bueno, razón_si_es_malo)
    """
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    confidence = data.get("confidence", 0.0)
    category = (data.get("category") or "").lower()

    # 1. Confianza mínima
    if confidence < 0.7:
        return False, "low_confidence"

    # 2. Respuesta muy corta
    if len(answer) < 100:
        return False, "too_short"

    # 3. Respuesta empieza con pregunta o puntuación extraña
    if answer.startswith("¿") or answer.startswith("?"):
        return False, "starts_with_question"
    if answer.startswith(",") or answer.startswith(".") or answer.startswith(";"):
        return False, "starts_with_punctuation"

    # 3b. Respuesta contiene pregunta en las primeras líneas
    first_lines = answer[:200]
    if first_lines.count("?") >= 2:
        return False, "too_many_questions"

    # 3c. Respuesta en inglés cuando debería ser español
    answer_lower = answer.lower()
    english_starts = ["certainly", "here's", "here is", "i would", "i can", "to help", "how can i"]
    for eng in english_starts:
        if answer_lower.startswith(eng):
            return False, "wrong_language"

    # 4. Respuesta menciona que no puede/sabe
    negative_patterns = [
        "no tengo acceso",
        "no puedo responder",
        "no puido",  # typo común
        "no tenga acceso",
        "no puedo proporcionar",
        "necesito más contexto",
        "necesitaría más información",
    ]
    for pattern in negative_patterns:
        if pattern in answer_lower:
            return False, "cannot_answer"

    # 5. Incoherencia de dominio
    math_category = "matematica" in category
    business_words = ["mercado", "consumidor", "empresa", "cliente", "negocio", "ventas"]
    if math_category:
        business_count = sum(1 for w in business_words if w in answer_lower)
        if business_count >= 2:
            return False, "domain_mismatch"

    # 6. Preguntas sin sentido
    nonsense_patterns = [
        r"futuro de (polígono|jacobiano|matriz|derivada|integral|función|ecuación)",
        r"problemas de seguridad en (polígono|jacobiano|derivada|integral|triángulo)",
    ]
    question_lower = question.lower()
    for pattern in nonsense_patterns:
        if re.search(pattern, question_lower):
            return False, "nonsense_question"

    # 7. Repetición excesiva
    if answer.count("Sin embargo") > 3:
        return False, "excessive_repetition"
    if answer.count("por ejemplo") > 5:
        return False, "excessive_repetition"

    # 8. Respuesta truncada (termina abruptamente)
    if answer.endswith("...") or answer.endswith(".."):
        return False, "truncated"
    if re.search(r'\b(el|la|los|las|un|una|y|o|que|de|en|con)\s*$', answer):
        return False, "truncated"

    # 9. Contenido irrelevante
    if "sitio web" in answer_lower and "web" not in question_lower:
        return False, "irrelevant_content"

    # 10. Respuesta es solo código sin explicación
    code_lines = len(re.findall(r'```', answer))
    text_without_code = re.sub(r'```[\s\S]*?```', '', answer)
    if code_lines >= 2 and len(text_without_code.strip()) < 50:
        return False, "code_only"

    return True, "good"


def clean_training_data(
    input_dir: Path,
    output_file: Path,
    backup: bool = True
) -> Dict:
    """
    Limpia los datos de entrenamiento.

    Args:
        input_dir: Directorio con archivos .jsonl
        output_file: Archivo de salida con datos limpios
        backup: Si hacer backup de los archivos originales

    Returns:
        Estadísticas de limpieza
    """
    stats = {
        "total_processed": 0,
        "kept": 0,
        "removed": 0,
        "removal_reasons": defaultdict(int),
        "by_file": {},
        "by_category": defaultdict(lambda: {"total": 0, "kept": 0}),
    }

    good_records = []

    # Procesar cada archivo
    for file in sorted(input_dir.glob("qa_*.jsonl")):
        file_stats = {"total": 0, "kept": 0, "removed": 0}

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    stats["total_processed"] += 1
                    file_stats["total"] += 1

                    category = data.get("category", "unknown")
                    stats["by_category"][category]["total"] += 1

                    is_good, reason = is_good_quality(data)

                    if is_good:
                        good_records.append(data)
                        stats["kept"] += 1
                        file_stats["kept"] += 1
                        stats["by_category"][category]["kept"] += 1
                    else:
                        stats["removed"] += 1
                        file_stats["removed"] += 1
                        stats["removal_reasons"][reason] += 1

                except json.JSONDecodeError:
                    continue

        stats["by_file"][file.name] = file_stats

    # Guardar datos limpios
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in good_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Backup de originales si se solicita
    if backup:
        backup_dir = input_dir / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for file in input_dir.glob("qa_*.jsonl"):
            import shutil
            shutil.copy(file, backup_dir / file.name)
        stats["backup_dir"] = str(backup_dir)

    return stats


def print_stats(stats: Dict):
    """Imprime estadísticas de limpieza."""
    print("=" * 70)
    print("  LIMPIEZA DE DATOS DE ENTRENAMIENTO - THAU")
    print("=" * 70)

    pct_kept = 100 * stats["kept"] / max(1, stats["total_processed"])
    pct_removed = 100 * stats["removed"] / max(1, stats["total_processed"])

    print(f"\n📊 RESUMEN:")
    print(f"   Total procesados: {stats['total_processed']}")
    print(f"   ✅ Conservados:   {stats['kept']} ({pct_kept:.1f}%)")
    print(f"   ❌ Eliminados:    {stats['removed']} ({pct_removed:.1f}%)")

    if stats.get("backup_dir"):
        print(f"\n💾 Backup guardado en: {stats['backup_dir']}")

    print(f"\n🔍 RAZONES DE ELIMINACIÓN:")
    for reason, count in sorted(stats["removal_reasons"].items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")

    print(f"\n📁 POR ARCHIVO:")
    for fname, fstats in stats["by_file"].items():
        kept_pct = 100 * fstats['kept'] / max(1, fstats['total'])
        status = "🟢" if kept_pct > 20 else "🟡" if kept_pct > 5 else "🔴"
        print(f"   {status} {fname}: {fstats['kept']}/{fstats['total']} conservados ({kept_pct:.0f}%)")

    print(f"\n📂 POR CATEGORÍA:")
    for cat, cat_stats in sorted(stats["by_category"].items(), key=lambda x: -x[1]["kept"]):
        if cat_stats["kept"] > 0:
            pct = 100 * cat_stats['kept'] / max(1, cat_stats['total'])
            print(f"   - {cat}: {cat_stats['kept']}/{cat_stats['total']} ({pct:.0f}%)")

    print("\n" + "=" * 70)


def show_examples(input_dir: Path, n_good: int = 3, n_bad: int = 3):
    """Muestra ejemplos de datos buenos y malos."""
    good_examples = []
    bad_examples = []

    for file in input_dir.glob("qa_*.jsonl"):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    is_good, reason = is_good_quality(data)

                    if is_good and len(good_examples) < n_good:
                        good_examples.append(data)
                    elif not is_good and len(bad_examples) < n_bad:
                        data["_rejection_reason"] = reason
                        bad_examples.append(data)

                    if len(good_examples) >= n_good and len(bad_examples) >= n_bad:
                        break
                except:
                    continue

    print("\n" + "=" * 70)
    print("  EJEMPLOS DE DATOS BUENOS ✅")
    print("=" * 70)
    for i, ex in enumerate(good_examples, 1):
        print(f"\n[{i}] Q: {ex['question'][:80]}...")
        print(f"    A: {ex['answer'][:150]}...")
        print(f"    Conf: {ex.get('confidence', 'N/A')}, Cat: {ex.get('category', 'N/A')}")

    print("\n" + "=" * 70)
    print("  EJEMPLOS DE DATOS MALOS ❌")
    print("=" * 70)
    for i, ex in enumerate(bad_examples, 1):
        print(f"\n[{i}] Q: {ex['question'][:80]}...")
        print(f"    A: {ex['answer'][:150]}...")
        print(f"    Razón: {ex['_rejection_reason']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpia datos de entrenamiento de THAU")
    parser.add_argument("--input-dir", type=str, default="./data/self_questioning",
                        help="Directorio con archivos de entrenamiento")
    parser.add_argument("--output", type=str, default="./data/self_questioning/cleaned_training.jsonl",
                        help="Archivo de salida con datos limpios")
    parser.add_argument("--no-backup", action="store_true",
                        help="No hacer backup de archivos originales")
    parser.add_argument("--examples", action="store_true",
                        help="Mostrar ejemplos de datos buenos y malos")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar estadísticas, no limpiar")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)

    if args.examples:
        show_examples(input_dir)

    if args.dry_run:
        # Solo analizar sin guardar
        stats = {
            "total_processed": 0,
            "kept": 0,
            "removed": 0,
            "removal_reasons": defaultdict(int),
            "by_file": {},
            "by_category": defaultdict(lambda: {"total": 0, "kept": 0}),
        }

        for file in sorted(input_dir.glob("qa_*.jsonl")):
            file_stats = {"total": 0, "kept": 0, "removed": 0}
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        stats["total_processed"] += 1
                        file_stats["total"] += 1
                        category = data.get("category", "unknown")
                        stats["by_category"][category]["total"] += 1

                        is_good, reason = is_good_quality(data)
                        if is_good:
                            stats["kept"] += 1
                            file_stats["kept"] += 1
                            stats["by_category"][category]["kept"] += 1
                        else:
                            stats["removed"] += 1
                            file_stats["removed"] += 1
                            stats["removal_reasons"][reason] += 1
                    except:
                        continue
            stats["by_file"][file.name] = file_stats

        print_stats(stats)
        print("\n⚠️  Modo dry-run: No se guardaron cambios")
    else:
        stats = clean_training_data(input_dir, output_file, backup=not args.no_backup)
        print_stats(stats)
        print(f"\n✅ Datos limpios guardados en: {output_file}")
