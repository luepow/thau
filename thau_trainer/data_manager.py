"""
Gestor de datos para THAU
Maneja la cola de ejemplos nuevos y el tracking de qué se ha entrenado
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib


class DataManager:
    """Gestor de datos de entrenamiento"""

    def __init__(self):
        self.queue_dir = Path("./data/training_queue")
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        self.trained_file = Path("./data/logs/trained_examples.jsonl")
        self.trained_file.parent.mkdir(parents=True, exist_ok=True)

        self.trained_hashes = self._load_trained_hashes()

    def _load_trained_hashes(self) -> set:
        """Carga los hashes de ejemplos ya entrenados"""
        hashes = set()

        if self.trained_file.exists():
            with open(self.trained_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        hashes.add(data.get('hash'))
                    except:
                        continue

        return hashes

    def _hash_example(self, example: Dict) -> str:
        """Genera hash único para un ejemplo"""
        content = f"{example.get('instruction', '')}{example.get('output', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def add_example(self, instruction: str, output: str, metadata: Dict = None) -> str:
        """Agrega un nuevo ejemplo a la cola de entrenamiento"""

        example = {
            "instruction": instruction,
            "input": "",
            "output": output,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat(),
        }

        example_hash = self._hash_example(example)
        example["hash"] = example_hash

        # Guardar en cola
        queue_file = self.queue_dir / f"{example_hash}.json"

        if queue_file.exists() or example_hash in self.trained_hashes:
            return "duplicate"

        with open(queue_file, 'w') as f:
            json.dump(example, f, indent=2)

        return example_hash

    def add_batch(self, examples: List[Dict]) -> Dict:
        """Agrega un lote de ejemplos"""
        results = {
            "added": 0,
            "duplicates": 0,
            "errors": 0
        }

        for ex in examples:
            try:
                result = self.add_example(
                    instruction=ex.get('instruction', ''),
                    output=ex.get('output', ''),
                    metadata=ex.get('metadata', {})
                )

                if result == "duplicate":
                    results["duplicates"] += 1
                else:
                    results["added"] += 1

            except Exception as e:
                results["errors"] += 1

        return results

    def get_new_examples(self) -> List[Dict]:
        """Obtiene todos los ejemplos nuevos en la cola"""
        examples = []

        for queue_file in self.queue_dir.glob("*.json"):
            try:
                with open(queue_file, 'r') as f:
                    example = json.load(f)

                example_hash = example.get('hash')

                if example_hash and example_hash not in self.trained_hashes:
                    examples.append(example)

            except Exception as e:
                continue

        return examples

    def mark_as_trained(self, examples: List[Dict]):
        """Marca ejemplos como entrenados"""

        with open(self.trained_file, 'a') as f:
            for example in examples:
                example_hash = example.get('hash')

                if example_hash:
                    # Agregar a trained
                    trained_record = {
                        "hash": example_hash,
                        "trained_at": datetime.now().isoformat(),
                        "instruction_preview": example.get('instruction', '')[:100]
                    }

                    f.write(json.dumps(trained_record) + '\n')
                    self.trained_hashes.add(example_hash)

                    # Eliminar de cola
                    queue_file = self.queue_dir / f"{example_hash}.json"
                    if queue_file.exists():
                        queue_file.unlink()

    def get_stats(self) -> Dict:
        """Estadísticas de datos"""
        return {
            "pending_examples": len(self.get_new_examples()),
            "total_trained": len(self.trained_hashes),
            "queue_files": len(list(self.queue_dir.glob("*.json")))
        }
