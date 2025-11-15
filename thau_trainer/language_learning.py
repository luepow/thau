"""
Sistema de Aprendizaje Multilingüe para THAU
Permite aprender vocabulario, fonética, gramática de múltiples idiomas
"""

import json
import requests
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


class PhoneticLearner:
    """Aprende y gestiona la fonética de palabras"""

    def __init__(self):
        self.phonetic_db = Path("./data/language/phonetics.jsonl")
        self.phonetic_db.parent.mkdir(parents=True, exist_ok=True)
        self.phonetics = {}
        self._load_phonetics()

    def _load_phonetics(self):
        """Carga base de datos fonética"""
        if self.phonetic_db.exists():
            with open(self.phonetic_db, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    key = f"{entry['word']}_{entry['language']}"
                    self.phonetics[key] = entry

    def add_phonetic(
        self,
        word: str,
        language: str,
        ipa: str,
        syllables: List[str] = None,
        stress: int = None
    ):
        """
        Añade información fonética de una palabra

        Args:
            word: Palabra
            language: Código de idioma (es, en, fr, etc.)
            ipa: Notación IPA (International Phonetic Alphabet)
            syllables: División silábica
            stress: Índice de sílaba tónica (0-based)
        """
        key = f"{word}_{language}"

        entry = {
            "word": word,
            "language": language,
            "ipa": ipa,
            "syllables": syllables or [],
            "stress": stress,
            "timestamp": datetime.now().isoformat()
        }

        self.phonetics[key] = entry

        # Guardar
        with open(self.phonetic_db, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def get_phonetic(self, word: str, language: str) -> Optional[Dict]:
        """Obtiene fonética de una palabra"""
        key = f"{word}_{language}"
        return self.phonetics.get(key)

    def auto_learn_phonetic(self, word: str, language: str) -> Optional[Dict]:
        """
        Intenta aprender fonética automáticamente
        Usa APIs públicas o reglas heurísticas
        """
        # TODO: Integrar con APIs de diccionarios
        # Por ahora, reglas básicas para español

        if language == "es":
            return self._spanish_phonetic_rules(word)
        elif language == "en":
            # Para inglés, necesitaríamos API o dataset
            return None

        return None

    def _spanish_phonetic_rules(self, word: str) -> Dict:
        """Reglas fonéticas básicas del español"""
        word_lower = word.lower()

        # Dividir en sílabas (aproximado)
        syllables = self._spanish_syllabify(word_lower)

        # Determinar sílaba tónica
        stress = self._spanish_stress(word_lower, syllables)

        # IPA aproximado (simplificado)
        ipa = self._spanish_to_ipa(word_lower)

        phonetic = {
            "word": word,
            "language": "es",
            "ipa": ipa,
            "syllables": syllables,
            "stress": stress,
            "auto_generated": True
        }

        # Guardar
        self.add_phonetic(word, "es", ipa, syllables, stress)

        return phonetic

    def _spanish_syllabify(self, word: str) -> List[str]:
        """División silábica simple para español"""
        # Regla simplificada: V = vocal, C = consonante
        # Patrón: CV, CVC, etc.

        vowels = "aeiouáéíóúü"
        syllables = []
        current = ""

        for i, char in enumerate(word):
            current += char

            # Si es vocal y siguiente es consonante + vocal, cortar
            if char in vowels:
                if i + 2 < len(word):
                    next1 = word[i + 1]
                    next2 = word[i + 2] if i + 2 < len(word) else ""

                    if next1 not in vowels and next2 in vowels:
                        syllables.append(current)
                        current = ""

        if current:
            syllables.append(current)

        return syllables if syllables else [word]

    def _spanish_stress(self, word: str, syllables: List[str]) -> int:
        """Determina sílaba tónica en español"""
        # Reglas de acentuación
        vowels_accent = "áéíóú"

        # Si tiene tilde, esa es la tónica
        for i, syl in enumerate(syllables):
            if any(v in syl for v in vowels_accent):
                return i

        # Sin tilde: reglas
        if word.endswith(('n', 's')) or word[-1] in "aeiou":
            # Palabra llana (penúltima sílaba)
            return max(0, len(syllables) - 2)
        else:
            # Palabra aguda (última sílaba)
            return len(syllables) - 1

    def _spanish_to_ipa(self, word: str) -> str:
        """Conversión aproximada español -> IPA"""
        ipa_map = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': '', 'j': 'x', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'ñ': 'ɲ', 'p': 'p', 'q': 'k', 'r': 'r',
            's': 's', 't': 't', 'v': 'b', 'w': 'w', 'x': 'ks',
            'y': 'ʝ', 'z': 'θ',
        }

        ipa = ""
        for char in word.lower():
            ipa += ipa_map.get(char, char)

        return f"/{ipa}/"


class VocabularyBuilder:
    """Construye y gestiona vocabulario de múltiples idiomas"""

    def __init__(self):
        self.vocab_dir = Path("./data/language/vocabulary")
        self.vocab_dir.mkdir(parents=True, exist_ok=True)

        self.vocabularies = defaultdict(dict)  # {language: {word: entry}}
        self._load_vocabularies()

    def _load_vocabularies(self):
        """Carga vocabularios existentes"""
        for vocab_file in self.vocab_dir.glob("*.jsonl"):
            language = vocab_file.stem
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.vocabularies[language][entry['word']] = entry

    def add_word(
        self,
        word: str,
        language: str,
        definition: str,
        pos: str = None,  # Part of speech (noun, verb, etc.)
        examples: List[str] = None,
        synonyms: List[str] = None,
        antonyms: List[str] = None,
        translations: Dict[str, str] = None,
        frequency: int = 1
    ):
        """Añade palabra al vocabulario"""

        entry = {
            "word": word,
            "language": language,
            "definition": definition,
            "pos": pos,
            "examples": examples or [],
            "synonyms": synonyms or [],
            "antonyms": antonyms or [],
            "translations": translations or {},
            "frequency": frequency,
            "learned_date": datetime.now().isoformat(),
            "last_reviewed": datetime.now().isoformat(),
            "mastery_level": 0  # 0-5, aumenta con revisiones
        }

        self.vocabularies[language][word] = entry

        # Guardar
        vocab_file = self.vocab_dir / f"{language}.jsonl"
        with open(vocab_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        return entry

    def get_word(self, word: str, language: str) -> Optional[Dict]:
        """Obtiene información de una palabra"""
        return self.vocabularies[language].get(word)

    def search_similar(self, word: str, language: str, max_results: int = 5) -> List[Dict]:
        """Busca palabras similares (por coincidencia parcial)"""
        if language not in self.vocabularies:
            return []

        word_lower = word.lower()
        results = []

        for vocab_word, entry in self.vocabularies[language].items():
            if word_lower in vocab_word.lower() or vocab_word.lower() in word_lower:
                results.append(entry)

        return results[:max_results]

    def get_stats(self, language: str) -> Dict:
        """Estadísticas del vocabulario"""
        if language not in self.vocabularies:
            return {"total_words": 0}

        vocab = self.vocabularies[language]

        # Contar por categoría gramatical
        pos_counts = defaultdict(int)
        for entry in vocab.values():
            pos = entry.get('pos', 'unknown')
            pos_counts[pos] += 1

        # Nivel de dominio promedio
        avg_mastery = sum(e.get('mastery_level', 0) for e in vocab.values()) / len(vocab) if vocab else 0

        return {
            "total_words": len(vocab),
            "by_pos": dict(pos_counts),
            "average_mastery": avg_mastery,
            "languages": list(self.vocabularies.keys())
        }

    def review_word(self, word: str, language: str, success: bool = True):
        """Registra revisión de palabra (spaced repetition)"""
        entry = self.get_word(word, language)

        if entry:
            entry['last_reviewed'] = datetime.now().isoformat()

            if success:
                entry['mastery_level'] = min(5, entry.get('mastery_level', 0) + 1)
            else:
                entry['mastery_level'] = max(0, entry.get('mastery_level', 0) - 1)

            # Re-guardar
            self._save_vocabulary(language)

    def _save_vocabulary(self, language: str):
        """Guarda vocabulario completo de un idioma"""
        vocab_file = self.vocab_dir / f"{language}.jsonl"

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for entry in self.vocabularies[language].values():
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')


class GrammarLearner:
    """Aprende y gestiona reglas gramaticales"""

    def __init__(self):
        self.grammar_db = Path("./data/language/grammar.jsonl")
        self.grammar_db.parent.mkdir(parents=True, exist_ok=True)

        self.rules = defaultdict(list)  # {language: [rules]}
        self._load_grammar()

    def _load_grammar(self):
        """Carga reglas gramaticales"""
        if self.grammar_db.exists():
            with open(self.grammar_db, 'r', encoding='utf-8') as f:
                for line in f:
                    rule = json.loads(line)
                    self.rules[rule['language']].append(rule)

    def add_rule(
        self,
        language: str,
        category: str,  # e.g., "conjugation", "agreement", "word_order"
        rule_name: str,
        description: str,
        examples: List[Dict] = None,
        exceptions: List[str] = None
    ):
        """Añade regla gramatical"""

        rule = {
            "language": language,
            "category": category,
            "name": rule_name,
            "description": description,
            "examples": examples or [],
            "exceptions": exceptions or [],
            "learned_date": datetime.now().isoformat()
        }

        self.rules[language].append(rule)

        # Guardar
        with open(self.grammar_db, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rule, ensure_ascii=False) + '\n')

    def get_rules(self, language: str, category: str = None) -> List[Dict]:
        """Obtiene reglas gramaticales"""
        rules = self.rules.get(language, [])

        if category:
            rules = [r for r in rules if r['category'] == category]

        return rules


class MultilingualLearningManager:
    """Gestor principal del sistema de aprendizaje multilingüe"""

    def __init__(self):
        self.phonetic_learner = PhoneticLearner()
        self.vocab_builder = VocabularyBuilder()
        self.grammar_learner = GrammarLearner()

        self.supported_languages = ["es", "en", "fr", "de", "it", "pt"]
        self.active_languages = ["es"]  # Idiomas que está aprendiendo actualmente

    def add_language(self, language_code: str):
        """Comienza a aprender un nuevo idioma"""
        if language_code not in self.supported_languages:
            print(f"⚠️  Idioma no soportado: {language_code}")
            return False

        if language_code not in self.active_languages:
            self.active_languages.append(language_code)
            print(f"✅ Idioma añadido: {language_code}")

        return True

    def learn_word(
        self,
        word: str,
        language: str,
        definition: str = None,
        auto_phonetic: bool = True,
        **kwargs
    ):
        """
        Aprende una palabra completa (vocabulario + fonética)
        """
        # Añadir al vocabulario
        vocab_entry = self.vocab_builder.add_word(
            word=word,
            language=language,
            definition=definition or "",
            **kwargs
        )

        # Aprender fonética
        if auto_phonetic:
            phonetic = self.phonetic_learner.auto_learn_phonetic(word, language)

            if phonetic:
                print(f"📚 Palabra aprendida: {word}")
                print(f"   IPA: {phonetic['ipa']}")
                print(f"   Sílabas: {'-'.join(phonetic['syllables'])}")
            else:
                print(f"📚 Palabra aprendida: {word} (sin fonética)")

        return vocab_entry

    def create_language_dataset(
        self,
        language: str,
        focus: str = "vocabulary",
        num_examples: int = 20
    ) -> List[Dict]:
        """
        Genera dataset de entrenamiento para practicar idioma

        Args:
            language: Código de idioma
            focus: "vocabulary", "grammar", "pronunciation"
            num_examples: Cantidad de ejemplos
        """
        examples = []

        if focus == "vocabulary":
            # Ejemplos de vocabulario
            vocab = self.vocab_builder.vocabularies.get(language, {})

            for word, entry in list(vocab.items())[:num_examples]:
                # Pregunta: definición
                examples.append({
                    "instruction": f"¿Qué significa '{word}' en {language}?",
                    "input": "",
                    "output": entry['definition']
                })

                # Pregunta: uso en oración
                if entry.get('examples'):
                    examples.append({
                        "instruction": f"Usa '{word}' en una oración",
                        "input": "",
                        "output": entry['examples'][0]
                    })

        elif focus == "pronunciation":
            # Ejemplos de pronunciación
            phonetics = self.phonetic_learner.phonetics

            for key, entry in list(phonetics.items())[:num_examples]:
                if entry['language'] == language:
                    examples.append({
                        "instruction": f"¿Cómo se pronuncia '{entry['word']}'?",
                        "input": "",
                        "output": f"Se pronuncia {entry['ipa']} con {len(entry['syllables'])} sílabas: {'-'.join(entry['syllables'])}"
                    })

        elif focus == "grammar":
            # Ejemplos de gramática
            rules = self.grammar_learner.get_rules(language)

            for rule in rules[:num_examples]:
                if rule.get('examples'):
                    for example in rule['examples'][:2]:
                        examples.append({
                            "instruction": f"Explica la regla gramatical: {rule['name']}",
                            "input": "",
                            "output": f"{rule['description']}\n\nEjemplo: {example}"
                        })

        return examples

    def get_learning_progress(self, language: str) -> Dict:
        """Progreso de aprendizaje de un idioma"""
        return {
            "language": language,
            "vocabulary_stats": self.vocab_builder.get_stats(language),
            "phonetic_entries": sum(1 for k in self.phonetic_learner.phonetics.keys() if k.endswith(f"_{language}")),
            "grammar_rules": len(self.grammar_learner.get_rules(language))
        }

    def initialize_spanish_basics(self):
        """Inicializa conocimientos básicos de español"""
        print("🇪🇸 Inicializando español básico...")

        # Vocabulario fundamental
        basic_vocab = [
            {"word": "hola", "def": "saludo de bienvenida", "pos": "interjection"},
            {"word": "casa", "def": "edificio para habitar", "pos": "noun"},
            {"word": "perro", "def": "animal doméstico canino", "pos": "noun"},
            {"word": "comer", "def": "ingerir alimentos", "pos": "verb"},
            {"word": "rápido", "def": "que se mueve con velocidad", "pos": "adjective"},
        ]

        for item in basic_vocab:
            self.learn_word(item["word"], "es", item["def"], pos=item["pos"])

        # Reglas gramaticales básicas
        self.grammar_learner.add_rule(
            language="es",
            category="gender",
            rule_name="Género de sustantivos",
            description="Los sustantivos en español tienen género masculino o femenino. Generalmente, palabras terminadas en -o son masculinas y en -a son femeninas.",
            examples=[
                "el niño (masculino)",
                "la niña (femenino)"
            ],
            exceptions=["el mapa", "la mano"]
        )

        print("✅ Español básico inicializado")

    def initialize_english_basics(self):
        """Inicializa conocimientos básicos de inglés"""
        print("🇬🇧 Inicializando inglés básico...")

        # Vocabulario fundamental
        basic_vocab = [
            {"word": "hello", "def": "greeting", "pos": "interjection", "trans": {"es": "hola"}},
            {"word": "house", "def": "building for living", "pos": "noun", "trans": {"es": "casa"}},
            {"word": "dog", "def": "domestic canine animal", "pos": "noun", "trans": {"es": "perro"}},
            {"word": "eat", "def": "consume food", "pos": "verb", "trans": {"es": "comer"}},
            {"word": "fast", "def": "moving with speed", "pos": "adjective", "trans": {"es": "rápido"}},
        ]

        for item in basic_vocab:
            self.learn_word(
                item["word"],
                "en",
                item["def"],
                pos=item["pos"],
                translations=item.get("trans", {})
            )

        # Reglas gramaticales
        self.grammar_learner.add_rule(
            language="en",
            category="word_order",
            rule_name="Subject-Verb-Object order",
            description="English follows SVO (Subject-Verb-Object) word order in declarative sentences.",
            examples=[
                "I (S) eat (V) pizza (O)",
                "She (S) loves (V) music (O)"
            ]
        )

        print("✅ Inglés básico inicializado")


# CLI para testing
if __name__ == "__main__":
    print("🌍 Sistema de Aprendizaje Multilingüe de THAU\n")

    manager = MultilingualLearningManager()

    # Inicializar español
    manager.initialize_spanish_basics()

    print("\n" + "="*60)

    # Añadir inglés
    manager.add_language("en")
    manager.initialize_english_basics()

    print("\n" + "="*60 + "\n")

    # Ver progreso
    for lang in ["es", "en"]:
        progress = manager.get_learning_progress(lang)
        print(f"\n📊 Progreso en {lang}:")
        print(json.dumps(progress, indent=2, ensure_ascii=False))

    print("\n" + "="*60 + "\n")

    # Generar dataset de vocabulario
    print("📝 Generando dataset de vocabulario en español...")
    dataset = manager.create_language_dataset("es", focus="vocabulary", num_examples=5)

    for i, example in enumerate(dataset, 1):
        print(f"\n{i}. {example['instruction']}")
        print(f"   Respuesta: {example['output']}")
