"""
Sistema de Memoria Vectorizada Eficiente para THAU
Gestión optimizada de embeddings y recuperación de conocimiento
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import hashlib
import pickle


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using fallback embedding.")


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss not installed. Using numpy fallback.")


class EfficientVectorMemory:
    """
    Memoria vectorizada súper eficiente con:
    - FAISS para búsqueda rápida (si disponible)
    - Sentence transformers para embeddings de calidad
    - Fallback a numpy si no hay dependencias
    - Compresión y gestión inteligente de memoria
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "Flat",  # Changed from IVF to Flat to avoid clustering issues with small datasets
        memory_dir: Path = Path("./data/memory")
    ):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.index_type = index_type

        # Embeddings model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"📦 Cargando modelo de embeddings: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_model = None
            print("⚠️  Usando embeddings básicos (TF-IDF like)")

        # Vector index
        self.index = None
        self.metadata = []  # Lista de metadatos para cada vector
        self.id_to_idx = {}  # Mapeo de ID a índice

        self._initialize_index()
        self._load_memory()

    def _initialize_index(self):
        """Inicializa el índice vectorial"""
        if FAISS_AVAILABLE:
            if self.index_type == "IVF":
                # IVF (Inverted File Index) - Rápido para búsquedas
                # Usa cuantización para comprimir
                quantizer = faiss.IndexFlatL2(self.dimension)
                n_centroids = 100  # Número de clusters
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_centroids)
            elif self.index_type == "HNSW":
                # HNSW (Hierarchical Navigable Small World) - Muy rápido
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                # Flat (simple L2 distance) - Exacto pero lento
                self.index = faiss.IndexFlatL2(self.dimension)

            print(f"✅ Índice FAISS inicializado: {self.index_type}")
        else:
            # Fallback: numpy
            self.vectors = np.array([]).reshape(0, self.dimension)
            print("✅ Índice numpy inicializado (fallback)")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Obtiene embedding de un texto"""
        if self.embedding_model:
            # Sentence transformer (alta calidad)
            return self.embedding_model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: hash-based embedding simple
            return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Embedding simple basado en hashing (fallback)"""
        # Crear vector de frecuencias de palabras
        words = text.lower().split()
        embedding = np.zeros(self.dimension)

        for i, word in enumerate(words[:self.dimension]):
            # Hash de palabra a posición en vector
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            embedding[idx] += 1.0

        # Normalizar
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def add(
        self,
        text: str,
        metadata: Dict = None,
        memory_id: str = None
    ) -> str:
        """
        Añade texto a la memoria vectorizada

        Returns:
            ID del vector añadido
        """
        # Generar ID si no se proporciona
        if memory_id is None:
            memory_id = hashlib.md5(
                f"{text}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

        # Obtener embedding
        embedding = self._get_embedding(text)

        # Guardar metadata
        meta = metadata or {}
        meta.update({
            "id": memory_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text)
        })

        # Añadir al índice
        if FAISS_AVAILABLE:
            # FAISS requiere entrenar IVF antes del primer add
            if self.index_type == "IVF" and not self.index.is_trained:
                # Entrenar con este vector (necesita al menos n_centroids vectores)
                # Por ahora, entrenar con vector duplicado
                train_data = np.array([embedding] * 100).astype('float32')
                self.index.train(train_data)

            # Añadir vector
            vector_array = np.array([embedding]).astype('float32')
            self.index.add(vector_array)
            idx = self.index.ntotal - 1
        else:
            # Numpy fallback
            if len(self.vectors) == 0:
                self.vectors = embedding.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, embedding])
            idx = len(self.vectors) - 1

        self.metadata.append(meta)
        self.id_to_idx[memory_id] = idx

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Busca los k vectores más similares

        Returns:
            Lista de resultados con texto, score y metadata
        """
        if len(self.metadata) == 0:
            return []

        # Obtener embedding de query
        query_embedding = self._get_embedding(query)

        if FAISS_AVAILABLE:
            # Búsqueda con FAISS
            query_array = np.array([query_embedding]).astype('float32')

            # Ajustar k si hay menos vectores
            k_search = min(k, self.index.ntotal)

            distances, indices = self.index.search(query_array, k_search)

            # Convertir distancia L2 a similitud (0-1)
            # Score = 1 / (1 + distance)
            scores = 1.0 / (1.0 + distances[0])
            indices = indices[0]
        else:
            # Búsqueda con numpy (similitud coseno)
            # Normalizar vectores
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)

            # Similitud coseno
            scores = np.dot(vectors_norm, query_norm)

            # Top k índices
            k_search = min(k, len(scores))
            indices = np.argsort(scores)[::-1][:k_search]
            scores = scores[indices]

        # Construir resultados
        results = []
        for idx, score in zip(indices, scores):
            if score >= min_score:
                meta = self.metadata[idx].copy()
                meta['score'] = float(score)
                results.append(meta)

        return results

    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        """Obtiene un vector por su ID"""
        if memory_id in self.id_to_idx:
            idx = self.id_to_idx[memory_id]
            return self.metadata[idx]
        return None

    def delete(self, memory_id: str) -> bool:
        """
        Elimina un vector de la memoria
        Nota: FAISS no soporta eliminación directa, marcaremos como deleted
        """
        if memory_id in self.id_to_idx:
            idx = self.id_to_idx[memory_id]
            self.metadata[idx]['deleted'] = True
            return True
        return False

    def _save_memory(self):
        """Guarda memoria a disco"""
        # Guardar metadata
        metadata_file = self.memory_dir / "metadata.jsonl"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for meta in self.metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')

        # Guardar índice
        if FAISS_AVAILABLE:
            index_file = self.memory_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
        else:
            vectors_file = self.memory_dir / "vectors.npy"
            np.save(vectors_file, self.vectors)

        # Guardar mapeo ID -> idx
        mapping_file = self.memory_dir / "id_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.id_to_idx, f)

    def _load_memory(self):
        """Carga memoria desde disco"""
        metadata_file = self.memory_dir / "metadata.jsonl"

        if not metadata_file.exists():
            print("ℹ️  No hay memoria previa para cargar")
            return

        # Cargar metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = [json.loads(line) for line in f]

        # Cargar índice
        if FAISS_AVAILABLE:
            index_file = self.memory_dir / "faiss_index.bin"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
        else:
            vectors_file = self.memory_dir / "vectors.npy"
            if vectors_file.exists():
                self.vectors = np.load(vectors_file)

        # Cargar mapeo
        mapping_file = self.memory_dir / "id_mapping.pkl"
        if mapping_file.exists():
            with open(mapping_file, 'rb') as f:
                self.id_to_idx = pickle.load(f)

        print(f"✅ Memoria cargada: {len(self.metadata)} vectores")

    def save(self):
        """Guarda cambios a disco"""
        self._save_memory()
        print(f"💾 Memoria guardada: {len(self.metadata)} vectores")

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la memoria"""
        active_count = sum(1 for m in self.metadata if not m.get('deleted', False))

        return {
            "total_vectors": len(self.metadata),
            "active_vectors": active_count,
            "deleted_vectors": len(self.metadata) - active_count,
            "dimension": self.dimension,
            "index_type": self.index_type if FAISS_AVAILABLE else "numpy",
            "memory_size_mb": self._estimate_memory_size()
        }

    def _estimate_memory_size(self) -> float:
        """Estima tamaño en memoria (MB)"""
        # Vectores
        if FAISS_AVAILABLE:
            vector_size = self.index.ntotal * self.dimension * 4  # float32 = 4 bytes
        else:
            vector_size = self.vectors.nbytes

        # Metadata (aproximado)
        metadata_size = len(json.dumps(self.metadata).encode('utf-8'))

        total_bytes = vector_size + metadata_size
        return total_bytes / (1024 * 1024)  # Convertir a MB

    def cleanup(self, max_vectors: int = 10000):
        """
        Limpia memoria eliminando vectores antiguos si excede límite
        Mantiene los más recientes y relevantes
        """
        if len(self.metadata) <= max_vectors:
            return

        print(f"🧹 Limpiando memoria ({len(self.metadata)} -> {max_vectors} vectores)")

        # Ordenar por timestamp (más recientes primero)
        indexed_meta = [(i, m) for i, m in enumerate(self.metadata) if not m.get('deleted', False)]
        indexed_meta.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)

        # Mantener solo los más recientes
        keep_indices = set(i for i, _ in indexed_meta[:max_vectors])

        # Marcar resto como deleted
        for i, meta in enumerate(self.metadata):
            if i not in keep_indices:
                meta['deleted'] = True

        print(f"✅ Limpieza completada: {len(keep_indices)} vectores activos")


# Testing
if __name__ == "__main__":
    print("Probando Memoria Vectorizada Eficiente...")

    memory = EfficientVectorMemory()

    # Añadir algunos textos
    texts = [
        "Python es un lenguaje de programación interpretado",
        "JavaScript se usa principalmente para desarrollo web",
        "Machine learning es una rama de la inteligencia artificial",
        "Los algoritmos de ordenamiento organizan datos",
        "Las bases de datos relacionales usan SQL",
    ]

    print("\nAñadiendo textos a memoria...")
    for text in texts:
        memory.add(text)

    print(f"\nEstadísticas: {json.dumps(memory.get_stats(), indent=2)}")

    # Buscar
    query = "programación web"
    print(f"\nBuscando: '{query}'")
    results = memory.search(query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Texto: {result['text']}")

    # Guardar
    memory.save()
