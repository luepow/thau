"""Vector store for semantic memory using ChromaDB."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
import uuid


class VectorStore:
    """Vector store using ChromaDB for semantic search.

    Stores text embeddings and enables similarity search for RAG.
    """

    def __init__(
        self,
        db_path: str = "./data/memory/chroma_db",
        collection_name: str = "my_llm_memory",
    ):
        """Initialize vector store.

        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # Create directory
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Long-term memory storage"},
        )

        logger.info(f"VectorStore initialized: {collection_name} at {db_path}")

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Store text with embeddings.

        Args:
            text: Text to store
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

        logger.debug(f"Stored document: {doc_id}")

        return doc_id

    def store_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Store multiple texts.

        Args:
            texts: List of texts
            metadatas: Optional list of metadata dicts
            doc_ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=doc_ids,
        )

        logger.info(f"Stored {len(texts)} documents")

        return doc_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of results with documents, metadatas, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                })

        return formatted_results

    def search_by_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            List with document (empty if not found)
        """
        try:
            result = self.collection.get(ids=[doc_id])

            if result['ids']:
                return [{
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {},
                }]
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")

        return []

    def update(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a document.

        Args:
            doc_id: Document ID
            text: New text
            metadata: New metadata
        """
        self.collection.update(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

        logger.debug(f"Updated document: {doc_id}")

    def delete(self, doc_id: str) -> None:
        """Delete a document.

        Args:
            doc_id: Document ID
        """
        self.collection.delete(ids=[doc_id])
        logger.debug(f"Deleted document: {doc_id}")

    def count(self) -> int:
        """Get number of documents in collection.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from collection."""
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        logger.warning("Vector store cleared")
