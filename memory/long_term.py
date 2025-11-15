"""Long-term memory using vector store for RAG."""

from typing import List, Dict, Optional, Any
from loguru import logger

from memory.vector_store import VectorStore


class LongTermMemory:
    """Long-term memory using vector embeddings for retrieval.

    Stores important information and memories that persist across sessions.
    Uses semantic search to retrieve relevant memories.
    """

    def __init__(
        self,
        db_path: str = "./data/memory/chroma_db",
        collection_name: str = "long_term_memory",
        max_memories: int = 10000,
    ):
        """Initialize long-term memory.

        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            max_memories: Maximum number of memories to store
        """
        self.vector_store = VectorStore(
            db_path=db_path,
            collection_name=collection_name,
        )
        self.max_memories = max_memories

        logger.info(f"LongTermMemory initialized at {db_path}")

    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Store a memory in long-term storage.

        Args:
            content: Memory content
            metadata: Optional metadata
            memory_id: Optional memory ID

        Returns:
            Memory ID
        """
        return self.vector_store.store(
            text=content,
            metadata=metadata,
            doc_id=memory_id,
        )

    def recall(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant memories
        """
        return self.vector_store.search(
            query=query,
            n_results=n_results,
            where=filter_metadata,
        )

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory dict or None if not found
        """
        results = self.vector_store.search_by_id(memory_id)
        return results[0] if results else None

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing memory.

        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
        """
        if content:
            self.vector_store.update(memory_id, content, metadata)
        logger.info(f"Updated memory: {memory_id}")

    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Args:
            memory_id: Memory ID to delete
        """
        self.vector_store.delete(memory_id)
        logger.info(f"Deleted memory: {memory_id}")

    def get_count(self) -> int:
        """Get total number of memories.

        Returns:
            Number of memories stored
        """
        return self.vector_store.count()

    def clear(self) -> None:
        """Clear all long-term memories."""
        self.vector_store.clear()
        logger.warning("All long-term memories cleared")
