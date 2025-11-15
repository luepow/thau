"""Memory manager coordinating all memory systems."""

from typing import List, Dict, Optional, Any
from loguru import logger

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from config.base_config import get_config


class MemoryManager:
    """Unified memory manager coordinating all memory systems.

    Integrates:
    - Short-term memory for recent context
    - Long-term memory for persistent knowledge (RAG)
    - Episodic memory for experiences and learning
    """

    def __init__(self, config=None):
        """Initialize memory manager.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        # Initialize memory systems
        self.short_term = ShortTermMemory(
            max_size=self.config.SHORT_TERM_MEMORY_SIZE
        )

        self.long_term = LongTermMemory(
            db_path=self.config.MEMORY_DB_PATH,
            max_memories=self.config.MAX_LONG_TERM_MEMORIES,
        )

        self.episodic = EpisodicMemory(
            db_path=self.config.EPISODIC_DB_PATH
        )

        logger.info("MemoryManager initialized with all subsystems")

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store information across appropriate memory systems.

        Args:
            content: Content to remember
            memory_type: Type of memory (fact, conversation, experience)
            importance: Importance score (1-10)
            metadata: Optional metadata

        Returns:
            Memory ID from long-term storage
        """
        # Always store in long-term
        memory_id = self.long_term.store(
            content=content,
            metadata={
                **(metadata or {}),
                "type": memory_type,
                "importance": importance,
            }
        )

        # Store in episodic if important enough
        if importance >= 5:
            self.episodic.store_episode(
                episode_type=memory_type,
                content=content,
                metadata=metadata,
                importance=importance,
            )

        logger.info(f"Stored memory: {memory_id} (type: {memory_type}, importance: {importance})")

        return memory_id

    def recall(
        self,
        query: str,
        n_results: int = 5,
        include_short_term: bool = True,
    ) -> Dict[str, Any]:
        """Recall relevant information from memory.

        Args:
            query: Search query
            n_results: Number of long-term results
            include_short_term: Whether to include short-term context

        Returns:
            Dictionary with memories from different systems
        """
        results = {
            "short_term": [],
            "long_term": [],
            "episodic": [],
        }

        # Get short-term context
        if include_short_term:
            results["short_term"] = self.short_term.get_context()

        # Search long-term memory
        results["long_term"] = self.long_term.recall(
            query=query,
            n_results=n_results,
        )

        # Search episodic memory
        results["episodic"] = self.episodic.search_episodes(
            search_term=query,
            limit=min(n_results, 10),
        )

        logger.debug(f"Recalled {len(results['long_term'])} long-term and {len(results['episodic'])} episodic memories")

        return results

    def update_context(
        self,
        role: str,
        content: str,
        store_in_episodic: bool = True,
    ) -> None:
        """Update conversational context.

        Args:
            role: Message role (user/assistant)
            content: Message content
            store_in_episodic: Whether to store in episodic memory
        """
        # Add to short-term
        self.short_term.add({
            "role": role,
            "content": content,
        })

        # Optionally store in episodic
        if store_in_episodic:
            self.episodic.store_episode(
                episode_type="conversation",
                content=f"{role}: {content}",
                importance=3,
            )

    def get_conversation_history(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent conversation history.

        Args:
            n: Number of recent messages (None for all)

        Returns:
            List of messages
        """
        return self.short_term.get_recent(n)

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of memory stats
        """
        return {
            "short_term": {
                "messages": len(self.short_term),
                "size": self.short_term.current_size,
                "max_size": self.short_term.max_size,
            },
            "long_term": {
                "memories": self.long_term.get_count(),
                "max_memories": self.long_term.max_memories,
            },
            "episodic": {
                "episodes": self.episodic.get_count(),
                "conversations": self.episodic.get_count("conversation"),
                "learnings": self.episodic.get_count("learning"),
            },
        }


if __name__ == "__main__":
    # Test memory manager
    print("Testing MemoryManager...")

    manager = MemoryManager()

    # Store some memories
    manager.remember("Python is a programming language", memory_type="fact", importance=7)
    manager.remember("Machine learning uses data to train models", memory_type="fact", importance=8)

    # Update context
    manager.update_context("user", "Tell me about Python")
    manager.update_context("assistant", "Python is a versatile programming language")

    # Recall
    results = manager.recall("programming language")
    print(f"\nRecall results:")
    print(f"- Short-term: {len(results['short_term'])} messages")
    print(f"- Long-term: {len(results['long_term'])} memories")
    print(f"- Episodic: {len(results['episodic'])} episodes")

    # Get stats
    stats = manager.get_stats()
    print(f"\nMemory stats:")
    import json
    print(json.dumps(stats, indent=2))

    print("\nTest completed!")
