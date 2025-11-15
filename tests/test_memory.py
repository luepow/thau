"""Tests for memory systems."""

import pytest
from memory.manager import MemoryManager
from memory.short_term import ShortTermMemory


def test_short_term_memory():
    """Test short-term memory."""
    stm = ShortTermMemory(max_size=1000)

    stm.add({"role": "user", "content": "Hello"})
    stm.add({"role": "assistant", "content": "Hi there!"})

    assert len(stm) == 2

    recent = stm.get_recent(1)
    assert len(recent) == 1
    assert recent[0]["role"] == "assistant"


def test_memory_manager():
    """Test memory manager."""
    manager = MemoryManager()

    # Store memory
    memory_id = manager.remember("Python is a programming language", importance=7)
    assert memory_id is not None

    # Update context
    manager.update_context("user", "Tell me about Python")

    # Get stats
    stats = manager.get_stats()
    assert "short_term" in stats
    assert "long_term" in stats
    assert "episodic" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
