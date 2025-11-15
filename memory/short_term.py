"""Short-term memory for recent conversations."""

from collections import deque
from typing import List, Dict, Optional
from loguru import logger


class ShortTermMemory:
    """Buffer for recent conversation history.

    Implements a fixed-size FIFO buffer for maintaining context
    within the current session.
    """

    def __init__(self, max_size: int = 4096):
        """Initialize short-term memory.

        Args:
            max_size: Maximum number of tokens to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=20)  # Store last 20 messages
        self.current_size = 0

        logger.info(f"ShortTermMemory initialized with max_size={max_size}")

    def add(self, message: Dict[str, str]) -> None:
        """Add a message to short-term memory.

        Args:
            message: Dict with 'role' and 'content' keys
        """
        # Estimate token count (rough approximation)
        estimated_tokens = len(message['content'].split()) * 1.3

        # Remove old messages if needed
        while self.current_size + estimated_tokens > self.max_size and len(self.buffer) > 0:
            old_msg = self.buffer.popleft()
            old_tokens = len(old_msg['content'].split()) * 1.3
            self.current_size -= old_tokens

        self.buffer.append(message)
        self.current_size += estimated_tokens

        logger.debug(f"Added message to short-term memory (size: {self.current_size}/{self.max_size})")

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent messages.

        Args:
            n: Number of recent messages (None for all)

        Returns:
            List of recent messages
        """
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def get_context(self) -> List[Dict[str, str]]:
        """Get all messages as conversation context.

        Returns:
            List of all messages in buffer
        """
        return list(self.buffer)

    def clear(self) -> None:
        """Clear short-term memory."""
        self.buffer.clear()
        self.current_size = 0
        logger.info("Short-term memory cleared")

    def __len__(self) -> int:
        """Get number of messages in buffer."""
        return len(self.buffer)
