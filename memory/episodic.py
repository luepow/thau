"""Episodic memory for storing experiences and events."""

import sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from loguru import logger


class EpisodicMemory:
    """Episodic memory for storing temporal experiences.

    Stores experiences, interactions, and events with timestamps
    for later review and learning.
    """

    def __init__(self, db_path: str = "./data/memory/episodic.db"):
        """Initialize episodic memory.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"EpisodicMemory initialized at {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    episode_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON episodes(episode_type)
            """)

            conn.commit()

    def store_episode(
        self,
        episode_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: int = 1,
    ) -> int:
        """Store an episode.

        Args:
            episode_type: Type of episode (e.g., 'conversation', 'learning', 'error')
            content: Episode content
            metadata: Optional metadata
            importance: Importance score (1-10)

        Returns:
            Episode ID
        """
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO episodes (timestamp, episode_type, content, metadata, importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, episode_type, content, metadata_json, importance, timestamp))

            episode_id = cursor.lastrowid
            conn.commit()

        logger.debug(f"Stored episode {episode_id} of type {episode_type}")

        return episode_id

    def get_episodes(
        self,
        episode_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_importance: int = 1,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve episodes with filters.

        Args:
            episode_type: Filter by episode type
            start_time: Start of time range
            end_time: End of time range
            min_importance: Minimum importance score
            limit: Maximum number of results

        Returns:
            List of episodes
        """
        query = "SELECT * FROM episodes WHERE importance >= ?"
        params = [min_importance]

        if episode_type:
            query += " AND episode_type = ?"
            params.append(episode_type)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episode = dict(row)
            if episode['metadata']:
                episode['metadata'] = json.loads(episode['metadata'])
            episodes.append(episode)

        return episodes

    def get_recent_episodes(
        self,
        n: int = 10,
        episode_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get most recent episodes.

        Args:
            n: Number of episodes to retrieve
            episode_type: Optional filter by type

        Returns:
            List of recent episodes
        """
        return self.get_episodes(episode_type=episode_type, limit=n)

    def search_episodes(
        self,
        search_term: str,
        episode_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search episodes by content.

        Args:
            search_term: Term to search for
            episode_type: Optional filter by type
            limit: Maximum number of results

        Returns:
            List of matching episodes
        """
        query = "SELECT * FROM episodes WHERE content LIKE ?"
        params = [f"%{search_term}%"]

        if episode_type:
            query += " AND episode_type = ?"
            params.append(episode_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episode = dict(row)
            if episode['metadata']:
                episode['metadata'] = json.loads(episode['metadata'])
            episodes.append(episode)

        return episodes

    def get_count(self, episode_type: Optional[str] = None) -> int:
        """Get count of episodes.

        Args:
            episode_type: Optional filter by type

        Returns:
            Number of episodes
        """
        query = "SELECT COUNT(*) FROM episodes"
        params = []

        if episode_type:
            query += " WHERE episode_type = ?"
            params.append(episode_type)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            count = cursor.fetchone()[0]

        return count

    def clear(self, episode_type: Optional[str] = None) -> None:
        """Clear episodes.

        Args:
            episode_type: Optional filter by type (clears all if None)
        """
        if episode_type:
            query = "DELETE FROM episodes WHERE episode_type = ?"
            params = [episode_type]
        else:
            query = "DELETE FROM episodes"
            params = []

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, params)
            conn.commit()

        logger.warning(f"Cleared episodic memory{' for ' + episode_type if episode_type else ''}")
