"""
HIVEMIND Memory Management System.

Provides persistent memory across pipeline runs:
- Episodic memory: full execution records
- Long-term memory: learned patterns and lessons
- Short-term shared workspace: agent collaboration within a run
- Semantic search: find similar past tasks and relevant experience
"""

from __future__ import annotations
import os

from .types import Episode, MemoryEntry, SharedMemoryItem
from .store import MemoryStore
from .embeddings import SemanticIndex
from .long_term import LongTermMemory
from .short_term import SharedWorkspace
from .episodic import EpisodeRecorder


class MemoryManager:
    """Single entry point for all memory operations.

    Created once at server startup, passed into run_task().
    """

    def __init__(self, data_dir: str = "data"):
        os.makedirs(data_dir, exist_ok=True)
        self.store = MemoryStore(db_path=os.path.join(data_dir, "hivemind_memory.db"))
        self.index = SemanticIndex(
            persist_dir=os.path.join(data_dir, "hivemind_vectors"),
            store=self.store,
        )
        self.long_term = LongTermMemory(self.store, self.index)

        # Per-run state (created fresh in begin_run)
        self.workspace: SharedWorkspace | None = None
        self.recorder: EpisodeRecorder | None = None

    def begin_run(self, task: str):
        """Called at the start of each pipeline run."""
        self.workspace = SharedWorkspace()
        self.recorder = EpisodeRecorder()
        self.recorder.start(task)

    def end_run(self, result: dict) -> Episode:
        """Called at the end of each pipeline run. Persists the episode."""
        episode = self.recorder.finalize(result)
        self.long_term.record_episode(episode)
        return episode

    # ── Delegation ─────────────────────────────────────────────────

    def get_planning_context(self, task: str) -> str:
        return self.long_term.get_context_for_planning(task)

    def get_agent_context(self, role: str, objective: str) -> str:
        return self.long_term.get_context_for_agent(role, objective)

    def get_compiler_context(self, task: str) -> str:
        return self.long_term.get_context_for_compiler(task)

    def get_context_for_compiler(self, task: str) -> str:
        """Alias used by compiler.py."""
        return self.long_term.get_context_for_compiler(task)

    def get_workspace(self) -> SharedWorkspace | None:
        return self.workspace

    def record_feedback(self, episode_id: str, feedback: str, score: float):
        self.long_term.record_user_feedback(episode_id, feedback, score)

    def get_episode_history(self, limit: int = 20, domain: str | None = None) -> list[Episode]:
        return self.store.list_episodes(limit=limit, domain=domain)

    def search_memory(self, query: str, n_results: int = 5) -> list[dict]:
        return self.index.search(query, n_results=n_results)


__all__ = [
    "MemoryManager",
    "Episode",
    "MemoryEntry",
    "SharedMemoryItem",
    "SharedWorkspace",
]
