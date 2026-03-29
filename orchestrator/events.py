"""
EventBus — lightweight event system for streaming pipeline progress
to the frontend via WebSocket.

Usage inside orchestrator modules:
    from .events import emit
    emit("forge_tool_done", {"tool_name": "search_web", "success": True})

The pipeline sets the active bus before starting; all emit() calls
are no-ops if no bus is set (i.e. CLI mode still works).
"""

import queue
import threading
from datetime import datetime, timezone


class EventBus:
    """Thread-safe event bus backed by a stdlib Queue."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def emit(self, event_type: str, data: dict | None = None):
        self._queue.put({
            "type": event_type,
            "data": data or {},
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def get(self, timeout: float = 0.5):
        """Blocking get with timeout. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_empty(self) -> bool:
        return self._queue.empty()


# ── Thread-local bus storage — each pipeline thread has its own bus ──
_local = threading.local()


def set_bus(bus: EventBus | None):
    """Attach an EventBus to the current thread.

    Using thread-local storage means concurrent pipeline executions
    (e.g. multiple WebSocket sessions) each stream to their own bus
    without cross-talk.
    """
    _local.bus = bus


def emit(event_type: str, data: dict | None = None):
    bus: EventBus | None = getattr(_local, "bus", None)
    if bus is not None:
        bus.emit(event_type, data)
