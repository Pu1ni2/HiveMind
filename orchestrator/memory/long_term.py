"""Long-term memory — extracts patterns from episodes and retrieves relevant context."""

from __future__ import annotations
import uuid

from .types import Episode, MemoryEntry
from .store import MemoryStore
from .embeddings import SemanticIndex


class LongTermMemory:
    """Learns from past runs and provides context for future ones."""

    def __init__(self, store: MemoryStore, index: SemanticIndex):
        self.store = store
        self.index = index

    def record_episode(self, episode: Episode):
        """Persist an episode and extract reusable learnings."""
        self.store.save_episode(episode)
        self.index.index_episode(episode)
        self._extract_learnings(episode)

    def _extract_learnings(self, ep: Episode):
        """Distill an episode into reusable memory entries."""
        agents = ep.plan.get("agents", [])
        roles = [a.get("role", "") for a in agents]

        # Quality signal — only record plan patterns for episodes with a known outcome
        # (either user-scored or issue-free).  Avoids polluting memory with failed plans.
        score = ep.success_score or 0.0
        has_issues = bool(ep.known_issues)
        is_successful = score >= 7.0 or (score == 0.0 and not has_issues)

        # 1. Plan pattern — only for successful runs
        if roles and is_successful:
            score_note = f" | user score: {score}/10" if score else ""
            entry = MemoryEntry(
                entry_id=uuid.uuid4().hex[:12],
                memory_type="plan_pattern",
                content=(
                    f"For a '{ep.task_domain}' task at {ep.task_complexity} complexity, "
                    f"a {len(agents)}-agent plan succeeded{score_note}: {', '.join(roles)}. "
                    f"Task was: {ep.task[:200]}"
                ),
                context={
                    "domain": ep.task_domain,
                    "complexity": ep.task_complexity,
                    "agent_count": len(agents),
                    "roles": roles,
                    "success_score": score,
                },
                source_episode_id=ep.episode_id,
            )
            self.store.save_memory_entry(entry)
            self.index.index_memory_entry(entry)

        # 2. Lessons from known issues — deduplicate against existing lessons
        existing_lessons = {
            e.content for e in self.store.get_entries_by_type("lesson_learned", limit=100)
        }
        for issue in ep.known_issues:
            if not issue or len(issue) < 10:
                continue
            content = f"Issue in '{ep.task_domain}' task: {issue}"
            # Skip near-duplicate lessons (same issue text already stored)
            if any(issue[:60] in existing for existing in existing_lessons):
                continue
            entry = MemoryEntry(
                entry_id=uuid.uuid4().hex[:12],
                memory_type="lesson_learned",
                content=content,
                context={"domain": ep.task_domain, "task_preview": ep.task[:150]},
                source_episode_id=ep.episode_id,
            )
            self.store.save_memory_entry(entry)
            self.index.index_memory_entry(entry)
            existing_lessons.add(content)

        # 3. Agent strategies — record role + tools + effectiveness signal
        for agent_spec in agents:
            agent_id = agent_spec.get("id", "")
            output_info = ep.agent_outputs.get(agent_id, {})
            output_text = output_info.get("output", "") if isinstance(output_info, dict) else str(output_info)
            if len(output_text) > 50:
                tools = [t.get("name", "") for t in agent_spec.get("tools_needed", [])]
                effectiveness = "successful" if is_successful else "attempted (issues found)"
                entry = MemoryEntry(
                    entry_id=uuid.uuid4().hex[:12],
                    memory_type="agent_strategy",
                    content=(
                        f"Agent '{agent_spec.get('role', '')}' ({effectiveness}) "
                        f"with tools [{', '.join(tools)}] "
                        f"for objective: {agent_spec.get('objective', '')[:200]}"
                    ),
                    context={
                        "role": agent_spec.get("role", ""),
                        "tools": tools,
                        "output_length": len(output_text),
                        "successful": is_successful,
                    },
                    source_episode_id=ep.episode_id,
                )
                self.store.save_memory_entry(entry)
                self.index.index_memory_entry(entry)

    def get_context_for_planning(self, task: str, max_chars: int = 2000) -> str:
        """Retrieve relevant past experience for the DA."""
        parts = []

        # Semantic search for similar past tasks
        similar = self.index.search_similar_tasks(task, n_results=3)
        for hit in similar:
            ep_id = hit.get("metadata", {}).get("episode_id", "")
            ep = self.store.get_episode(ep_id) if ep_id else None
            if ep:
                roles = [a.get("role", "") for a in ep.plan.get("agents", [])]
                score_str = f" (user score: {ep.success_score}/10)" if ep.success_score else ""
                parts.append(
                    f"- Similar task: \"{ep.task[:120]}\"\n"
                    f"  Plan: {len(roles)} agents — {', '.join(roles)}{score_str}\n"
                    f"  Domain: {ep.task_domain} | Complexity: {ep.task_complexity}"
                )

        # Relevant lessons
        lessons = self.index.search_relevant_memories(task, memory_type="lesson_learned", n_results=3)
        for hit in lessons:
            parts.append(f"- Lesson: {hit['content'][:200]}")

        # Plan patterns from SQLite (fallback if vector search is unavailable)
        if not parts:
            patterns = self.store.get_entries_by_type("plan_pattern", limit=5)
            for p in patterns:
                parts.append(f"- Past pattern: {p.content[:200]}")

        if not parts:
            return ""

        text = "\n".join(parts)
        return text[:max_chars]

    def get_context_for_agent(self, role: str, objective: str, max_chars: int = 1000) -> str:
        """Retrieve relevant past experience for a specific agent role."""
        query = f"{role}: {objective}"
        results = self.index.search_relevant_memories(query, memory_type="agent_strategy", n_results=3)

        if not results:
            # Fallback to SQLite
            entries = self.store.get_entries_by_type("agent_strategy", limit=5)
            results = [{"content": e.content} for e in entries if role.lower() in e.content.lower()][:3]

        if not results:
            return ""

        parts = ["Relevant past agent experience:"]
        for hit in results:
            content = hit.get("content", hit) if isinstance(hit, dict) else str(hit)
            parts.append(f"- {content[:250]}")

        text = "\n".join(parts)
        return text[:max_chars]

    def get_context_for_compiler(self, task: str, max_chars: int = 1000) -> str:
        """Retrieve relevant past compilation strategies."""
        results = self.index.search(task, n_results=3, filter_type="episode_task")

        if not results:
            return ""

        parts = ["Past compilation notes:"]
        for hit in results:
            ep_id = hit.get("metadata", {}).get("episode_id", "")
            ep = self.store.get_episode(ep_id) if ep_id else None
            if ep and ep.coverage_report:
                quality = ep.coverage_report.get("quality_assessment", "")
                if quality:
                    parts.append(f"- Similar task coverage: {quality[:200]}")

        if len(parts) <= 1:
            return ""

        text = "\n".join(parts)
        return text[:max_chars]

    def record_user_feedback(self, episode_id: str, feedback: str, score: float):
        """Record user feedback and create a learning entry."""
        self.store.update_episode_feedback(episode_id, feedback, score)

        ep = self.store.get_episode(episode_id)
        if ep and feedback:
            entry = MemoryEntry(
                entry_id=uuid.uuid4().hex[:12],
                memory_type="user_preference",
                content=f"User feedback on '{ep.task_domain}' task (score {score}/10): {feedback}",
                context={"domain": ep.task_domain, "score": score, "task_preview": ep.task[:150]},
                source_episode_id=episode_id,
            )
            self.store.save_memory_entry(entry)
            self.index.index_memory_entry(entry)
