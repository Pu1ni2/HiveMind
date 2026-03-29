from __future__ import annotations

from dataclasses import dataclass, field

from config import MAX_ACTIVE_SEGMENTS


@dataclass
class Segment:
    topic: str
    messages: list[dict] = field(default_factory=list)
    compressed: bool = False
    summary: str | None = None

    def token_estimate(self) -> int:
        if self.compressed:
            return len(self.summary) // 4
        return sum(len(m["content"]) // 4 for m in self.messages)


class TopicMap:
    def __init__(self, initial_topic: str = "general"):
        self.segments: list[Segment] = []
        self.current_segment = Segment(topic=initial_topic)

    def start_new_topic(self, topic_label: str) -> None:
        self.segments.append(self.current_segment)
        self.current_segment = Segment(topic=topic_label)

    def should_compress(self) -> bool:
        uncompressed = sum(1 for s in self.segments if not s.compressed)
        return uncompressed > MAX_ACTIVE_SEGMENTS

    def compress_oldest(self) -> Segment | None:
        for seg in self.segments:
            if not seg.compressed:
                return seg
        return None

    def get_all_segments_for_context(self) -> list[Segment]:
        return self.segments + [self.current_segment]

    def print_map(self) -> None:
        all_segs = self.get_all_segments_for_context()
        total_tokens = 0
        uncompressed_total = 0

        print("\n--- Topic Map ---")
        for i, seg in enumerate(all_segs):
            is_current = seg is self.current_segment
            tokens = seg.token_estimate()
            total_tokens += tokens

            raw_tokens = sum(len(m["content"]) // 4 for m in seg.messages)
            uncompressed_total += raw_tokens

            if is_current:
                status = "CURRENT"
            elif seg.compressed:
                status = "COMPRESSED"
            else:
                status = "ACTIVE"

            print(
                f"  [{i}] {status:11s} | "
                f"topic: {seg.topic:20s} | "
                f"msgs: {len(seg.messages):3d} | "
                f"~{tokens} tokens"
            )

        print(f"  Total context tokens: ~{total_tokens}")
        print(f"  Without compression:  ~{uncompressed_total}")
        savings = uncompressed_total - total_tokens
        if savings > 0:
            print(f"  Savings:              ~{savings} tokens")
        print("--- End Map ---\n")
