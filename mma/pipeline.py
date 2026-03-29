from __future__ import annotations

import config
from llm import ChatLLM, ManagerLLM
from models import TopicMap


class ChatPipeline:
    def __init__(self) -> None:
        self.topic_map = TopicMap()
        self.manager = ManagerLLM()
        self.chat = ChatLLM()

    def step(self, user_input: str) -> str:
        """Run one full turn: classify topic, compress if needed, respond."""
        # Step 1: Classify topic
        recent = self.topic_map.current_segment.messages[-2:]
        same_topic, topic_label = self.manager.classify_topic(
            self.topic_map.current_segment.topic, recent, user_input,
        )

        # Step 2: Handle topic change and compression
        if not same_topic:
            self.topic_map.start_new_topic(topic_label)

            if self.topic_map.should_compress():
                oldest = self.topic_map.compress_oldest()
                if oldest is not None:
                    oldest.summary = self.manager.summarize_segment(oldest)
                    oldest.compressed = True

        # Step 3: Get response with full context
        response = self.chat.respond(self.topic_map, user_input)

        # Step 4: Record messages
        self.topic_map.current_segment.messages.append(
            {"role": "user", "content": user_input},
        )
        self.topic_map.current_segment.messages.append(
            {"role": "assistant", "content": response},
        )

        return response
