from __future__ import annotations

import json
import re

import ollama

from config import CHAT_MODEL, MANAGER_MODEL
from models import Segment, TopicMap


class ManagerLLM:
    def classify_topic(
        self,
        current_topic: str,
        recent_messages: list[dict],
        new_message: str,
    ) -> tuple[bool, str]:
        recent_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent_messages
        )

        prompt = (
            "You are a conversation topic classifier. Given the current topic "
            "and a new user message, determine if the message continues the "
            "current topic or starts a new one.\n\n"
            f"Current topic: {current_topic}\n"
            f"Recent context: {recent_text}\n"
            f"New message: {new_message}\n\n"
            'Respond ONLY with JSON, no other text:\n'
            '{"same_topic": true/false, "topic_label": "short 2-4 word label for the topic"}'
        )

        response = ollama.chat(
            model=MANAGER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        raw = response.message.content.strip()
        # Extract JSON from the response in case the model wraps it
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
            same_topic = bool(data.get("same_topic", True))
            topic_label = str(data.get("topic_label", current_topic))
        except (json.JSONDecodeError, KeyError):
            same_topic = True
            topic_label = current_topic

        return same_topic, topic_label

    def summarize_segment(self, segment: Segment) -> str:
        formatted = "\n".join(
            f"{m['role']}: {m['content']}" for m in segment.messages
        )

        prompt = (
            "Summarize the following conversation segment in 2-3 sentences. "
            "Preserve key facts, decisions, and any information the user might "
            "reference later.\n\n"
            f"Topic: {segment.topic}\n"
            f"Messages:\n{formatted}\n\n"
            "Summary:"
        )

        response = ollama.chat(
            model=MANAGER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )

        return response.message.content.strip()


class ChatLLM:
    def respond(self, topic_map: TopicMap, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You have context from previous "
                    "conversation topics provided as summaries, and recent "
                    "conversation in full."
                ),
            }
        ]

        for seg in topic_map.get_all_segments_for_context():
            if seg is topic_map.current_segment:
                # Current segment messages added in order
                messages.extend(seg.messages)
            elif seg.compressed:
                messages.append(
                    {
                        "role": "system",
                        "content": f"[Summary of topic '{seg.topic}']: {seg.summary}",
                    }
                )
            else:
                # Uncompressed past segment: full messages
                messages.extend(seg.messages)

        messages.append({"role": "user", "content": user_message})

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
        )

        return response.message.content.strip()
