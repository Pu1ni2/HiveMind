from __future__ import annotations

import json
import re

from config import CHAT_MODEL, MANAGER_MODEL
from llm_client import call_llm
from models import Segment, TopicMap
from prompts import CHAT_SYSTEM_PROMPT, SEGMENT_SUMMARY_PROMPT, TOPIC_CLASSIFIER_PROMPT


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

        prompt = TOPIC_CLASSIFIER_PROMPT.format(
            current_topic=current_topic,
            recent_text=recent_text,
            new_message=new_message,
        )

        raw = call_llm(
            MANAGER_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        ).strip()

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

        prompt = SEGMENT_SUMMARY_PROMPT.format(
            topic=segment.topic,
            formatted_messages=formatted,
        )

        return call_llm(
            MANAGER_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        ).strip()


class ChatLLM:
    def respond(self, topic_map: TopicMap, user_message: str) -> str:
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        for seg in topic_map.get_all_segments_for_context():
            if seg is topic_map.current_segment:
                messages.extend(seg.messages)
            elif seg.compressed:
                messages.append(
                    {
                        "role": "system",
                        "content": f"[Summary of topic '{seg.topic}']: {seg.summary}",
                    }
                )
            else:
                messages.extend(seg.messages)

        messages.append({"role": "user", "content": user_message})

        return call_llm(CHAT_MODEL, messages).strip()
