TOPIC_CLASSIFIER_PROMPT = """\
You are a conversation topic classifier. Given the current topic \
and a new user message, determine if the message continues the \
current topic or starts a new one.

Current topic: {current_topic}
Recent context: {recent_text}
New message: {new_message}

Respond ONLY with JSON, no other text:
{{"same_topic": true/false, "topic_label": "short 2-4 word label for the topic"}}"""

SEGMENT_SUMMARY_PROMPT = """\
Summarize the following conversation segment in 2-3 sentences. \
Preserve key facts, decisions, and any information the user might \
reference later.

Topic: {topic}
Messages:
{formatted_messages}

Summary:"""

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You have context from previous "
    "conversation topics provided as summaries, and recent "
    "conversation in full."
)
