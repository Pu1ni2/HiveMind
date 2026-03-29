import argparse

import config
from llm import ChatLLM, ManagerLLM
from models import TopicMap


def main():
    parser = argparse.ArgumentParser(description="Topic-aware context compression chat")
    parser.add_argument("--debug", action="store_true", help="Show topic map after each turn")
    args = parser.parse_args()

    if args.debug:
        config.DEBUG = True

    topic_map = TopicMap()
    manager = ManagerLLM()
    chat = ChatLLM()

    print("Chat started. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() in ("quit", "exit"):
            break

        if not user_input.strip():
            continue

        # Step 1: Classify topic
        recent = topic_map.current_segment.messages[-2:]
        same_topic, topic_label = manager.classify_topic(
            topic_map.current_segment.topic, recent, user_input
        )

        # Step 2: If topic changed, start new segment and check compression
        if not same_topic:
            topic_map.start_new_topic(topic_label)

            if topic_map.should_compress():
                oldest = topic_map.compress_oldest()
                if oldest is not None:
                    summary = manager.summarize_segment(oldest)
                    oldest.summary = summary
                    oldest.compressed = True

        # Step 3: Get chat response with full context
        response = chat.respond(topic_map, user_input)

        # Step 4: Append messages to current segment
        topic_map.current_segment.messages.append({"role": "user", "content": user_input})
        topic_map.current_segment.messages.append({"role": "assistant", "content": response})

        # Step 5: Print response
        print(f"\n{response}\n")

        # Step 6: Debug map
        if config.DEBUG:
            topic_map.print_map()

    # Always print final map on exit
    print("\nFinal state:")
    topic_map.print_map()


if __name__ == "__main__":
    main()
