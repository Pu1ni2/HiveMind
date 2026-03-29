import argparse

import config
from pipeline import ChatPipeline


def main():
    parser = argparse.ArgumentParser(description="Topic-aware context compression chat")
    parser.add_argument("--debug", action="store_true", help="Show topic map after each turn")
    args = parser.parse_args()

    if args.debug:
        config.DEBUG = True

    pipeline = ChatPipeline()

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

        response = pipeline.step(user_input)

        print(f"\n{response}\n")

        if config.DEBUG:
            pipeline.topic_map.print_map()

    print("\nFinal state:")
    pipeline.topic_map.print_map()


if __name__ == "__main__":
    main()
