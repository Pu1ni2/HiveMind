import os
from dotenv import load_dotenv

load_dotenv()

# Fix broken SSL_CERT_FILE env var (common in conda on Windows)
ssl_cert = os.environ.get("SSL_CERT_FILE")
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ.pop("SSL_CERT_FILE")

# Debate settings
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))

# API settings — OpenAI only
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"

# Model for debate phases (DA + Evaluator in Phase 1-4)
DEBATE_MODEL = "gpt-4o"

# Model for DA validation/compilation (Phase 6)
VALIDATION_MODEL = "gpt-4o"

# Model tiers for sub-agents (Phase 5)
TIER_TO_MODEL = {
    "FAST": "gpt-4o-mini",
    "BALANCED": "gpt-4o",
    "HEAVY": "gpt-4o",
}
