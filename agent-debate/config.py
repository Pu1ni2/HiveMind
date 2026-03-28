import os
from dotenv import load_dotenv

load_dotenv()

# Fix broken SSL_CERT_FILE env var (common in conda on Windows)
ssl_cert = os.environ.get("SSL_CERT_FILE")
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ.pop("SSL_CERT_FILE")

# Debate settings
MAX_ROUNDS = 1

# OpenRouter settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-chat-v3-0324"
