import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# --------------- Models ---------------
PLANNER_MODEL = "gpt-4o"
EVALUATOR_MODEL = "gpt-4o"
COMPILER_MODEL = "gpt-4o"
FORGE_MODEL = "gpt-4o-mini"

TIER_TO_MODEL = {
    "FAST": "gpt-4o-mini",
    "BALANCED": "gpt-4o",
    "HEAVY": "gpt-4o",
}

# --------------- Debate ---------------
MAX_DEBATE_ROUNDS = 3

# --------------- Agents ---------------
MAX_AGENTS = 8
MAX_AGENT_STEPS = 25

# --------------- MCP ---------------
# Format: {"server_name": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-xxx"]}}
MCP_SERVERS = {}

# Load MCP config from env if present
mcp_config_path = os.getenv("MCP_CONFIG_PATH")
if mcp_config_path and os.path.exists(mcp_config_path):
    import json
    with open(mcp_config_path) as f:
        MCP_SERVERS = json.load(f)
