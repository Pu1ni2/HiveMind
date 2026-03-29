import json
from .llm_client import get_client
from .prompts import SUB_AGENT_EXECUTION_TEMPLATE


class SubAgent:
    def __init__(self, config: dict, tool_executor=None):
        """
        Args:
            config: Agent spec from Phase 3-4 plan. Keys:
                    id, role, objective, model, tools_needed,
                    input, expected_output, context_from_agents
            tool_executor: function(tool_name, tool_args) -> str
                           Provided by LangGraph at runtime.
        """
        self.id = config["id"]
        self.role = config["role"]
        self.objective = config.get("objective", config.get("goal", ""))
        self.model = config["model"]
        self.tools = config.get("tools_needed", [])
        self.tool_schemas = config.get("tool_schemas", [])
        self.input_desc = config.get("input", "")
        self.expected_output = config.get("expected_output", "")
        self.tool_executor = tool_executor

    def run(self, task: str, context: str = "") -> str:
        messages = self._build_messages(task, context)

        # If no tools, single LLM call
        if not self.tool_schemas or not self.tool_executor:
            response = get_client().chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content or "(no output)"

        # Agentic tool-use loop
        while True:
            response = get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or "(no output)"

            messages.append(message)

            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"    [{self.role}] tool: {tool_name}")
                result = self.tool_executor(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    def _build_messages(self, task: str, context: str) -> list:
        tools_str = ", ".join(self.tools) if self.tools else "none"

        system = SUB_AGENT_EXECUTION_TEMPLATE.format(
            task_description=self.objective,
            input_content=self.input_desc or task,
            context_from_other_agents=context or "none",
            available_tools=tools_str,
            expected_output=self.expected_output or "Provide a thorough response.",
        )

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": task},
        ]
