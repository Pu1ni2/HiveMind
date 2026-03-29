"""
Tool Forge — dynamically generates Python tools from LLM-written code.

Flow:
  1.  Receive tool specs from the approved plan.
  2.  For each spec, ask gpt-4o to write a Python function.
  3.  Validate syntax (ast.parse) and run safety checks.
  4.  exec() the code to obtain the callable.
  5.  Wrap it as a LangChain StructuredTool.
  6.  On failure, retry once with the error fed back to the LLM.
"""

import ast
import json
import inspect
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OPENAI_API_KEY, FORGE_MODEL
from .prompts import TOOL_FORGE_PROMPT
from .events import emit
from .capabilities import CAPABILITY_NAMESPACE

_FORBIDDEN_MODULES = frozenset({
    "subprocess", "shutil", "ctypes", "importlib", "pickle",
    "shelve", "multiprocessing", "signal", "socket",
})

_FORBIDDEN_CALLS = frozenset({
    "os.system", "os.remove", "os.rmdir", "os.unlink",
    "shutil.rmtree", "__import__",
})

MAX_RETRIES = 1


def forge_tools_for_plan(plan: dict) -> dict[str, list[StructuredTool]]:
    """Forge tools for every agent in the plan.
    Returns {agent_id: [StructuredTool, ...]}

    Tools are forged in PARALLEL using a thread pool for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    model = ChatOpenAI(
        model=FORGE_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=2048,
    )

    agent_tools: dict[str, list[StructuredTool]] = {}
    tool_cache: dict[str, StructuredTool] = {}
    cache_lock = threading.Lock()

    # Collect all unique tool specs
    all_specs: list[tuple[str, dict]] = []  # (agent_id, tool_spec)
    seen_names: set[str] = set()

    for agent_spec in plan.get("agents", []):
        agent_id = agent_spec["id"]
        for tool_spec in agent_spec.get("tools_needed", []):
            name = tool_spec["name"]
            if name not in seen_names:
                seen_names.add(name)
                all_specs.append((agent_id, tool_spec))

    emit("forge_start", {"total_specs": len(all_specs)})

    def forge_one(agent_id: str, spec: dict) -> tuple[str, str, StructuredTool | None]:
        name = spec["name"]
        emit("forge_tool_start", {"tool_name": name, "agent_id": agent_id,
                                   "description": spec.get("description", "")})
        tool = _forge_single_tool(spec, model)
        success = tool is not None
        emit("forge_tool_done", {"tool_name": name, "agent_id": agent_id, "success": success})
        return name, agent_id, tool

    # Forge all tools in parallel (up to 6 concurrent)
    max_workers = min(len(all_specs), 6)
    if max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(forge_one, aid, spec): (aid, spec)
                for aid, spec in all_specs
            }
            for future in as_completed(futures):
                name, aid, tool = future.result()
                if tool is not None:
                    with cache_lock:
                        tool_cache[name] = tool

    # Assign cached tools to agents; insert stubs for any that failed
    for agent_spec in plan.get("agents", []):
        agent_id = agent_spec["id"]
        tools = []
        for tool_spec in agent_spec.get("tools_needed", []):
            name = tool_spec["name"]
            if name in tool_cache:
                tools.append(tool_cache[name])
            else:
                stub = _make_stub_tool(name, tool_spec.get("description", name))
                tools.append(stub)
                print(f"[FORGE] {agent_id}: stub inserted for failed tool '{name}'")
        agent_tools[agent_id] = tools
        print(f"[FORGE] {agent_id}: {len(tools)} tool(s) ready")

    emit("forge_complete", {"total_tools": sum(len(t) for t in agent_tools.values())})
    return agent_tools


def _make_stub_tool(name: str, description: str) -> StructuredTool:
    """Return an inert stub tool used when forge fails after all retries.

    The stub is callable and returns a descriptive error string so the
    agent can continue and explain the gap rather than crash.
    """
    def stub(**kwargs) -> str:  # noqa: ANN202
        return (
            f"[Tool '{name}' could not be forged — all generation attempts failed. "
            "Please complete this step using your general knowledge.]"
        )

    stub.__name__ = name
    stub.__doc__ = description or f"Stub for unforged tool: {name}"
    return StructuredTool.from_function(func=stub, name=name, description=stub.__doc__)


def _forge_single_tool(spec: dict, model: ChatOpenAI) -> StructuredTool | None:
    last_error = None

    for attempt in range(1 + MAX_RETRIES):
        code = _generate_code(spec, model, last_error)
        if code is None:
            return None

        try:
            ast.parse(code)
        except SyntaxError as exc:
            last_error = f"SyntaxError: {exc}"
            print(f"  [FORGE] {spec['name']} syntax error (attempt {attempt+1})")
            continue

        safe, reason = _is_safe(code)
        if not safe:
            last_error = f"Safety violation: {reason}"
            print(f"  [FORGE] {spec['name']} safety fail: {reason}")
            continue

        try:
            # Inject real capabilities so forged tools can do actual work
            namespace: dict[str, Any] = {**CAPABILITY_NAMESPACE}
            exec(code, namespace)  # noqa: S102
        except Exception as exc:
            last_error = f"Exec error: {exc}"
            print(f"  [FORGE] {spec['name']} exec error: {exc}")
            continue

        func = _extract_function(namespace, spec["name"])
        if func is None:
            last_error = "No callable function found in generated code"
            continue

        try:
            tool = StructuredTool.from_function(
                func=_make_safe_wrapper(func, spec["name"]),
                name=spec["name"],
                description=spec.get("description", spec["name"]),
            )
            print(f"  [FORGE] {spec['name']} created")
            return tool
        except Exception as exc:
            last_error = f"Tool wrapping error: {exc}"
            continue

    print(f"  [FORGE] {spec['name']} FAILED after {1 + MAX_RETRIES} attempts")
    return None


def _generate_code(spec: dict, model: ChatOpenAI, prev_error: str | None) -> str | None:
    user_content = (
        f"Generate this tool:\n"
        f"- Name: {spec['name']}\n"
        f"- Description: {spec.get('description', '')}\n"
        f"- Parameters: {json.dumps(spec.get('parameters', []))}\n"
        f"- Returns: {spec.get('returns', 'str')}\n"
    )
    if prev_error:
        user_content += f"\nYour previous attempt failed with:\n{prev_error}\nFix the issue and try again."

    messages = [
        SystemMessage(content=TOOL_FORGE_PROMPT),
        HumanMessage(content=user_content),
    ]
    try:
        response = model.invoke(messages)
        code = response.content.strip()
        return _strip_markdown(code)
    except Exception as exc:
        print(f"  [FORGE] LLM call failed for {spec['name']}: {exc}")
        return None


def _strip_markdown(code: str) -> str:
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _extract_function(namespace: dict, preferred_name: str):
    if preferred_name in namespace and callable(namespace[preferred_name]):
        return namespace[preferred_name]
    for name, obj in namespace.items():
        if name.startswith("_") or not callable(obj):
            continue
        if inspect.isfunction(obj):
            return obj
    return None


def _make_safe_wrapper(func, tool_name: str):
    def wrapper(*args, **kwargs) -> str:
        try:
            result = func(*args, **kwargs)
            return str(result)
        except Exception as exc:
            return f"[Tool {tool_name} error] {exc}"

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__ or f"Dynamically forged tool: {tool_name}"
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__annotations__ = getattr(func, "__annotations__", {})
    return wrapper


def _is_safe(code: str) -> tuple[bool, str]:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    return False, f"Forbidden import from: {node.module}"
        if isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name in ("eval", "exec", "__import__"):
                return False, f"Forbidden call: {call_name}"
            if call_name in _FORBIDDEN_CALLS:
                return False, f"Forbidden call: {call_name}"
    return True, ""


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        n = node.func
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return ".".join(reversed(parts))
    return ""
