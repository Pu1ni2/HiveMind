"""
Quick Actions — bypass the full pipeline for simple, direct tasks.

When the DA detects a task that maps directly to built-in tools
(send email, post to Slack, create calendar event, etc.), it
executes immediately without debate, forge, or agent creation.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OPENAI_API_KEY, PLANNER_MODEL
from .events import emit
from .capabilities import (
    search_web, scrape_url, save_file, read_file, list_files,
    fetch_json, compute, create_html_form,
)
from .integrations import (
    send_email, send_slack_message, create_calendar_event,
    parse_resume, read_pdf, create_spreadsheet, send_webhook,
    create_kanban_board,
)
from .utils import call_llm, parse_json_response


QUICK_DETECT_PROMPT = """\
You are a task classifier. Determine if a user task can be handled
DIRECTLY with one or a few built-in tool calls, or if it requires
a full multi-agent pipeline.

BUILT-IN TOOLS AVAILABLE:
- send_email(to, subject, body, cc="", html=False) — send an email
- send_slack(message, channel="") — post to Slack
- create_calendar_event(title, start, end="", description="", location="", attendees="") — create .ics event
- search_web(query, max_results=8) — web search
- scrape_url(url, max_chars=8000) — fetch webpage text
- save_file(filename, content) — save a file
- create_form(filename, title, fields, submit_action="#") — create HTML form
- create_spreadsheet(filename, headers, rows) — create CSV/Excel
- create_kanban_board(title, columns) — create interactive board
- parse_resume(text) — analyze resume text
- read_pdf(filepath) — extract text from PDF
- send_webhook(url, payload) — HTTP POST to any URL
- compute(code_str) — execute Python calculations

RULES:
- If the task is a SINGLE action (send one email, create one event, post one message, do one search), classify as "quick".
- If the task needs 2-3 simple sequential actions (search + send email, create event + send slack), classify as "quick".
- If the task needs research, analysis, planning, multiple agents, or complex multi-step work, classify as "full_pipeline".
- When in doubt, choose "full_pipeline".

Return JSON only:
{
  "mode": "quick" | "full_pipeline",
  "reason": "<one line why>",
  "actions": [
    {
      "tool": "<tool_name>",
      "params": { "<param>": "<value>", ... }
    }
  ]
}

For "full_pipeline" mode, actions should be an empty list.
For "quick" mode, actions must list every tool call needed with exact parameters.

IMPORTANT:
- For dates/times, use format "YYYY-MM-DD HH:MM"
- For email body, write the full email text
- For Slack messages, write the full message
- Extract ALL required info from the user's task
"""


# Map tool names to actual functions
_TOOL_MAP = {
    "send_email": send_email,
    "send_slack": send_slack_message,
    "create_calendar_event": create_calendar_event,
    "search_web": search_web,
    "scrape_url": scrape_url,
    "save_file": save_file,
    "create_form": create_html_form,
    "create_spreadsheet": create_spreadsheet,
    "create_kanban_board": create_kanban_board,
    "parse_resume": parse_resume,
    "read_pdf": read_pdf,
    "send_webhook": send_webhook,
    "compute": compute,
}


def try_quick_execute(task: str) -> dict | None:
    """Attempt to handle a task via quick action.

    Returns a result dict if the task was handled, or None if it
    should go through the full pipeline.
    """
    emit("quick_detect_start", {"task": task[:100]})

    try:
        decision = call_llm(
            PLANNER_MODEL,
            QUICK_DETECT_PROMPT,
            f"Task:\n{task}",
            api_key=OPENAI_API_KEY,
            temperature=0,
            json_mode=True,
        )
    except Exception as exc:
        print(f"[QUICK] Detection failed ({exc}), falling through to full pipeline")
        emit("quick_detect_done", {"mode": "full_pipeline", "reason": f"detection error: {exc}"})
        return None

    mode = decision.get("mode", "full_pipeline")
    reason = decision.get("reason", "")
    actions = decision.get("actions", [])

    print(f"[QUICK] Mode: {mode} — {reason}")
    emit("quick_detect_done", {"mode": mode, "reason": reason, "action_count": len(actions)})

    if mode != "quick" or not actions:
        return None

    # ── Execute actions directly ───────────────────────────────────
    emit("quick_start", {"action_count": len(actions), "task": task[:100]})

    results = []
    for i, action in enumerate(actions):
        tool_name = action.get("tool", "")
        params = action.get("params", {})
        func = _TOOL_MAP.get(tool_name)

        if func is None:
            results.append(f"[Unknown tool: {tool_name}]")
            emit("quick_action", {"index": i, "tool": tool_name, "status": "unknown_tool"})
            continue

        emit("quick_action", {"index": i, "tool": tool_name, "status": "running", "params": _safe_params(params)})

        try:
            result = func(**params)
            results.append(f"**{tool_name}**: {result}")
            emit("quick_action", {"index": i, "tool": tool_name, "status": "done", "preview": str(result)[:300]})
            print(f"[QUICK] {tool_name} done")
        except Exception as exc:
            error_msg = f"[{tool_name} error: {exc}]"
            results.append(error_msg)
            emit("quick_action", {"index": i, "tool": tool_name, "status": "error", "error": str(exc)[:200]})
            print(f"[QUICK] {tool_name} error: {exc}")

    final_output = f"# Quick Action Results\n\n" + "\n\n".join(results)
    emit("quick_done", {"results_count": len(results)})

    return {
        "final_output": final_output,
        "coverage_report": {"quality_assessment": "Direct execution via quick actions"},
        "known_issues": [],
        "plan": {"quick_mode": True, "actions": actions},
        "agent_outputs": {},
        "metadata": {
            "mode": "quick",
            "action_count": len(actions),
            "tools_used": [a.get("tool", "") for a in actions],
        },
    }


def _safe_params(params: dict) -> dict:
    """Redact sensitive params for event emission."""
    safe = {}
    for k, v in params.items():
        if any(s in k.lower() for s in ("pass", "secret", "key", "token")):
            safe[k] = "***"
        else:
            safe[k] = str(v)[:100]
    return safe
