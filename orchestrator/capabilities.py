"""
Real-world capabilities that forged tools can use.

These are injected into the forged tool's namespace so the LLM-generated
code can call them directly:

    def conduct_survey(topic: str) -> str:
        results = _search(f"{topic} market research survey data")
        _save_file(f"{topic}_survey.md", results)
        return results

The functions do REAL work — actual web requests, actual file writes,
actual data processing.
"""

import os
import re
import json

# ── Output directory for files created by agents ────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# WEB SEARCH — real DuckDuckGo search
# ═══════════════════════════════════════════════════════════════════

def search_web(query: str, max_results: int = 8) -> str:
    """Perform a REAL web search via DuckDuckGo. Returns actual results."""
    import requests

    results = []

    # ── DuckDuckGo Instant Answer API ───────────────────────────────
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
            timeout=10,
        )
        data = resp.json()

        if data.get("AbstractText"):
            results.append(f"[Summary] {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"  Source: {data['AbstractURL']}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"]
                url = topic.get("FirstURL", "")
                results.append(f"- {text}" + (f"\n  URL: {url}" if url else ""))
            elif isinstance(topic, dict) and "Topics" in topic:
                for sub in topic["Topics"][:3]:
                    if "Text" in sub:
                        results.append(f"- {sub['Text']}")
    except Exception:
        pass

    # ── DuckDuckGo HTML search (fallback for richer results) ────────
    if len(results) < 3:
        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=10,
            )
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
                resp.text, re.DOTALL,
            )
            urls = re.findall(
                r'class="result__url"[^>]*>(.*?)</a>',
                resp.text, re.DOTALL,
            )
            for i, s in enumerate(snippets[:max_results]):
                clean = re.sub(r"<[^>]+>", "", s).strip()
                if clean and clean not in str(results):
                    url_text = re.sub(r"<[^>]+>", "", urls[i]).strip() if i < len(urls) else ""
                    entry = f"- {clean}"
                    if url_text:
                        entry += f"\n  URL: https://{url_text}"
                    results.append(entry)
        except Exception:
            pass

    if not results:
        return f"No search results found for: {query}"

    return f"Search results for '{query}':\n\n" + "\n\n".join(results[:max_results])


# ═══════════════════════════════════════════════════════════════════
# WEB SCRAPE — fetch and extract text from a real URL
# ═══════════════════════════════════════════════════════════════════

def scrape_url(url: str, max_chars: int = 8000) -> str:
    """Fetch a REAL webpage and extract its text content."""
    import requests

    # Clean URL — strip common prefixes from search results
    url = url.strip()
    if url.lower().startswith("url:"):
        url = url[4:].strip()
    if not url.startswith("http"):
        url = "https://" + url

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        html = resp.text

        # Strip scripts and styles
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL)

        # Convert some tags to text markers
        html = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n## \1\n", html, flags=re.DOTALL)
        html = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", html, flags=re.DOTALL)
        html = re.sub(r"<br\s*/?>", "\n", html)
        html = re.sub(r"<p[^>]*>", "\n", html)

        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\n ", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text:
            return f"No content extracted from: {url}"

        return f"Content from {url}:\n\n{text[:max_chars]}"

    except Exception as e:
        return f"Scrape error for {url}: {e}"


# ═══════════════════════════════════════════════════════════════════
# FILE I/O — create real files in the output directory
# ═══════════════════════════════════════════════════════════════════

def save_file(filename: str, content: str) -> str:
    """Save a REAL file to the output/ directory. Returns the file path."""
    # Sanitize filename
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filepath = os.path.join(OUTPUT_DIR, safe_name)

    try:
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) != OUTPUT_DIR else OUTPUT_DIR, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved successfully: {filepath} ({len(content)} chars)"
    except Exception as e:
        return f"File save error: {e}"


def read_file(filename: str) -> str:
    """Read a file from the output/ directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"File read error: {e}"


def list_files() -> str:
    """List all files in the output/ directory."""
    try:
        files = os.listdir(OUTPUT_DIR)
        if not files:
            return "Output directory is empty."
        return "Files in output/:\n" + "\n".join(f"  - {f}" for f in files)
    except Exception as e:
        return f"List files error: {e}"


# ═══════════════════════════════════════════════════════════════════
# DATA / JSON — fetch and process real JSON APIs
# ═══════════════════════════════════════════════════════════════════

def fetch_json(url: str) -> str:
    """Fetch REAL JSON data from a URL and return it formatted."""
    import requests

    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data, indent=2)[:8000]
    except Exception as e:
        return f"JSON fetch error: {e}"


# ═══════════════════════════════════════════════════════════════════
# COMPUTATION — real Python data processing
# ═══════════════════════════════════════════════════════════════════

def compute(code_str: str, context: dict | None = None) -> str:
    """Execute a Python expression/code block and return the result.

    The code runs in a restricted namespace with math, statistics,
    json, re, datetime, and collections available.
    """
    import math
    import statistics
    import datetime
    import collections

    safe_ns = {
        "__builtins__": {
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
            "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "print": print, "isinstance": isinstance, "type": type,
        },
        "math": math,
        "statistics": statistics,
        "json": json,
        "re": re,
        "datetime": datetime,
        "collections": collections,
    }
    if context:
        safe_ns.update(context)

    try:
        # Try as expression first
        result = eval(code_str, safe_ns)
        return str(result)
    except SyntaxError:
        # Try as statements
        try:
            exec(code_str, safe_ns)
            # Return the last assigned variable
            output = safe_ns.get("result", safe_ns.get("output", "Code executed successfully."))
            return str(output)
        except Exception as e:
            return f"Computation error: {e}"
    except Exception as e:
        return f"Computation error: {e}"


# ═══════════════════════════════════════════════════════════════════
# HTML FORMS — create real, working HTML pages
# ═══════════════════════════════════════════════════════════════════

def create_html_form(filename: str, title: str, fields: list[dict], submit_action: str = "#") -> str:
    """Create a REAL working HTML form page that can be opened in a browser.

    Parameters:
        filename: e.g. "registration_form.html"
        title: Form title displayed on the page
        fields: list of {"name": "...", "label": "...", "type": "text|email|number|textarea|select", "options": [...]}
        submit_action: URL to submit to (default "#" for local)

    Returns path to the saved HTML file.
    """
    fields_html = ""
    for f in fields:
        name = f.get("name", "field")
        label = f.get("label", name)
        ftype = f.get("type", "text")
        required = "required" if f.get("required", True) else ""

        if ftype == "textarea":
            fields_html += f'<div class="field"><label for="{name}">{label}</label><textarea id="{name}" name="{name}" rows="4" {required}></textarea></div>\n'
        elif ftype == "select":
            opts = "".join(f'<option value="{o}">{o}</option>' for o in f.get("options", []))
            fields_html += f'<div class="field"><label for="{name}">{label}</label><select id="{name}" name="{name}" {required}>{opts}</select></div>\n'
        else:
            fields_html += f'<div class="field"><label for="{name}">{label}</label><input type="{ftype}" id="{name}" name="{name}" {required}></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0b;color:#e4e4e7;min-height:100vh;display:flex;justify-content:center;padding:40px 20px}}
.container{{max-width:560px;width:100%}}
h1{{font-size:1.8rem;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{color:#71717a;margin-bottom:32px;font-size:.95rem}}
.field{{margin-bottom:20px}}
label{{display:block;font-size:.85rem;font-weight:600;color:#a1a1aa;margin-bottom:6px}}
input,select,textarea{{width:100%;padding:12px 16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;color:#fafafa;font-size:.92rem;font-family:inherit;outline:none;transition:border .2s}}
input:focus,select:focus,textarea:focus{{border-color:rgba(99,102,241,.5);box-shadow:0 0 0 3px rgba(99,102,241,.12)}}
select{{cursor:pointer}}
.submit{{margin-top:24px;width:100%;padding:14px;background:linear-gradient(135deg,#6366f1,#a78bfa);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;transition:transform .15s,box-shadow .15s}}
.submit:hover{{transform:translateY(-1px);box-shadow:0 8px 24px rgba(99,102,241,.3)}}
.success{{display:none;text-align:center;padding:40px;background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);border-radius:16px;margin-top:20px}}
.success h2{{color:#34d399;margin-bottom:8px}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Please fill out the form below. All fields are required unless noted.</p>
<form id="mainForm" action="{submit_action}" method="POST">
{fields_html}
<button type="submit" class="submit">Submit</button>
</form>
<div class="success" id="successMsg">
<h2>Thank you!</h2>
<p>Your response has been recorded successfully.</p>
</div>
</div>
<script>
document.getElementById('mainForm').addEventListener('submit',function(e){{
e.preventDefault();
const data=Object.fromEntries(new FormData(this));
console.log('Form submitted:',JSON.stringify(data,null,2));
this.style.display='none';
document.getElementById('successMsg').style.display='block';
// Store in localStorage as backup
const key='yconic_form_'+Date.now();
localStorage.setItem(key,JSON.stringify(data));
}});
</script>
</body>
</html>"""

    return save_file(filename, html)


# ═══════════════════════════════════════════════════════════════════
# Namespace dict to inject into forged tools
# ═══════════════════════════════════════════════════════════════════

CAPABILITY_NAMESPACE = {
    "_search": search_web,
    "_scrape": scrape_url,
    "_save_file": save_file,
    "_read_file": read_file,
    "_list_files": list_files,
    "_fetch_json": fetch_json,
    "_compute": compute,
    "_create_form": create_html_form,
    "_OUTPUT_DIR": OUTPUT_DIR,
}
