"""Tool implementations and compact skill routing for the MiniCPM agent."""

import ast
import base64
import collections
import csv
import datetime
import html
import json
import math
import operator
import os
import platform
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = {"User-Agent": "minicpm-agent/2.0 (+local tool harness)"}
MAX_OUTPUT = 5000


def _truncate(value, limit=MAX_OUTPUT):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_ALLOWED_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "factorial",
        "degrees",
        "radians",
    )
}
_ALLOWED_FUNCS.update(abs=abs, round=round, min=min, max=max)
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ALLOWED_FUNCS
        and not node.keywords
    ):
        return _ALLOWED_FUNCS[node.func.id](*[_safe_eval(arg) for arg in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def tool_calculator(expression):
    result = _safe_eval(ast.parse(str(expression), mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def tool_percentage(percent, value):
    result = float(percent) / 100 * float(value)
    if result.is_integer():
        result = int(result)
    return str(result)


def tool_current_datetime():
    now = datetime.datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y %H:%M:%S %Z")


def _workspace_root():
    return Path(os.environ.get("MINICPM_WORKSPACE", os.getcwd())).resolve()


def _safe_path(value):
    root = _workspace_root()
    path = Path(str(value)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if os.environ.get("MINICPM_ALLOW_ANY_PATH") != "1":
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PermissionError(
                f"path must stay inside workspace {root}; set "
                "MINICPM_ALLOW_ANY_PATH=1 to override"
            ) from error
    return resolved


def tool_read_file(path):
    resolved = _safe_path(path)
    with resolved.open("r", errors="replace") as handle:
        return _truncate(handle.read(MAX_OUTPUT + 1))


def tool_list_directory(path="."):
    resolved = _safe_path(path)
    rows = []
    for item in sorted(resolved.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name)):
        kind = "dir " if item.is_dir() else "file"
        size = "-" if item.is_dir() else str(item.stat().st_size)
        rows.append(f"{kind} {size:>10} {item.name}")
    return _truncate("\n".join(rows) or "(empty directory)")


def tool_write_file(path, content, mode="overwrite"):
    try:
        resolved = _safe_path(path)
    except PermissionError:
        # 1B models invent absolute homes like /home/user/hello.py; keep
        # the file in the workspace under its basename instead of failing.
        resolved = _workspace_root() / Path(str(path)).name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if str(mode).lower() == "append" else "w"
    with resolved.open(file_mode, encoding="utf-8") as handle:
        handle.write(str(content))
    return f"Wrote {len(str(content))} characters to {resolved} ({'append' if file_mode == 'a' else 'overwrite'})."


def tool_search_files(query, path=".", pattern="*"):
    root = _safe_path(path)
    needle = str(query).lower()
    matches = []
    files_seen = 0
    for candidate in root.rglob(str(pattern)):
        if not candidate.is_file() or candidate.stat().st_size > 1_000_000:
            continue
        files_seen += 1
        if files_seen > 500:
            break
        try:
            data = candidate.read_text(errors="replace")
        except OSError:
            continue
        if "\x00" in data[:1000]:
            continue
        for line_number, line in enumerate(data.splitlines(), 1):
            if needle in line.lower():
                relative = candidate.relative_to(_workspace_root())
                matches.append(f"{relative}:{line_number}: {line.strip()[:240]}")
                if len(matches) >= 40:
                    return "\n".join(matches) + "\n...[match limit reached]"
    return "\n".join(matches) or "No matches found."


def tool_system_info():
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                memory[key] = value.strip()
    except OSError:
        pass
    disk = shutil.disk_usage("/")
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
    return "\n".join(
        [
            f"OS: {platform.platform()}",
            f"CPU: {os.cpu_count()} logical cores; machine {platform.machine()}",
            "Memory: " + ", ".join(f"{key}={value}" for key, value in memory.items()),
            f"Disk /: total={disk.total // 2**30} GiB, free={disk.free // 2**30} GiB",
            f"Load average: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}",
        ]
    )


def tool_disk_usage(path="/"):
    resolved = Path(str(path)).expanduser()
    usage = shutil.disk_usage(resolved)
    used_percent = usage.used / usage.total * 100 if usage.total else 0
    return (
        f"Disk {resolved}: total={usage.total / 2**30:.1f} GiB, "
        f"used={usage.used / 2**30:.1f} GiB, "
        f"free={usage.free / 2**30:.1f} GiB "
        f"({used_percent:.1f}% used)"
    )


def tool_process_list(query="", limit=15):
    limit = max(1, min(int(limit), 50))
    process = subprocess.run(
        ["ps", "-eo", "pid,ppid,%cpu,%mem,etime,comm,args", "--sort=-%cpu"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    lines = process.stdout.splitlines()
    if query:
        needle = str(query).lower()
        lines = [lines[0]] + [line for line in lines[1:] if needle in line.lower()]
    return "\n".join(lines[: limit + 1]) or "No matching processes."


def tool_shell(command):
    process = subprocess.run(
        str(command),
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_workspace_root(),
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    return _truncate(output or f"(no output, exit code {process.returncode})")


def _http_get(url, max_bytes=500_000):
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https")
    request = urllib.request.Request(str(url), headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(max_bytes).decode("utf-8", errors="replace")


def tool_fetch_url(url):
    page = _http_get(url)
    page = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", page)
    text = re.sub(r"(?s)<[^>]+>", " ", page)
    return _truncate(re.sub(r"\s+", " ", html.unescape(text)).strip())


def _search_mojeek(query):
    """Parse Mojeek's lightweight, no-JavaScript result page."""
    url = "https://www.mojeek.com/search?q=" + urllib.parse.quote(str(query))
    page = _http_get(url)
    results = re.findall(
        r'<h2>\s*<a[^>]+class="title"[^>]+href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>',
        page,
        re.S,
    )
    snippets = re.findall(r'<p[^>]+class="s"[^>]*>(.*?)</p>', page, re.S)
    rows = []
    for index, (href, title) in enumerate(results[:5]):
        clean_title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        snippet = snippets[index] if index < len(snippets) else ""
        snippet = re.sub(r"<[^>]+>", " ", snippet)
        snippet = re.sub(r"\s+", " ", html.unescape(snippet)).strip()
        rows.append(
            f"{index + 1}. {clean_title}\n{html.unescape(href)}\n{snippet[:240]}"
        )
    return rows


def _decode_bing_link(href):
    """Bing wraps result links; the real URL is base64url in the u= param."""
    match = re.search(r"[?&]u=a1([A-Za-z0-9_-]+)", href)
    if not match:
        return href
    encoded = match.group(1)
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return href


def _search_jina_bing(query):
    """Read Bing's result page through Jina's keyless text endpoint."""
    url = (
        "https://r.jina.ai/http://www.bing.com/search?q="
        + urllib.parse.quote(str(query))
    )
    page = _http_get(url)
    matches = re.findall(
        r"(?ms)^\d+\.\s+##\s+\[(.*?)\]\((.*?)\)\s*\n+(.*?)"
        r"(?=^\d+\.\s+##\s+\[|\Z)",
        page,
    )
    rows = []
    for index, (title, href, snippet) in enumerate(matches[:5], 1):
        title = re.sub(r"[*_`]", "", title).strip()
        link = _decode_bing_link(html.unescape(href))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        rows.append(f"{index}. {title}\n{link}\n{snippet[:400]}")
    return rows


def _search_bing(query):
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(str(query))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                "Gecko/20100101 Firefox/120.0"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        page = response.read(500_000).decode("utf-8", errors="replace")
    rows = []
    for index, block in enumerate(re.split(r'<li class="b_algo', page)[1:6]):
        match = re.search(
            r'(?s)<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block
        )
        if not match:
            continue
        link = _decode_bing_link(html.unescape(match.group(1)))
        title = html.unescape(re.sub(r"<[^>]+>", "", match.group(2))).strip()
        snippet_match = re.search(r"(?s)<p[^>]*>(.*?)</p>", block)
        snippet = ""
        if snippet_match:
            snippet = re.sub(r"<[^>]+>", " ", snippet_match.group(1))
            snippet = re.sub(r"\s+", " ", html.unescape(snippet)).strip()
        rows.append(f"{index + 1}. {title}\n{link}\n{snippet[:240]}")
    return rows


def _search_duckduckgo(query):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(str(query))
    page = _http_get(url)
    results = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', page, re.S
    )
    rows = []
    for index, (href, title) in enumerate(results[:5]):
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = snippets[index] if index < len(snippets) else ""
        snippet = re.sub(r"<[^>]+>", " ", snippet)
        snippet = re.sub(r"\s+", " ", html.unescape(snippet)).strip()
        redirect = re.search(r"uddg=([^&]+)", href)
        link = urllib.parse.unquote(redirect.group(1)) if redirect else href
        rows.append(f"{index + 1}. {html.unescape(title)}\n{link}\n{snippet[:240]}")
    return rows


_QUERY_FILLER_WORDS = {
    "what", "whats", "who", "whom", "whose", "when", "where", "which", "why",
    "how", "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "will", "would", "should", "the", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "from", "with", "that", "this", "these", "those",
    "there", "it", "its", "be", "been", "being", "have", "has", "had", "i",
    "you", "me", "my", "your", "please", "tell", "about", "find", "out",
}


def _keywordize_question(text):
    """Keyword engines (Mojeek especially) do badly on natural-language
    questions — 'What models are in the Claude 5 family' surfaces fashion
    models. Keep only the content words, entities first: a leading generic
    word ('models') otherwise dominates the ranking."""
    words = re.findall(r"[A-Za-z0-9'’.\-]+", text)
    kept = [word for word in words if word.lower() not in _QUERY_FILLER_WORDS]
    if len(kept) < 2:
        return text
    entities = [
        word for word in kept if word[:1].isupper() or word[:1].isdigit()
    ]
    generic = [word for word in kept if word not in entities]
    return " ".join(entities + generic)


def _normalize_search_query(query):
    text = re.sub(r"\s+", " ", str(query)).strip()
    lowered = text.lower()
    if (
        re.search(r"\b(usa|u\.s\.?|united states)\b", lowered)
        and re.search(r"\bstates?\b", lowered)
        and re.search(r"\bpopulation\b", lowered)
        and re.search(r"\b(largest|highest|most populous)\b", lowered)
    ):
        year = re.search(r"\b(?:19|20)\d{2}\b", lowered)
        suffix = year.group(0) if year else "current"
        return f"largest US state population {suffix}"
    if text.endswith("?") or re.match(
        r"(?i)^(what|who|when|where|which|why|how|is|are|does|do|did|can|tell)\b",
        text,
    ):
        return _keywordize_question(text.rstrip("?"))
    return text


def _page_excerpt(url, limit=900):
    """Plain-text excerpt of a page — snippets alone rarely hold the answer,
    and a 1B model invents logic when facts are missing."""
    try:
        page = _http_get(url, max_bytes=200_000)
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return ""
    page = re.sub(
        r"(?is)<(script|style|noscript|svg|nav|header|footer).*?</\1>", " ", page
    )
    text = re.sub(r"(?s)<[^>]+>", " ", page)
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return text[:limit]


_SEARCH_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
    "are", "what", "which", "who", "how", "much", "many", "does", "do",
    "with", "at", "by", "from", "us", "usa", "current", "latest", "2026",
}


def _results_look_relevant(rows, query):
    """Shared proxies and Bing sometimes serve someone else's cached SERP
    (e.g. 'Used Nissan Sentra' for a population query) — reject rows that
    share no significant term with the query."""
    terms = {
        word
        for word in re.findall(r"[a-z0-9]{4,}", str(query).lower())
        if word not in _SEARCH_STOPWORDS
    }
    if not terms:
        return True
    joined = " ".join(rows).lower()
    return any(term in joined for term in terms)


def tool_web_search(query):
    # Public search pages can rate-limit datacenter IPs, so use fallbacks.
    query = _normalize_search_query(query)
    for backend in (
        _search_jina_bing,
        _search_mojeek,
        _search_bing,
        _search_duckduckgo,
    ):
        try:
            rows = backend(query)
        except (urllib.error.URLError, ValueError, TimeoutError):
            continue
        if rows and not _results_look_relevant(rows, query):
            continue
        if rows:
            output = "\n\n".join(rows)
            excerpts = []
            for row in rows[:2]:
                lines = row.splitlines()
                if len(lines) < 2:
                    continue
                excerpt = _page_excerpt(lines[1].strip())
                if excerpt:
                    excerpts.append(f"Excerpt from {lines[1].strip()}:\n{excerpt}")
            if excerpts:
                output += "\n\n" + "\n\n".join(excerpts)
            return output
    return "No results found."


def tool_weather(location):
    url = (
        f"https://wttr.in/{urllib.parse.quote(str(location))}"
        "?format=%l:+%C,+%t+(feels+%f),+humidity+%h,+wind+%w"
    )
    return _http_get(url, max_bytes=10_000).strip()


def _json_path(value, path):
    if not path:
        return value
    for part in str(path).split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def tool_fetch_json(url, path=""):
    data = json.loads(_http_get(url))
    return _truncate(json.dumps(_json_path(data, path), indent=2, ensure_ascii=False))


def _notes_path():
    path = _workspace_root() / ".minicpm-agent" / "notes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tool_remember(title, content):
    record = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": str(title),
        "content": str(content),
    }
    with _notes_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Remembered note: {record['title']}"


def _load_notes():
    path = _notes_path()
    if not path.exists():
        return []
    notes = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            notes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return notes


def tool_recall(query):
    needle = str(query).lower()
    terms = re.findall(r"[a-z0-9]+", needle)
    matches = [
        note
        for note in reversed(_load_notes())
        if (
            needle
            in f"{note.get('title', '')} {note.get('content', '')}".lower()
            or (
                terms
                and all(
                    term
                    in f"{note.get('title', '')} {note.get('content', '')}".lower()
                    for term in terms
                )
            )
        )
    ][:10]
    if not matches:
        return "No matching notes."
    return _truncate(
        "\n\n".join(
            f"[{note['time']}] {note['title']}\n{note['content']}" for note in matches
        )
    )


def tool_list_notes():
    notes = list(reversed(_load_notes()))[:30]
    return "\n".join(
        f"{note['time']} — {note['title']}" for note in notes
    ) or "No saved notes."


TOOL_DEFINITIONS = {
    "percentage": {
        "fn": lambda args: tool_percentage(args["percent"], args["value"]),
        "direct": True,
        "description": "Compute a percentage of a value; for 12% of 4580 use percent=12 and value=4580.",
        "parameters": {"percent": "number", "value": "number"},
    },
    "calculator": {
        "fn": lambda args: tool_calculator(args["expression"]),
        "direct": True,
        "description": "Evaluate arithmetic or a math expression exactly.",
        "parameters": {"expression": "string"},
    },
    # not "direct": often an intermediate step (e.g. days-until questions)
    "current_datetime": {
        "fn": lambda args: tool_current_datetime(),
        "description": "Get the current local date, time, and timezone.",
        "parameters": {},
    },
    "read_file": {
        "fn": lambda args: tool_read_file(args["path"]),
        "description": "Read a text file inside the workspace.",
        "parameters": {"path": "string"},
    },
    "list_directory": {
        "fn": lambda args: tool_list_directory(args.get("path", ".")),
        "description": "List files and folders inside the workspace.",
        "parameters": {"path": "string, optional"},
    },
    "write_file": {
        "fn": lambda args: tool_write_file(
            args["path"], args["content"], args.get("mode", "overwrite")
        ),
        "direct": True,
        "description": "Create, overwrite, or append to a workspace text file.",
        "parameters": {
            "path": "string",
            "content": "string",
            "mode": "overwrite or append, optional",
        },
    },
    "search_files": {
        "fn": lambda args: tool_search_files(
            args["query"], args.get("path", "."), args.get("pattern", "*")
        ),
        "description": "Search text across workspace files and return matching lines.",
        "parameters": {
            "query": "string",
            "path": "string, optional",
            "pattern": "glob such as *.py, optional",
        },
    },
    "system_info": {
        "fn": lambda args: tool_system_info(),
        "direct": True,
        "description": "Get OS, CPU, memory, disk, and load information.",
        "parameters": {},
    },
    "disk_usage": {
        "fn": lambda args: tool_disk_usage(args.get("path", "/")),
        "direct": True,
        "description": "Get exact total, used, and free disk space for a path.",
        "parameters": {"path": "string, optional; defaults to /"},
    },
    "process_list": {
        "fn": lambda args: tool_process_list(
            args.get("query", ""), args.get("limit", 15)
        ),
        "direct": True,
        "description": "List high-CPU processes, optionally filtered by name.",
        "parameters": {"query": "string, optional", "limit": "integer, optional"},
    },
    "shell": {
        "fn": lambda args: tool_shell(args["command"]),
        "direct": True,
        "description": "Run a Linux shell command in the workspace. Use only when explicitly requested.",
        "parameters": {"command": "string"},
    },
    "web_search": {
        "fn": lambda args: tool_web_search(args["query"]),
        "description": "Search the live web and return five results.",
        "parameters": {"query": "string"},
    },
    "fetch_url": {
        "fn": lambda args: tool_fetch_url(args["url"]),
        "description": "Fetch a web page and return visible text.",
        "parameters": {"url": "http(s) URL"},
    },
    "fetch_json": {
        "fn": lambda args: tool_fetch_json(args["url"], args.get("path", "")),
        "direct": True,
        "description": "Fetch a JSON API; optionally select a dot path such as data.0.name.",
        "parameters": {"url": "http(s) URL", "path": "dot path, optional"},
    },
    "weather": {
        "fn": lambda args: tool_weather(args["location"]),
        "direct": True,
        "description": "Get current weather for a city or location.",
        "parameters": {"location": "string"},
    },
    "remember": {
        "fn": lambda args: tool_remember(args["title"], args["content"]),
        "direct": True,
        "description": "Save a durable local note for later conversations.",
        "parameters": {"title": "string", "content": "string"},
    },
    "recall": {
        "fn": lambda args: tool_recall(args["query"]),
        "direct": True,
        "description": "Search saved local notes by words in title or content.",
        "parameters": {"query": "string"},
    },
    "list_notes": {
        "fn": lambda args: tool_list_notes(),
        "direct": True,
        "description": "List recent saved-note titles and dates.",
        "parameters": {},
    },
}

# --------------------------------------------------------- convert pack ---

_UNIT_ALIASES = {
    "millimeter": "mm", "millimetre": "mm",
    "centimeter": "cm", "centimetre": "cm",
    "meter": "m", "metre": "m",
    "kilometer": "km", "kilometre": "km",
    "inch": "in", "inches": "in",
    "foot": "ft", "feet": "ft",
    "yard": "yd",
    "mile": "mi",
    "milligram": "mg",
    "gram": "g",
    "kilogram": "kg", "kilo": "kg",
    "tonne": "t", "ton": "t",
    "ounce": "oz",
    "pound": "lb", "lbs": "lb",
    "stone": "st",
    "milliliter": "ml", "millilitre": "ml",
    "liter": "l", "litre": "l",
    "gallon": "gal",
    "quart": "qt",
    "pint": "pt",
    "fluid ounce": "floz", "fl oz": "floz",
    "celsius": "c", "°c": "c", "centigrade": "c",
    "fahrenheit": "f", "°f": "f",
    "kelvin": "k",
    "m/s": "mps", "meters per second": "mps",
    "km/h": "kph", "kmh": "kph", "kilometers per hour": "kph",
    "miles per hour": "mph",
    "knots": "knot",
    "bit": "bit",
    "byte": "b",
    "kilobyte": "kb", "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "kibibyte": "kib", "mebibyte": "mib", "gibibyte": "gib", "tebibyte": "tib",
    "second": "s", "sec": "s",
    "minute": "min",
    "hour": "h", "hr": "h",
    "days": "day",
    "weeks": "week",
}
# canonical unit -> (category, factor to that category's base unit)
_UNIT_FACTORS = {
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "km": ("length", 1000.0), "in": ("length", 0.0254),
    "ft": ("length", 0.3048), "yd": ("length", 0.9144),
    "mi": ("length", 1609.344),
    "mg": ("mass", 1e-6), "g": ("mass", 0.001), "kg": ("mass", 1.0),
    "t": ("mass", 1000.0), "oz": ("mass", 0.028349523125),
    "lb": ("mass", 0.45359237), "st": ("mass", 6.35029318),
    "ml": ("volume", 0.001), "l": ("volume", 1.0),
    "gal": ("volume", 3.785411784), "qt": ("volume", 0.946352946),
    "pt": ("volume", 0.473176473), "floz": ("volume", 0.0295735295625),
    "mps": ("speed", 1.0), "kph": ("speed", 1000.0 / 3600.0),
    "mph": ("speed", 0.44704), "knot": ("speed", 0.514444),
    "bit": ("data", 0.125), "b": ("data", 1.0),
    "kb": ("data", 1e3), "mb": ("data", 1e6), "gb": ("data", 1e9),
    "tb": ("data", 1e12), "kib": ("data", 1024.0), "mib": ("data", 2.0**20),
    "gib": ("data", 2.0**30), "tib": ("data", 2.0**40),
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "day": ("time", 86400.0), "week": ("time", 604800.0),
}
_TEMPERATURE_UNITS = {"c", "f", "k"}


def _normalize_unit(unit):
    text = str(unit).strip().lower().rstrip(".")
    for candidate in (text, text[:-1] if text.endswith("s") else text):
        if candidate in _UNIT_ALIASES:
            return _UNIT_ALIASES[candidate]
        if candidate in _UNIT_FACTORS or candidate in _TEMPERATURE_UNITS:
            return candidate
    raise ValueError(
        f"unknown unit {unit!r}; supported categories: length, mass, volume, "
        "speed, data, time, temperature (c/f/k)"
    )


def tool_unit_convert(value, from_unit, to_unit):
    amount = float(value)
    source = _normalize_unit(from_unit)
    target = _normalize_unit(to_unit)
    if source in _TEMPERATURE_UNITS or target in _TEMPERATURE_UNITS:
        if not (source in _TEMPERATURE_UNITS and target in _TEMPERATURE_UNITS):
            raise ValueError("temperature converts only to temperature")
        celsius = {
            "c": amount,
            "f": (amount - 32.0) * 5.0 / 9.0,
            "k": amount - 273.15,
        }[source]
        result = {
            "c": celsius,
            "f": celsius * 9.0 / 5.0 + 32.0,
            "k": celsius + 273.15,
        }[target]
    else:
        source_category, source_factor = _UNIT_FACTORS[source]
        target_category, target_factor = _UNIT_FACTORS[target]
        if source_category != target_category:
            raise ValueError(
                f"cannot convert {source_category} to {target_category}"
            )
        result = amount * source_factor / target_factor
    return f"{amount:g} {source} = {result:.6g} {target}"


_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
    "%B %d %Y", "%B %d, %Y", "%d %B %Y",
    "%b %d %Y", "%b %d, %Y", "%d %b %Y",
)


def _parse_date(text):
    cleaned = str(text).strip()
    if cleaned.lower() in ("today", "now"):
        return datetime.date.today()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"could not parse date {text!r}; use YYYY-MM-DD")


def tool_days_between(start, end="today"):
    first = _parse_date(start)
    second = _parse_date(end or "today")
    return f"{(second - first).days} days from {first.isoformat()} to {second.isoformat()}"


def tool_date_add(start, days):
    base = _parse_date(start)
    result = base + datetime.timedelta(days=int(days))
    return f"{base.isoformat()} + {int(days)} days = {result.strftime('%A, %Y-%m-%d')}"


# --------------------------------------------------------- finance pack ---

_CRYPTO_IDS = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
    "doge": "dogecoin", "xrp": "ripple", "ada": "cardano",
    "ltc": "litecoin", "bnb": "binancecoin", "dot": "polkadot",
}


def tool_stock_price(ticker):
    symbol = str(ticker).strip().upper()
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?interval=1d&range=1d"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read(200_000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError(f"unknown ticker {ticker!r}") from error
        raise
    result = (data.get("chart", {}).get("result") or [{}])[0]
    meta = result.get("meta", {})
    price = meta.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"no quote for ticker {ticker!r}")
    previous = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = ""
    if previous:
        move = (price - previous) / previous * 100
        change = f", {move:+.2f}% vs prev close {previous:g}"
    return (
        f"{meta.get('symbol', symbol)} ({meta.get('fullExchangeName', 'market')}): "
        f"{price:g} {meta.get('currency', 'USD')}{change}"
    )


def tool_crypto_price(coin):
    name = str(coin).strip().lower()
    coin_id = _CRYPTO_IDS.get(name, name)
    url = (
        "https://api.coingecko.com/api/v3/simple/price?ids="
        f"{urllib.parse.quote(coin_id)}&vs_currencies=usd"
    )
    data = json.loads(_http_get(url, max_bytes=10_000))
    if coin_id not in data or "usd" not in data[coin_id]:
        raise ValueError(f"unknown coin {coin!r}; try the CoinGecko id, e.g. 'bitcoin'")
    return f"{coin_id}: ${data[coin_id]['usd']:,} USD"


def tool_currency_convert(amount, from_currency, to_currency):
    quantity = float(amount)
    source = str(from_currency).strip().upper()
    target = str(to_currency).strip().upper()
    url = (
        f"https://api.frankfurter.app/latest?amount={quantity}"
        f"&from={urllib.parse.quote(source)}&to={urllib.parse.quote(target)}"
    )
    data = json.loads(_http_get(url, max_bytes=10_000))
    if target not in data.get("rates", {}):
        raise ValueError(f"unsupported currency pair {source}->{target}")
    return (
        f"{quantity:g} {source} = {data['rates'][target]:g} {target} "
        f"(ECB reference rate, {data.get('date', 'latest')})"
    )


# ------------------------------------------------------------- git pack ---

def _run_git(*args):
    process = subprocess.run(
        ["git", "-C", str(_workspace_root()), *args],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    return _truncate(output or "(no output)")


def tool_git_status():
    return _run_git("status", "--short", "--branch")


def tool_git_log(limit=10):
    count = max(1, min(int(limit), 50))
    return _run_git("log", "--oneline", "--decorate", "-n", str(count))


def tool_git_diff(path=""):
    if str(path).strip():
        return _run_git("diff", "--", str(path))
    return _run_git("diff", "--stat")


# ------------------------------------------------------------ data pack ---

def _read_csv_rows(path, max_rows=10_000):
    resolved = _safe_path(path)
    with resolved.open("r", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        rows = []
        for index, row in enumerate(reader):
            if index > max_rows:
                return rows, True, dialect.delimiter
            rows.append(row)
    return rows, False, dialect.delimiter


def tool_csv_inspect(path):
    rows, capped, delimiter = _read_csv_rows(path)
    if not rows:
        return "Empty file."
    header, data = rows[0], rows[1:]
    count = f"{len(data)}{'+' if capped else ''}"
    sample = "\n".join(" | ".join(cell[:40] for cell in row) for row in data[:3])
    return (
        f"Delimiter: {delimiter!r}\n"
        f"Columns ({len(header)}): {', '.join(header)}\n"
        f"Data rows: {count}\n"
        f"First rows:\n{sample or '(none)'}"
    )


def tool_csv_column_stats(path, column):
    rows, capped, _ = _read_csv_rows(path)
    if not rows:
        return "Empty file."
    header, data = rows[0], rows[1:]
    wanted = str(column).strip().lower()
    try:
        index = [cell.strip().lower() for cell in header].index(wanted)
    except ValueError:
        raise ValueError(f"no column {column!r}; columns are: {', '.join(header)}")
    values = [row[index].strip() for row in data if len(row) > index and row[index].strip()]
    numbers = []
    for value in values:
        try:
            numbers.append(float(value.replace(",", "").lstrip("$").rstrip("%")))
        except ValueError:
            pass
    suffix = " (sampled first 10000 rows)" if capped else ""
    if numbers and len(numbers) >= len(values) / 2:
        return (
            f"{header[index]}: count {len(numbers)}, sum {sum(numbers):.6g}, "
            f"min {min(numbers):.6g}, max {max(numbers):.6g}, "
            f"mean {sum(numbers) / len(numbers):.6g}{suffix}"
        )
    top = collections.Counter(values).most_common(5)
    listing = ", ".join(f"{value!r} x{count}" for value, count in top)
    return f"{header[index]}: {len(values)} non-empty values; top values: {listing}{suffix}"


# --------------------------------------------------------- network pack ---

def tool_public_ip():
    return _http_get("https://api.ipify.org", max_bytes=100).strip()


def tool_dns_lookup(host):
    name = str(host).strip()
    addresses = sorted({item[4][0] for item in socket.getaddrinfo(name, None)})
    return f"{name}: {', '.join(addresses)}"


def tool_http_status(url):
    target = str(url).strip()
    if not target.startswith(("http://", "https://")):
        target = "https://" + target
    browser_agent = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(target, headers=browser_agent, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return f"{response.status} {response.reason} — {response.geturl()}"
        except urllib.error.HTTPError as error:
            if method == "HEAD" and error.code in (403, 405, 501):
                continue  # some servers reject HEAD; retry with GET
            return f"{error.code} {error.reason} — {target}"
    return f"request failed — {target}"


def tool_ping(host):
    process = subprocess.run(
        ["ping", "-c", "3", "-W", "2", str(host).strip()],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    return "\n".join(output.splitlines()[-4:]) or "(no output)"


TOOL_DEFINITIONS_EXTRA = {
    "unit_convert": {
        "fn": lambda args: tool_unit_convert(
            args["value"], args["from_unit"], args["to_unit"]
        ),
        "direct": True,
        "description": "Convert between units of length, mass, volume, speed, data, time, or temperature.",
        "parameters": {
            "value": "number",
            "from_unit": "string, e.g. km",
            "to_unit": "string, e.g. miles",
        },
    },
    "days_between": {
        "fn": lambda args: tool_days_between(args["start"], args.get("end", "today")),
        "direct": True,
        "description": "Days between two dates; 'today' is allowed.",
        "parameters": {"start": "date, e.g. 2026-01-01", "end": "date or today, optional"},
    },
    "date_add": {
        "fn": lambda args: tool_date_add(args["start"], args["days"]),
        "direct": True,
        "description": "Add or subtract days from a date; returns the resulting weekday and date.",
        "parameters": {"start": "date or today", "days": "integer, negative to go back"},
    },
    "stock_price": {
        "fn": lambda args: tool_stock_price(args["ticker"]),
        "direct": True,
        "description": "Latest stock quote by ticker symbol, e.g. AAPL or MSFT.",
        "parameters": {"ticker": "string"},
    },
    "crypto_price": {
        "fn": lambda args: tool_crypto_price(args["coin"]),
        "direct": True,
        "description": "Current cryptocurrency price in USD, e.g. bitcoin or eth.",
        "parameters": {"coin": "string"},
    },
    "currency_convert": {
        "fn": lambda args: tool_currency_convert(
            args["amount"], args["from_currency"], args["to_currency"]
        ),
        "direct": True,
        "description": "Convert money between currencies using ECB reference rates.",
        "parameters": {
            "amount": "number",
            "from_currency": "ISO code, e.g. USD",
            "to_currency": "ISO code, e.g. EUR",
        },
    },
    "git_status": {
        "fn": lambda args: tool_git_status(),
        "direct": True,
        "description": "Show git branch and changed files in the workspace repo.",
        "parameters": {},
    },
    "git_log": {
        "fn": lambda args: tool_git_log(args.get("limit", 10)),
        "direct": True,
        "description": "Show recent git commits, one line each.",
        "parameters": {"limit": "integer, optional"},
    },
    "git_diff": {
        "fn": lambda args: tool_git_diff(args.get("path", "")),
        "direct": True,
        "description": "Show unstaged changes: a summary, or the full diff of one file.",
        "parameters": {"path": "string, optional file path"},
    },
    # not "direct": usually step 1 before csv_column_stats
    "csv_inspect": {
        "fn": lambda args: tool_csv_inspect(args["path"]),
        "description": "Show a CSV file's columns, row count, and first rows.",
        "parameters": {"path": "string"},
    },
    "csv_column_stats": {
        "fn": lambda args: tool_csv_column_stats(args["path"], args["column"]),
        "direct": True,
        "description": "Numeric summary (or top values) of one CSV column.",
        "parameters": {"path": "string", "column": "column name"},
    },
    "public_ip": {
        "fn": lambda args: tool_public_ip(),
        "direct": True,
        "description": "Get this machine's public IP address.",
        "parameters": {},
    },
    "dns_lookup": {
        "fn": lambda args: tool_dns_lookup(args["host"]),
        "direct": True,
        "description": "Resolve a hostname to its IP addresses.",
        "parameters": {"host": "string, e.g. example.com"},
    },
    "http_status": {
        "fn": lambda args: tool_http_status(args["url"]),
        "direct": True,
        "description": "Check whether a website is up and what HTTP status it returns.",
        "parameters": {"url": "URL or bare domain"},
    },
    "ping": {
        "fn": lambda args: tool_ping(args["host"]),
        "direct": True,
        "description": "Ping a host 3 times and report latency.",
        "parameters": {"host": "string"},
    },
}
TOOL_DEFINITIONS.update(TOOL_DEFINITIONS_EXTRA)

SKILL_TOOLS = {
    "core": ["percentage", "calculator", "current_datetime"],
    "files": ["read_file", "list_directory", "write_file", "search_files"],
    "system": ["disk_usage", "system_info", "process_list"],
    "web": ["web_search", "fetch_url", "fetch_json", "weather"],
    "memory": ["remember", "recall", "list_notes"],
    "convert": ["unit_convert", "days_between", "date_add"],
    "finance": ["stock_price", "crypto_price", "currency_convert"],
    "git": ["git_status", "git_log", "git_diff"],
    "data": ["csv_inspect", "csv_column_stats"],
    "network": ["public_ip", "dns_lookup", "http_status", "ping"],
    "shell": ["shell"],
}

SKILL_DESCRIPTIONS = {
    "core": "math and local date/time",
    "files": "safe workspace reading, writing, listing, and search",
    "system": "computer health and process inspection",
    "web": "search, pages, JSON APIs, and weather",
    "memory": "durable local notes and recall",
    "convert": "unit conversion and date arithmetic",
    "finance": "stock, crypto, and currency rates (keyless public APIs)",
    "git": "read-only git status, log, and diff",
    "data": "CSV inspection and column statistics",
    "network": "public IP, DNS, ping, and HTTP status checks",
    "shell": "unrestricted shell commands (explicit opt-in)",
}


def auto_select_skills(text):
    """Select a small catalog with deterministic keyword routing."""
    lowered = str(text).lower()
    selected = ["core"]
    patterns = {
        "web": (
            r"\b(web|online|internet|latest|newest|current|currently|today|recent"
            r"|recently|news|weather|forecast|url|website|http|search the web"
            r"|look up|price|as of|up[- ]to[- ]date|right now|just released"
            r"|most recent|population|census|president|governor"
            r"|election|score|schedule|standings|winner|ceo)\b"
        ),
        "files": r"\b(file|files|folder|directory|path|document|read|write|append|save|codebase|source code|grep|todo)\b",
        "system": r"\b(system|computer|machine|cpu|memory|ram|disk|storage|process|processes|uptime|load average)\b",
        "memory": r"\b(remember|recall|note|notes|memorize|what did i|saved)\b",
        "convert": (
            r"\b(convert|celsius|fahrenheit|kelvin|miles?|km|kilomet\w+|kg|kilograms?"
            r"|pounds?|lbs|ounces?|liters?|litres?|gallons?|inches|feet|meters?"
            r"|centimet\w+|mph|kph|knots?|gigabytes?|megabytes?|terabytes?"
            r"|how many days|days? (?:between|until|since|from|after|before|ago))\b"
        ),
        "finance": (
            r"\b(stocks?|shares?|ticker|nasdaq|nyse|crypto|bitcoin|btc|ethereum|eth"
            r"|dogecoin|solana|exchange rate|currency|currencies|usd|eur|gbp|jpy"
            r"|dollars? (?:in|to)|euros?)\b"
        ),
        "git": r"\b(git|repo|repository|branch|commits?|diff|staged|unstaged|uncommitted)\b",
        "data": r"\b(csv|tsv|spreadsheet|columns?|dataset|rows)\b",
        "network": (
            r"\b(ip address|public ip|my ip|dns|ping|resolves?|reachable"
            r"|status code|http status)\b"
            r"|is [\w. -]{1,60}\b(up|down|online|offline)\b"
        ),
        "shell": r"\b(run (?:this )?(?:command|shell)|execute (?:this )?(?:command|shell)|terminal command|bash command)\b",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lowered):
            selected.append(name)
    if re.search(r"\b(unit tests?|test suite)\b", lowered) and "files" not in selected:
        selected.append("files")
    return selected


_CORE_TOOL_SIGNAL = re.compile(
    # digits only count as a math signal next to an operator or percent —
    # a bare version number like "Claude 5" or "GPT-4" is not arithmetic
    r"\d[\d,.]*\s*[%×*/^+-]|[×*/^+-]\s*\d|% of"
    r"|\b(calculat\w*|comput\w*|math|plus|minus|times|divided?|multipl\w*"
    r"|percent\w*|sum|square root|squared|cubed|factorial|time is it"
    r"|current time|the time|date|today|tomorrow|yesterday|what day|clock)\b"
)


def question_needs_tools(text, skill_names):
    """False when only core matched and there is no math/time signal — the
    model should answer from its own knowledge, with no tool catalog to
    tempt it into refusing general-knowledge questions."""
    if list(skill_names) != ["core"]:
        return True
    return bool(_CORE_TOOL_SIGNAL.search(str(text).lower()))


def tools_for_skills(skill_names):
    tools = {}
    for skill_name in skill_names:
        for tool_name in SKILL_TOOLS[skill_name]:
            tools[tool_name] = TOOL_DEFINITIONS[tool_name]
    return tools


def narrow_tools_for_request(text, tools):
    """Remove same-pack tools that a 1B model commonly confuses."""
    lowered = str(text).lower()
    memory_tools = {"remember", "recall", "list_notes"}
    chosen_memory_tool = None
    if memory_tools.intersection(tools):
        if re.search(r"\b(list|show)\b.*\bnotes?\b", lowered):
            chosen_memory_tool = "list_notes"
        elif re.search(r"\b(recall|find|search|what did i|do you remember)\b", lowered):
            chosen_memory_tool = "recall"
        elif re.search(r"\b(remember|memorize|save (?:this )?note)\b", lowered):
            chosen_memory_tool = "remember"
    if chosen_memory_tool:
        tools = {
            name: tool
            for name, tool in tools.items()
            if name not in memory_tools or name == chosen_memory_tool
        }
    # A create/save request should see one file action: write_file. With the
    # full files pack the model refuses ("I can't create a program").
    if "write_file" in tools and re.search(
        r"\b(create|write|make|generate|save)\b.{0,60}"
        r"\b(file|program|script|code|page|note|document"
        r"|\.(?:py|js|ts|html|css|txt|md|csv|json|sh))\b",
        lowered,
    ):
        reading_tools = {"read_file", "search_files", "list_directory"}
        tools = {
            name: tool for name, tool in tools.items() if name not in reading_tools
        }
    # For a numeric column question, csv_inspect only tempts the model to do
    # (bad) mental math on the sample rows — leave it one correct path.
    if "csv_column_stats" in tools and re.search(
        r"\b(average|mean|sum|total|count|min|max|median)\b", lowered
    ):
        tools = {name: tool for name, tool in tools.items() if name != "csv_inspect"}
    # A question about a local data file should never route to the web.
    if (
        ("csv_inspect" in tools or "csv_column_stats" in tools)
        and re.search(r"\b[\w./-]+\.(csv|tsv)\b", lowered)
    ):
        web_tools = {"web_search", "fetch_url", "fetch_json", "weather"}
        tools = {name: tool for name, tool in tools.items() if name not in web_tools}
    # Give the small model one clear web action. A URL can be fetched directly;
    # a current/external fact should start with search.
    web_tools = {"web_search", "fetch_url", "fetch_json", "weather"}
    if web_tools.intersection(tools):
        if re.search(r"\b(weather|forecast|temperature)\b", lowered):
            chosen_web_tool = "weather"
        elif re.search(r"https?://\S+", lowered) and re.search(
            r"\b(json|api)\b", lowered
        ):
            chosen_web_tool = "fetch_json"
        elif re.search(r"https?://\S+", lowered):
            chosen_web_tool = "fetch_url"
        else:
            chosen_web_tool = "web_search"
        tools = {
            name: tool
            for name, tool in tools.items()
            if name not in web_tools or name == chosen_web_tool
        }
        # A current-fact search question with no math/time signal doesn't
        # need core tools — the model otherwise feeds words like
        # 'California' into percentage() and reports the ValueError.
        if chosen_web_tool == "web_search" and not _CORE_TOOL_SIGNAL.search(lowered):
            tools = {
                name: tool
                for name, tool in tools.items()
                if name not in SKILL_TOOLS["core"]
            }
    return tools


def should_seed_web_search(text, tools):
    """True when routing already decided this is a search question (the only
    non-core tool is web_search) — run the search up front instead of hoping
    the 1B model calls it; it sometimes refuses without searching."""
    non_core = set(tools) - set(SKILL_TOOLS["core"])
    return non_core == {"web_search"} and not re.search(r"https?://", str(text))


def direct_tool_call_for_request(text, tools):
    """Bypass model reasoning for obvious, argument-free factual requests."""
    lowered = str(text).lower()
    if (
        "disk_usage" in tools
        and re.search(r"\b(disk|storage|drive)\b", lowered)
        and re.search(r"\b(free|available|space|usage|capacity|full)\b", lowered)
    ):
        return "disk_usage", {"path": "/"}
    if "system_info" in tools and re.search(
        r"\b(system info|system status|computer specs|machine specs)\b", lowered
    ):
        return "system_info", {}
    if "current_datetime" in tools and re.search(
        r"\b(what time is it|current time|what(?:'s| is) the time"
        r"|today'?s date|current date|what day is (?:it|today))\b",
        lowered,
    ):
        return "current_datetime", {}
    if "search_files" in tools:
        match = re.search(
            r"\b(?:find|search(?: for)?|grep)\s+(?:for\s+)?['\"]?([\w-]{2,})['\"]?"
            r".{0,30}?\b(?:in|under|within)\b\s+(?:the\s+)?['\"]?([\w./-]+)",
            lowered,
        )
        if match:
            term, location = match.groups()
            generic = {
                "this", "current", "my", "project", "folder", "directory",
                "codebase", "files", "workspace", "repo",
            }
            path = "."
            if location not in generic and (_workspace_root() / location).exists():
                path = location
            return "search_files", {"query": term, "path": path}
    if "csv_column_stats" in tools:
        stat = re.search(
            r"\b(average|mean|sum|total|count|min|max|median)\b", lowered
        )
        column = re.search(r"(?:the\s+)?['\"]?([\w-]+)['\"]?\s+column", lowered)
        data_file = re.search(r"\b([\w./-]+\.(?:csv|tsv))\b", lowered)
        if stat and column and data_file:
            return "csv_column_stats", {
                "path": data_file.group(1),
                "column": column.group(1),
            }
    return None


def build_system_prompt(tools):
    specs = [
        {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for name, tool in tools.items()
    ]
    # Prefer the routed non-core tool in the example. A 1B model otherwise
    # copies the always-present calculator example for unrelated requests.
    non_core_names = [
        name for name in tools if name not in SKILL_TOOLS["core"]
    ]
    example_tool = non_core_names[0] if non_core_names else "calculator"
    if example_tool == "web_search":
        example = (
            '<function name="web_search">'
            '<param name="query">current US state population 2026</param></function>'
        )
    elif example_tool == "write_file":
        example = (
            '<function name="write_file"><param name="path">hello.py</param>'
            '<param name="content">print("Hello, world!")</param></function>'
        )
    elif example_tool == "calculator":
        example = (
            '<function name="calculator">'
            '<param name="expression">2 + 2</param></function>'
        )
    else:
        name, tool = example_tool, tools[example_tool]
        params = "".join(
            f'<param name="{param}">...</param>' for param in tool["parameters"]
        )
        example = f'<function name="{name}">{params}</function>'
    rules = [
        "You are a concise local assistant. Available tools:",
        json.dumps(specs, separators=(",", ":")),
        "Call one tool by outputting only a tag in exactly this form, "
        "using a real tool name and its parameter names:",
        example,
        "Use tools for facts or actions they provide; never invent a tool result.",
        "Do not output <think> or explain before a tool call.",
        "After a result, answer or make the next necessary call. Never repeat a successful call.",
    ]
    if "calculator" in tools:
        rules.append("Use calculator for arithmetic instead of calculating yourself.")
    if "percentage" in tools:
        rules.append("For a percent-of question, use percentage, not calculator.")
    if "shell" in tools:
        rules.append("Use shell only when the user explicitly asks to run a command.")
    if "remember" in tools:
        rules.append("If asked to remember or save a note, you MUST call remember.")
    if "recall" in tools:
        rules.append("If asked to recall a note, you MUST call recall; do not save anything.")
    if "list_notes" in tools:
        rules.append("If asked to list saved notes, you MUST call list_notes.")
    if "web_search" in tools:
        rules.append(
            "For current, recent, as-of, or external factual questions, "
            "you MUST call web_search before answering."
        )
    if "unit_convert" in tools:
        rules.append(
            "For unit conversions call unit_convert; never convert numbers yourself."
        )
    if "days_between" in tools:
        rules.append(
            "For 'days until/since' questions call days_between with start=today "
            "and the target date as end, e.g. end=2026-12-25 for Christmas 2026."
        )
    if "stock_price" in tools:
        rules.append(
            "For stock quotes call stock_price with the ticker; for crypto call "
            "crypto_price; to convert money call currency_convert."
        )
    if "csv_inspect" in tools:
        rules.append(
            "For CSV questions call csv_inspect first; use csv_column_stats "
            "for sums, averages, or counts of one column."
        )
    if "git_status" in tools:
        rules.append("For git questions use git_status, git_log, or git_diff.")
    if "write_file" in tools:
        rules.append(
            "To create or save any file, program, or note to disk, you MUST "
            "call write_file with a relative filename (like hello.py) in "
            "path and the COMPLETE file text in content. You CAN create files."
        )
    return "\n".join(rules)
