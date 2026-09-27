#!/usr/bin/env python3
"""Small, routed tool-calling agent for MiniCPM5 running on Ollama.

Examples:
    python3 agent.py "What is 12% of 4580?"
    python3 agent.py "How much disk space is free?"
    python3 agent.py --skills files "Find TODO in this project"
    python3 agent.py --skills all
    python3 agent.py --list-skills

The Ollama registry model is completion-only, so this harness injects a small
tool catalog and parses MiniCPM5's native XML-style function calls.
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

from toolkit import (
    SKILL_DESCRIPTIONS,
    auto_select_skills,
    build_system_prompt,
    direct_tool_call_for_request,
    narrow_tools_for_request,
    question_needs_tools,
    should_seed_web_search,
    tool_web_search,
    tools_for_skills,
)

GENERAL_PROMPT = (
    "You are a concise, knowledgeable assistant. Answer from your own "
    "knowledge in one to three short sentences. If you genuinely do not "
    "know, say so briefly."
)

MODEL = os.environ.get("MINICPM_MODEL", "openbmb/minicpm5")
OLLAMA_URL = os.environ.get(
    "OLLAMA_CHAT_URL", "http://localhost:11434/api/chat"
)
MAX_TOOL_ROUNDS = 5

THINK_RE = re.compile(r"(?s)<think>.*?</think>\s*")
CALL_RE = re.compile(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>")
XML_CALL_RE = re.compile(
    r'(?s)<function\s+name=["\']([^"\']+)["\']\s*>(.*?)(?:</function>|$)'
)
XML_PARAM_RE = re.compile(
    r'(?s)<param\s+name=["\']([^"\']+)["\']\s*>(.*?)</param>'
)
REFUSAL_RE = re.compile(
    r"(?i)\b(i cannot|i can'?t|cannot (?:provide|determine|answer)"
    r"|do(?:n't| not) have access|unable to (?:provide|determine|access))\b"
)
LOCAL_ACTION_TOOLS = {
    "read_file", "write_file", "search_files", "list_directory",
    "csv_inspect", "csv_column_stats", "git_status", "git_log", "git_diff",
    "shell", "remember", "recall", "list_notes",
}


def chat(messages):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "num_ctx": 4096,
                "num_predict": 384,
            },
        }
    )
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)["message"]["content"]
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}: {error}. "
            "Start it with `ollama serve`."
        ) from error


def parse_tool_call(text, tools):
    """Parse native XML and common JSON tool-call variants."""
    match = XML_CALL_RE.search(text)
    if match and match.group(1) in tools:
        arguments = {
            key: html.unescape(value.strip())
            for key, value in XML_PARAM_RE.findall(match.group(2))
        }
        return match.group(1), arguments

    match = CALL_RE.search(text)
    if not match:
        names = "|".join(re.escape(name) for name in tools)
        match = re.search(
            rf'(?s)\{{[^{{}}]*["\']name["\']\s*:\s*["\']({names})["\'].*?\}}(?:\}})?',
            text,
        )
        if not match:
            return None
        raw = match.group(0).replace("'", '"')
    else:
        raw = match.group(1)

    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        return None
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    name = call.get("name")
    return (name, arguments) if name in tools and isinstance(arguments, dict) else None


def canonical_call(name, arguments):
    params = "".join(
        f'<param name="{html.escape(str(key))}">{html.escape(str(value))}</param>'
        for key, value in arguments.items()
    )
    return f'<function name="{html.escape(name)}">{params}</function>'


def clean_model_output(content):
    """Remove complete thoughts and flag a generation cut off inside one."""
    visible = THINK_RE.sub("", content)
    unfinished_thinking = "<think>" in visible
    if unfinished_thinking:
        visible = visible.split("<think>", 1)[0]
    return visible.strip(), unfinished_thinking


def execute_tool(name, arguments, tools, verbose):
    if verbose:
        print(f"  [tool] {name}({json.dumps(arguments)})", file=sys.stderr)
    try:
        result = str(tools[name]["fn"](arguments))
    except Exception as error:
        result = f"ERROR: {type(error).__name__}: {error}"
    if verbose:
        first_line = result.splitlines()[0] if result else ""
        print(f"  [tool] -> {first_line[:140]}", file=sys.stderr)
    return result


def looks_like_echo(visible, last_result):
    """True when the model regurgitated the tool result instead of answering.

    Whitespace-insensitive and prefix-based: the echo is often truncated by
    the num_predict cap or reflowed, so exact containment misses it.
    """
    if not last_result or len(visible) < 100:
        return False
    normalized_visible = re.sub(r"\s+", " ", visible).strip()
    normalized_result = re.sub(r"\s+", " ", last_result).strip()
    return (
        normalized_visible[:200] in normalized_result
        or normalized_result[:200] in normalized_visible
    )


def synthesize_answer(user_text, tool_name, result):
    """A fresh, catalog-free turn that answers from one tool result.

    Instructions must never ride inside tool messages of the main loop —
    the 1B model parrots them into its answer. A separate turn keeps the
    instruction in the system role where it belongs.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question in one or two short sentences, "
                "combining the tool output with your own knowledge. When they "
                "conflict, prefer the tool output."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {user_text}\n\nOutput of {tool_name}:\n{result}",
        },
    ]
    for _ in range(2):
        visible, _unfinished = clean_model_output(chat(messages))
        if visible:
            return visible
        messages[0]["content"] += "\nAnswer immediately without <think>."
    return result


def escalate_to_search(user_text, verbose):
    """When the model says it doesn't know, search the web and synthesize.
    Returns None if the search produced nothing usable."""
    if verbose:
        print(
            "  [skills] model doesn't know — escalating to web search",
            file=sys.stderr,
        )
    try:
        result = tool_web_search(user_text)
    except Exception:
        return None
    if not result or result.startswith("ERROR:") or result == "No results found.":
        return None
    return synthesize_answer(user_text, "web_search", result)


def run_turn(history, user_text, requested_skills=None, verbose=True):
    skill_names = requested_skills or auto_select_skills(user_text)
    if not requested_skills and not question_needs_tools(user_text, skill_names):
        # General-knowledge question: no tool catalog, or the 1B model
        # refuses facts it knows ("I only have calculator...").
        if verbose:
            print("  [skills] none — answering from model knowledge", file=sys.stderr)
        messages = [
            {"role": "system", "content": GENERAL_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]
        for _ in range(2):
            visible, _unfinished = clean_model_output(chat(messages))
            if visible:
                if REFUSAL_RE.search(visible[:250]):
                    visible = escalate_to_search(user_text, verbose) or visible
                messages.append({"role": "assistant", "content": visible})
                return visible, messages[1:]
            messages[0]["content"] += "\nAnswer immediately without <think>."
        answer = "(The model exhausted its token limit while reasoning.)"
        messages.append({"role": "assistant", "content": answer})
        return answer, messages[1:]

    tools = narrow_tools_for_request(user_text, tools_for_skills(skill_names))
    if verbose:
        print(
            f"  [skills] {', '.join(skill_names)} -> {', '.join(tools)}",
            file=sys.stderr,
        )

    messages = [
        {"role": "system", "content": build_system_prompt(tools)},
        *history,
        {"role": "user", "content": user_text},
    ]
    direct_call = direct_tool_call_for_request(user_text, tools)
    if direct_call:
        name, arguments = direct_call
        result = execute_tool(name, arguments, tools, verbose)
        messages.append(
            {"role": "assistant", "content": canonical_call(name, arguments)}
        )
        messages.append({"role": "tool", "content": f"Result from {name}:\n{result}"})
        messages.append({"role": "assistant", "content": result})
        return result, messages[1:]

    think_retries = 0
    last_call, last_result = None, None
    if should_seed_web_search(user_text, tools):
        arguments = {"query": user_text}
        result = execute_tool("web_search", arguments, tools, verbose)
        if not result.startswith("ERROR:"):
            messages.append(
                {"role": "assistant", "content": canonical_call("web_search", arguments)}
            )
            messages.append(
                {"role": "tool", "content": f"Result from web_search:\n{result}"}
            )
            last_call, last_result = ("web_search", arguments), result
    for _ in range(MAX_TOOL_ROUNDS):
        content = chat(messages)
        visible, unfinished_thinking = clean_model_output(content)
        call = parse_tool_call(visible, tools)
        if not call:
            if unfinished_thinking:
                if think_retries < 1:
                    think_retries += 1
                    messages[0]["content"] += (
                        "\nRespond immediately. Never emit <think>. Output only the "
                        "tool tag or the final answer."
                    )
                    continue
                answer = "(The model exhausted its token limit while reasoning; partial reasoning was hidden.)"
                messages.append({"role": "assistant", "content": answer})
                return answer, messages[1:]
            if re.search(r"<function|<tool_call", visible):
                # looked like a tool call but didn't parse — let it retry
                messages.append({"role": "assistant", "content": visible})
                messages.append(
                    {
                        "role": "tool",
                        "content": (
                            "ERROR: malformed tool call. Use a real tool name "
                            "and its exact parameter names, e.g. "
                            '<function name="calculator">'
                            '<param name="expression">2 + 2</param></function>'
                        ),
                    }
                )
                continue
            if last_result is not None and (
                looks_like_echo(visible, last_result)
                or REFUSAL_RE.search(visible[:200])
            ):
                # The model echoed the tool result, or refused despite
                # having one — synthesize the answer from it instead.
                answer = synthesize_answer(user_text, last_call[0], last_result)
                messages.append({"role": "assistant", "content": answer})
                return answer, messages[1:]
            if (
                last_result is None
                and REFUSAL_RE.search(visible[:250])
                and not LOCAL_ACTION_TOOLS.intersection(tools)
            ):
                # Refused without touching any tool (e.g. a digit in the
                # question pulled in core math tools it can't use) — the
                # web may still know. Never escalate local-action requests
                # (files, git, shell): the web cannot answer those.
                answer = escalate_to_search(user_text, verbose)
                if answer:
                    messages.append({"role": "assistant", "content": answer})
                    return answer, messages[1:]
            answer = visible or "(The model produced no answer before its token limit.)"
            messages.append({"role": "assistant", "content": answer})
            return answer, messages[1:]

        name, arguments = call
        if (name, arguments) == last_call and last_result is not None:
            # The model re-issued the identical call instead of answering:
            # synthesize the final answer in a clean, catalog-free turn.
            answer = synthesize_answer(user_text, name, last_result)
            messages.append({"role": "assistant", "content": answer})
            return answer, messages[1:]
        result = execute_tool(name, arguments, tools, verbose)
        last_call, last_result = (name, arguments), result

        messages.append(
            {"role": "assistant", "content": canonical_call(name, arguments)}
        )
        messages.append(
            {"role": "tool", "content": f"Result from {name}:\n{result}"}
        )
        if tools[name].get("direct") and not result.startswith("ERROR:"):
            messages.append({"role": "assistant", "content": result})
            return result, messages[1:]

    if last_result is not None:
        answer = synthesize_answer(user_text, last_call[0], last_result)
    else:
        answer = "(Stopped after too many tool calls.)"
    messages.append({"role": "assistant", "content": answer})
    return answer, messages[1:]


def parse_skill_option(value):
    if not value or value == "auto":
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if "all" in names:
        return list(SKILL_DESCRIPTIONS)
    unknown = sorted(set(names) - set(SKILL_DESCRIPTIONS))
    if unknown:
        raise ValueError(
            f"unknown skill pack(s): {', '.join(unknown)}; choose from "
            f"{', '.join(SKILL_DESCRIPTIONS)}"
        )
    if "core" not in names:
        names.insert(0, "core")
    return names


def print_skills():
    for name, description in SKILL_DESCRIPTIONS.items():
        tools = ", ".join(tools_for_skills([name]))
        print(f"{name:8} {description}\n         tools: {tools}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills",
        default="auto",
        help="comma-separated skill packs, 'all', or 'auto' (default)",
    )
    parser.add_argument(
        "--list-skills", action="store_true", help="show skill packs and exit"
    )
    parser.add_argument(
        "--quiet-tools", action="store_true", help="hide skill/tool trace output"
    )
    parser.add_argument("prompt", nargs="*", help="one-shot prompt")
    args = parser.parse_args()

    if args.list_skills:
        print_skills()
        return
    try:
        requested_skills = parse_skill_option(args.skills)
    except ValueError as error:
        parser.error(str(error))

    if args.prompt:
        answer, _ = run_turn(
            [],
            " ".join(args.prompt),
            requested_skills=requested_skills,
            verbose=not args.quiet_tools,
        )
        print(answer)
        return

    mode = args.skills if requested_skills else "auto-routed"
    print(f"minicpm-agent ({MODEL}) — {mode} skill packs")
    print("Type a question, /skills, /clear, or /bye.\n")
    history = []
    while True:
        try:
            user_text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "/bye"):
            break
        if user_text == "/skills":
            print_skills()
            continue
        if user_text == "/clear":
            history = []
            print("Conversation cleared.\n")
            continue
        try:
            answer, history = run_turn(
                history,
                user_text,
                requested_skills=requested_skills,
                verbose=not args.quiet_tools,
            )
            print(answer + "\n")
        except RuntimeError as error:
            print(f"ERROR: {error}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
