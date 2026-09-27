# minicpm-agent

A zero-dependency tool-calling agent that makes a **1-billion-parameter** local
model (MiniCPM5-1B on Ollama) genuinely useful: 30+ tools, deterministic
routing, and a harness that works around everything a tiny model gets wrong.

The Ollama registry model (`openbmb/minicpm5`) rejects the OpenAI-style
`tools` parameter ("does not support tools"), so this harness injects a
compact tool catalog into the system prompt and parses the model's native
XML-style function calls
(`<function name="..."><param name="...">...</param></function>`).

## Features

- **30+ tools in 11 skill packs**: math, dates, files (sandboxed), system
  health, web search/fetch, weather, notes/memory, unit conversion, stocks,
  crypto, currency, read-only git, CSV stats, network checks, and an opt-in shell.
- **Deterministic keyword routing** so the 1B model only ever sees a handful
  of relevant tools, plus narrowing of easily-confused tools.
- **Direct answers without the model** for unmistakable requests (disk
  space, CSV column stats), which are faster and never hallucinated.
- **Multi-backend web search** (Mojeek, Bing, Bing via r.jina.ai,
  DuckDuckGo) with off-topic-result rejection and page-excerpt enrichment.
- **Refusal/echo recovery**: "I don't have access..." answers escalate to a
  web search and a clean synthesis turn.
- **Stdlib only**: no pip installs needed to run the agent.

## Quick start

1. Install [Ollama](https://ollama.com/download) and start it (`ollama serve`).
2. Pull the model (about 700 MB):

   ```bash
   ollama pull openbmb/minicpm5
   ```

3. Run the agent with Python 3.10+:

   ```bash
   git clone <this repo> && cd minicpm-agent
   python3 agent.py "What is 345 * 453?"
   ```

## Usage

```bash
python3 agent.py "What is 345 * 453?"            # one-shot
python3 agent.py                                  # interactive REPL
python3 agent.py --skills files "Find TODO here"  # force a skill pack
python3 agent.py --skills all                     # everything enabled
python3 agent.py --list-skills                    # show packs and tools
python3 agent.py --quiet-tools "..."              # hide the tool trace
```

In the REPL: `/skills`, `/clear`, `/bye`.

### Example session

A deterministic request against the synthetic `examples/sample-sales.csv`
(the router picks the `data` pack, and the request is answered directly by
`csv_column_stats` without a model round-trip):

```text
$ cd examples
$ python3 ../agent.py "What is the total of the qty column in sample-sales.csv?"
  [skills] core, data -> percentage, calculator, current_datetime, csv_column_stats
  [tool] csv_column_stats({"path": "sample-sales.csv", "column": "qty"})
  [tool] -> qty: count 4, sum 28, min 3, max 12, mean 7
qty: count 4, sum 28, min 3, max 12, mean 7
```

## Architecture

- `agent.py`: the chat loop. Talks to Ollama's `/api/chat`, builds the
  system prompt, parses XML tool calls (`parse_tool_call`), executes tools,
  feeds results back, and handles re-synthesis and search escalation.
- `toolkit.py`: tool implementations, the tool registry (`TOOL_DEFINITIONS`), skill
  packs (`SKILL_TOOLS`), the keyword router (`auto_select_skills`), tool narrowing
  (`narrow_tools_for_request`), and model-free shortcuts
  (`direct_tool_call_for_request`).
- Why prompt-injected tools: the model can't take a `tools` param through
  Ollama, but it does emit well-formed XML calls when shown concrete
  examples, so the harness parses those instead of forcing JSON.
- Why deterministic routing: a 1B model shown 30 tools picks the wrong one.
  Regex routing keeps each catalog to a few tools, which is what makes the
  small model reliable.
- File tools are sandboxed to `MINICPM_WORKSPACE` (default: cwd). Notes from
  the memory tools are stored in `.minicpm-agent/notes.jsonl` in the workspace.

## Skill packs

Tools are grouped into packs; a deterministic keyword router
(`auto_select_skills`) picks packs per question so the 1B model sees a small
catalog instead of all 30+ tools at once. `narrow_tools_for_request` then
removes same-pack tools the model confuses, and
`direct_tool_call_for_request` answers unmistakable requests (disk space,
CSV column stats) without calling the model at all.

| Pack    | Tools |
|---------|-------|
| core    | percentage, calculator, current_datetime (always on) |
| files   | read_file, list_directory, write_file, search_files |
| system  | disk_usage, system_info, process_list |
| web     | web_search, fetch_url, fetch_json, weather |
| memory  | remember, recall, list_notes |
| convert | unit_convert, days_between, date_add |
| finance | stock_price (Yahoo), crypto_price (CoinGecko), currency_convert (ECB) |
| git     | git_status, git_log, git_diff (read-only) |
| data    | csv_inspect, csv_column_stats |
| network | public_ip, dns_lookup, http_status, ping |
| shell   | shell (opt-in via keywords like "run this command") |

All finance/network data comes from keyless public APIs (Yahoo Finance
chart API, CoinGecko simple price, frankfurter.app ECB rates, ipify,
wttr.in) — no accounts or API keys needed.

## Configuration (environment variables)

- `MINICPM_MODEL` — Ollama model name (default `openbmb/minicpm5`)
- `OLLAMA_CHAT_URL` — chat endpoint (default `http://localhost:11434/api/chat`)
- `MINICPM_WORKSPACE` — sandbox root for file tools (default: cwd)
- `MINICPM_ALLOW_ANY_PATH=1` — disable the file-tool sandbox

## Field notes (things that bit us)

- The model emits **XML tool calls**, not JSON — parse
  `<function name=...>`, don't fight it with JSON formats.
- Prompt examples must be **concrete**: with a placeholder like
  `<function name="TOOL">`, the 1B model copies `TOOL` literally.
- DuckDuckGo bot-challenges datacenter IPs; `web_search` uses **Bing first**
  and decodes its base64 `u=a1...` redirect links.
- wttr.in serves HTML to browser-like User-Agents; plain/curl-ish UAs get text.
- The model tends to repeat a successful tool call; the harness feeds back an
  error on malformed calls and `direct` tools short-circuit to the result.
- Don't mark step-1 tools (`current_datetime`, `csv_inspect`) as `direct` —
  short-circuiting them kills multi-step chains like "days until Christmas".
- Given sample CSV rows, the model does mental math on them (and gets it
  wrong); numeric column questions are narrowed to `csv_column_stats` only.
- Stooq's CSV quote endpoint 404s now; `stock_price` uses Yahoo's
  `/v8/finance/chart/` JSON API instead (works keyless with a browser UA).
- Shown a tool catalog, the model refuses general knowledge it actually has
  ("I only have calculator..."). Core-only questions with no math/time signal
  skip the tool harness entirely (`question_needs_tools`).
- Conflicting prompt rules are poison at 1B: "answer directly from knowledge"
  next to "MUST call web_search" made the model echo the system prompt.
- Never embed instructions in tool-result messages — the model parrots them
  verbatim into its answer ("Do not call any more tools..."). Synthesis runs
  as a separate catalog-free turn instead.
- Search snippets rarely contain the answer; `web_search` appends plain-text
  excerpts of the top two result pages so answers are grounded in content.
- Bing (direct and via r.jina.ai) sometimes serves someone else's cached
  SERP ("Used Nissan Sentra" for a population query); results sharing no
  significant query term are rejected and the next backend is tried.
- When routing decides a question is a search (only non-core tool is
  web_search), the harness runs the search up front — the model sometimes
  refuses to call tools on its own.
- If the model echoes the tool result or refuses despite having one, the
  answer is re-synthesized in a clean turn; core math tools are dropped from
  search questions (the model fed 'California' into percentage()).
- If the model answers "I don't have access to..." on a no-tools or
  tool-less turn, the harness escalates: web search → excerpt scan →
  synthesis (`escalate_to_search`).
- A bare digit is not a math signal — "Claude 5" routed to core tools and
  the model answered with the percentage example's numbers (549.6).
- Natural-language queries do badly on keyword engines: question-shaped
  queries are keywordized with entities first ("Claude 5 Anthropic models
  family"), else 'models' surfaces Vogue instead of Anthropic.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q
```

The tests mock the model and HTTP calls, so they need neither Ollama nor
network access.

## Status / limitations

- A personal experiment in getting the most out of a 1B model; answers are
  only as good as MiniCPM5-1B plus the tools, and multi-step reasoning is
  still fragile.
- Web search scrapes public search engine HTML (Mojeek, Bing, DuckDuckGo) and
  can break when those pages change or bot-challenge your IP. Finance,
  weather and network tools call public keyless APIs (Yahoo Finance chart
  API, CoinGecko, frankfurter.app, wttr.in, ipify).
- The `shell` pack runs arbitrary commands; it is only enabled when the
  request explicitly asks to run a command, but treat it with care.
- Tested on Linux with Python 3.10+.

## License

MIT, see [LICENSE](LICENSE).
