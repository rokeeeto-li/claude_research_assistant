#!/usr/bin/env python3
"""Function A: Literature search powered by Claude API.

Takes a research topic and project context. Calls Claude API with the
web_search tool to find, evaluate, and rank relevant papers.

Requires: ANTHROPIC_API_KEY environment variable.

Usage:
    python research_agent/function_a.py "topic" output.json
    python research_agent/function_a.py "topic" output.json --state state.json
    python research_agent/function_a.py --auto output.json --state state.json

The --auto flag generates the search topic automatically from the last
iteration's feedback in state.json (no manual topic needed).
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are a research paper search agent. Your job is to find academic papers \
relevant to a specific topic within an ML project's context.

Instructions:
1. Use the web_search tool to search for papers (3-5 queries from different angles).
2. For each result, evaluate relevance to the project (1-5 scale).
3. Only keep papers scoring 3+.
4. Return ONLY a JSON array (no markdown fences, no commentary). Each element:
   {
     "title": "Paper Title",
     "authors": "First Author et al.",
     "year": 2024,
     "abstract": "First 2-3 sentences...",
     "url": "https://...",
     "arxiv_id": "2401.12345 or empty string",
     "relevance": 5,
     "relevance_reason": "Why this paper matters",
     "key_idea": "Main takeaway applicable to our work"
   }
   Sort by relevance desc, then year desc.

Rules:
- Quality over quantity: 3-8 highly relevant papers.
- No hallucinated papers: only include papers you actually found.
- Valid JSON only in your final response.\
"""


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    return key


def _api_call(messages: list[dict], system: str,
              tools: list[dict] | None = None) -> dict:
    """Call the Anthropic Messages API and return the response."""
    body: dict = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": _api_key(),
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        print(f"API error {e.code}: {err_body[:500]}", file=sys.stderr)
        sys.exit(1)


def _build_context(state_path: str | None) -> str:
    """Build project context from state.json."""
    if not state_path or not Path(state_path).exists():
        return ""

    try:
        state = json.loads(Path(state_path).read_text())
    except (json.JSONDecodeError, IOError):
        return ""

    parts = []
    if state.get("goal"):
        parts.append(f"Project goal: {state['goal']}")
    if state.get("primary_metric"):
        parts.append(f"Primary metric: {state['primary_metric']}")
    bl = state.get("baseline")
    if bl and bl.get("metrics"):
        parts.append(f"Baseline: {json.dumps(bl['metrics'])}")
    best = state.get("best")
    if best and best.get("metrics"):
        parts.append(f"Best so far (iter {best.get('iteration', '?')}): "
                      f"{json.dumps(best['metrics'])}")

    # Previous papers (for deduplication)
    seen_papers = []
    for it in state.get("iterations", []):
        for p in it.get("papers_referenced", []):
            seen_papers.append(p)

    # Recent iterations
    for it in state.get("iterations", [])[-5:]:
        parts.append(f"Iter {it['id']}: {it.get('change_summary', '')} "
                      f"-> {json.dumps(it.get('metrics', {}))} "
                      f"| feedback: {it.get('feedback', '')}")

    if seen_papers:
        parts.append(f"\nPapers already used (DO NOT return these): "
                      f"{', '.join(seen_papers)}")

    return "\n".join(parts)


def _auto_topic(state_path: str) -> str:
    """Generate search topic from last iteration's feedback."""
    if not Path(state_path).exists():
        print("Error: --auto requires --state with existing state.json",
              file=sys.stderr)
        sys.exit(1)

    state = json.loads(Path(state_path).read_text())
    iters = state.get("iterations", [])
    goal = state.get("goal", "")

    if not iters:
        # No iterations yet — search based on the goal
        return f"techniques for: {goal}"

    last = iters[-1]
    feedback = last.get("feedback", "")
    change = last.get("change_summary", "")
    hypothesis = last.get("hypothesis", "")

    # Build a topic from the last iteration context
    topic_parts = []
    if feedback:
        topic_parts.append(f"Based on result: {feedback}")
    if change:
        topic_parts.append(f"Last change: {change}")
    topic_parts.append(f"Goal: {goal}")

    return ". ".join(topic_parts)


def _extract_json_array(text: str) -> list | None:
    """Extract JSON array from text, handling markdown fences."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    for pattern in [r"```json\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```", r"(\[.*\])"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
    return None


def run_search(topic: str, output_path: str,
               state_path: str | None = None) -> list | None:
    """Run Claude-powered paper search with web_search tool."""
    context = _build_context(state_path)

    user_msg = f"Search topic: {topic}"
    if context:
        user_msg += f"\n\nProject context:\n{context}"
    user_msg += "\n\nFind relevant papers. Return ONLY a JSON array."

    # Web search is a server-side tool
    tools = [{"type": "web_search_20250305", "name": "web_search",
              "max_uses": 10}]

    messages = [{"role": "user", "content": user_msg}]

    print(f"Searching: {topic}", file=sys.stderr)
    print(f"Using Claude API ({MODEL}) with web_search", file=sys.stderr)

    # Agentic loop: Claude may call web_search multiple times
    for turn in range(15):
        resp = _api_call(messages, SYSTEM_PROMPT, tools)
        stop = resp.get("stop_reason", "")

        # Collect text and tool_use blocks
        text_parts = []
        tool_results = []

        for block in resp.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            # Server-side web_search results are handled automatically
            # by the API — we don't need to execute them ourselves

        # If stop_reason is "end_turn", Claude is done
        if stop == "end_turn":
            full_text = "\n".join(text_parts)
            papers = _extract_json_array(full_text)
            if papers is not None:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                print(f"Found {len(papers)} papers -> {output_path}",
                      file=sys.stderr)
                return papers
            else:
                print("Warning: Could not parse JSON from response.",
                      file=sys.stderr)
                print(f"Raw: {full_text[:500]}", file=sys.stderr)
                Path(output_path).write_text(full_text)
                return None

        # If still going (tool use), add assistant message and continue
        messages.append({"role": "assistant", "content": resp["content"]})

        # For server-side tools, results come back in the response
        # Just continue the loop if stop_reason is "tool_use"
        if stop == "tool_use":
            # Server-side web_search results are embedded in content
            # We need to pass them back — but for server-side tools,
            # the API handles this automatically. Just continue.
            # Add a user message to prompt continuation
            messages.append({"role": "user",
                             "content": "Continue searching and compile results."})
            continue

        # Unknown stop reason
        print(f"Unexpected stop_reason: {stop}", file=sys.stderr)
        break

    print("Search did not complete within turn limit.", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Function A: Claude-powered literature search.",
        epilog="""Examples:
  python function_a.py "Householder orthogonal adapters ViT" results.json
  python function_a.py "PEFT medical segmentation" results.json --state state.json
  python function_a.py --auto results.json --state state.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("topic", nargs="?", default=None,
                        help="Search topic (omit if using --auto)")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument("--state", default=None,
                        help="Path to state.json for context + dedup")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-generate topic from last iteration feedback")
    args = parser.parse_args()

    if args.auto:
        if not args.state:
            print("Error: --auto requires --state", file=sys.stderr)
            sys.exit(1)
        topic = _auto_topic(args.state)
        print(f"Auto-generated topic: {topic}", file=sys.stderr)
    elif args.topic:
        topic = args.topic
    else:
        print("Error: provide a topic or use --auto", file=sys.stderr)
        sys.exit(1)

    papers = run_search(topic, args.output, state_path=args.state)

    if papers is not None:
        json.dump(papers, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
