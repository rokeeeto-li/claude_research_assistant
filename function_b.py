#!/usr/bin/env python3
"""Function B: Code implementation powered by Claude API.

Takes paper results from Function A (or user instructions) and implements
changes in the codebase. Runs as an agentic loop — Claude reads code,
plans changes, and edits files through tool calls.

Requires: ANTHROPIC_API_KEY environment variable.

Usage:
    # From paper results:
    python research_agent/function_b.py --papers results.json --project-dir .

    # From direct instruction:
    python research_agent/function_b.py --instruction "increase spd_rank to 8" --project-dir .

    # With specific files to focus on:
    python research_agent/function_b.py --papers results.json --project-dir . \
        --files models/sam/modeling/common.py cfg.py

Output:
    - Modified files in the project directory
    - JSON summary to stdout: {"changes": [...], "hypothesis": "...", "papers_used": [...]}
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    return key


def _api_call(messages: list[dict], system: str,
              tools: list[dict]) -> dict:
    """Call the Anthropic Messages API."""
    import urllib.error
    import urllib.request

    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        print(f"API error {e.code}: {err_body[:500]}", file=sys.stderr)
        sys.exit(1)


# ── Tool definitions ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the project. Returns the file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to project root"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace a specific string in a file. The old_string must "
                       "match exactly (including whitespace). Use this for targeted "
                       "edits — read the file first to get exact content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to project root"
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string"
                }
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory (non-recursive by default).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to project root"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter (e.g. '*.py')"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_code",
        "description": "Search for a pattern in files (grep-like). Returns matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: project root)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob to filter files (e.g. '*.py')"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "done",
        "description": "Call this when you have finished implementing all changes. "
                       "Provide a summary of what was changed and why.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "The hypothesis for this change"
                },
                "change_summary": {
                    "type": "string",
                    "description": "Short summary of what was changed"
                },
                "files_modified": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files that were modified"
                },
                "papers_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Papers that inspired this change"
                }
            },
            "required": ["hypothesis", "change_summary", "files_modified"]
        }
    },
]


# ── Tool execution ───────────────────────────────────────────────────

def _resolve(project_dir: str, rel_path: str) -> Path:
    """Resolve a relative path within the project, preventing escape."""
    p = (Path(project_dir) / rel_path).resolve()
    proj = Path(project_dir).resolve()
    if not str(p).startswith(str(proj)):
        raise ValueError(f"Path escapes project: {rel_path}")
    return p


def exec_read_file(project_dir: str, args: dict) -> str:
    p = _resolve(project_dir, args["path"])
    if not p.exists():
        return f"Error: file not found: {args['path']}"
    if p.is_dir():
        return f"Error: {args['path']} is a directory, use list_files"
    try:
        content = p.read_text()
        if len(content) > 50000:
            return content[:50000] + f"\n... (truncated, {len(content)} chars total)"
        return content
    except Exception as e:
        return f"Error reading {args['path']}: {e}"


def exec_edit_file(project_dir: str, args: dict) -> str:
    p = _resolve(project_dir, args["path"])
    if not p.exists():
        return f"Error: file not found: {args['path']}"

    content = p.read_text()
    old = args["old_string"]
    new = args["new_string"]

    count = content.count(old)
    if count == 0:
        return (f"Error: old_string not found in {args['path']}. "
                f"Read the file first to get the exact content.")
    if count > 1:
        return (f"Error: old_string found {count} times in {args['path']}. "
                f"Use a longer/more specific string to match uniquely.")

    content = content.replace(old, new, 1)
    p.write_text(content)
    return f"OK: edited {args['path']} ({len(old)} chars -> {len(new)} chars)"


def exec_list_files(project_dir: str, args: dict) -> str:
    p = _resolve(project_dir, args["path"])
    if not p.exists():
        return f"Error: directory not found: {args['path']}"

    pattern = args.get("pattern", "*")
    try:
        files = sorted(p.glob(pattern))
        # Filter out __pycache__, .git, node_modules
        files = [f for f in files
                 if not any(skip in f.parts for skip in
                            ("__pycache__", ".git", "node_modules", ".ipynb_checkpoints"))]
        return "\n".join(str(f.relative_to(Path(project_dir).resolve()))
                         for f in files[:100])
    except Exception as e:
        return f"Error: {e}"


def exec_search_code(project_dir: str, args: dict) -> str:
    search_path = args.get("path", ".")
    p = _resolve(project_dir, search_path)
    pattern = args["pattern"]
    file_pattern = args.get("file_pattern", "")

    cmd = ["grep", "-rn", "--include", file_pattern or "*.py",
           "-E", pattern, str(p)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = r.stdout
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"
        return output or "No matches found."
    except Exception as e:
        return f"Error: {e}"


def execute_tool(project_dir: str, name: str, args: dict) -> str:
    """Execute a tool call and return the result string."""
    if name == "read_file":
        return exec_read_file(project_dir, args)
    elif name == "edit_file":
        return exec_edit_file(project_dir, args)
    elif name == "list_files":
        return exec_list_files(project_dir, args)
    elif name == "search_code":
        return exec_search_code(project_dir, args)
    elif name == "done":
        return "DONE"
    else:
        return f"Error: unknown tool {name}"


# ── Main agent loop ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an ML research engineer. You receive paper findings or direct \
instructions and implement changes in an existing codebase.

Rules:
1. Read the relevant code files FIRST to understand the current implementation.
2. Make ONE focused change (do not refactor unrelated code).
3. Use edit_file for targeted edits — match the exact existing text.
4. After implementing, call the "done" tool with a summary.
5. If you cannot implement the change, call "done" explaining why.
6. Keep changes minimal and surgical — don't rewrite entire files.\
"""


def run_implementation(papers_path: str | None, instruction: str | None,
                       project_dir: str, focus_files: list[str] | None,
                       state_path: str | None) -> dict | None:
    """Run the agentic code implementation loop."""

    # Build the user message
    parts = []

    if papers_path and Path(papers_path).exists():
        papers = json.loads(Path(papers_path).read_text())
        # Show top 3 most relevant papers
        top = sorted(papers, key=lambda p: p.get("relevance", 0), reverse=True)[:3]
        parts.append("## Papers to implement from:\n")
        for p in top:
            parts.append(f"### {p['title']} ({p.get('year', '?')})")
            parts.append(f"Key idea: {p.get('key_idea', p.get('abstract', '')[:200])}")
            parts.append(f"Relevance: {p.get('relevance', '?')}/5")
            parts.append("")

    if instruction:
        parts.append(f"## Instruction\n{instruction}\n")

    # Add project context from state
    if state_path and Path(state_path).exists():
        try:
            state = json.loads(Path(state_path).read_text())
            parts.append(f"## Project context")
            parts.append(f"Goal: {state.get('goal', 'N/A')}")
            parts.append(f"Primary metric: {state.get('primary_metric', 'N/A')}")
            bl = state.get("baseline")
            if bl and bl.get("metrics"):
                parts.append(f"Baseline: {json.dumps(bl['metrics'])}")
            best = state.get("best")
            if best and best.get("metrics"):
                parts.append(f"Best (iter {best.get('iteration')}): "
                              f"{json.dumps(best['metrics'])}")
            # Last iteration
            iters = state.get("iterations", [])
            if iters:
                last = iters[-1]
                parts.append(f"Last change: {last.get('change_summary', 'N/A')} "
                              f"-> {json.dumps(last.get('metrics', {}))}")
                parts.append(f"Feedback: {last.get('feedback', 'N/A')}")
            parts.append("")
        except (json.JSONDecodeError, IOError):
            pass

    if focus_files:
        parts.append(f"## Key files to examine\n" +
                      "\n".join(f"- {f}" for f in focus_files))

    parts.append("\n## Task")
    parts.append("Read the relevant code, then implement ONE change based on "
                 "the papers/instruction above. Call the 'done' tool when finished.")

    user_msg = "\n".join(parts)
    messages = [{"role": "user", "content": user_msg}]

    print(f"Starting implementation agent ({MODEL})...", file=sys.stderr)

    for turn in range(30):  # Max 30 API calls
        resp = _api_call(messages, SYSTEM_PROMPT, TOOLS)
        stop = resp.get("stop_reason", "")
        content = resp.get("content", [])

        # Check for done tool or extract tool calls
        tool_calls = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(block)

        # If no tool calls and end_turn, Claude is done (shouldn't happen
        # without calling done tool, but handle gracefully)
        if stop == "end_turn" and not tool_calls:
            text = " ".join(b.get("text", "") for b in content
                            if b.get("type") == "text")
            print(f"Agent ended without calling done: {text[:200]}",
                  file=sys.stderr)
            return {"hypothesis": "", "change_summary": text[:200],
                    "files_modified": [], "papers_used": []}

        # Execute tool calls
        messages.append({"role": "assistant", "content": content})
        tool_results = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("input", {})

            print(f"  [{turn+1}] {name}({json.dumps(args)[:100]})",
                  file=sys.stderr)

            if name == "done":
                # Implementation complete
                result = {
                    "hypothesis": args.get("hypothesis", ""),
                    "change_summary": args.get("change_summary", ""),
                    "files_modified": args.get("files_modified", []),
                    "papers_used": args.get("papers_used", []),
                }
                print(f"Done: {result['change_summary']}", file=sys.stderr)
                return result

            result_text = execute_tool(project_dir, name, args)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

    print("Agent did not finish within turn limit.", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Function B: Claude-powered code implementation.",
        epilog="""Examples:
  python function_b.py --papers results.json --project-dir /path/to/project
  python function_b.py --instruction "increase spd_rank to 8" --project-dir .
  python function_b.py --papers results.json --project-dir . --files common.py cfg.py
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--papers", default=None,
                        help="Path to papers JSON from function_a")
    parser.add_argument("--instruction", default=None,
                        help="Direct instruction (alternative to papers)")
    parser.add_argument("--project-dir", default=".",
                        help="Project root directory (default: cwd)")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Key files to focus on")
    parser.add_argument("--state", default=None,
                        help="Path to state.json for context")
    args = parser.parse_args()

    if not args.papers and not args.instruction:
        print("Error: provide --papers or --instruction", file=sys.stderr)
        sys.exit(1)

    result = run_implementation(
        args.papers, args.instruction, args.project_dir,
        args.files, args.state,
    )

    if result:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
