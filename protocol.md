## Research Loop Protocol

When asked to start a research loop, follow this protocol. You are running in a **live tmux session** — the user can watch your progress, detach/reattach, and provide feedback.

### Operating Modes

The loop supports two modes, set by the user at startup or changed between iterations:

- **Autonomous mode**: After each iteration, analyze results and auto-decide the next experiment. Continue without waiting for user input. Stop only when the goal is reached, the metric plateaus for 3+ iterations, or you are unsure what to try.
- **Interactive mode** (default): After each iteration, present a summary and **wait for user feedback** before continuing.

The user can switch modes at any time by saying "continue autonomously" or "wait for my feedback".

### Core Functions

The loop is built around two Claude API-powered Python functions:

- **Function A** (`function_a.py`): Literature search. Takes a topic → calls Claude API with web_search → returns ranked papers as JSON.
- **Function B** (`function_b.py`): Code implementation. Takes papers or instructions → calls Claude API agentic loop with file read/edit tools → modifies the codebase.

These are called by the main Claude Code session (you) via Bash. You orchestrate, they execute.

### progress.md

The user creates `progress.md` to define the research goal. The agent auto-updates it with tracking data below the sentinel line. **Never edit the user's goal section above the sentinel.**

**User creates:**
```markdown
# Research Goal

Improve heart segmentation 3D Dice above 0.92 using adapter architecture changes.

## Constraints
- Keep parameter count under 1M
- Must converge within 200 epochs
```

**Agent updates everything below** `<!-- AGENT PROGRESS BELOW -->` automatically via `state.py`.

### Git Tracking

Every iteration is tracked as a **git branch** with structured commits. This gives full traceability — `git log` shows every hypothesis and result, `git diff` between iterations shows exactly what changed.

**Branch structure:**
```
main                          ← always has the best-performing code
├── iter/1-spd-rank-increase  ← branch per iteration
├── iter/2-tokenwise-film     ← each has 2 commits: code + results
└── iter/3-bias-scale-tuning
```

**Each iteration creates 2 commits:**
1. **Code commit** (before experiment) — records hypothesis, change, papers
2. **Results commit** (after experiment) — records metrics, delta vs baseline

**Best iteration merges to main**, so `main` always reflects the top configuration.

### Setup (first time only)

1. Read the user's `progress.md` to understand the goal.
2. Initialize state from it:
   ```
   python -m research_agent.state init --progress progress.md --metric "<primary_metric>"
   ```
3. Identify baseline: read existing results, record in state:
   ```
   python -m research_agent.state set-baseline --checkpoint "<path>" --metrics '{"metric": value}'
   ```
   (This auto-updates `progress.md`.)

### Each Iteration

1. **Read state** — recover full context after compression:
   ```
   python -m research_agent.state read
   ```

2. **Decide what to try** — the change can come from different sources:
   - **User instruction** — the user told you exactly what to try (skip Function A, go to Function B with `--instruction`).
   - **Previous results** — analysis of the last iteration suggests an obvious next step (skip Function A).
   - **New technique needed** — call Function A first.

3. **(Optional) Function A — literature search:**
   ```
   python research_agent/function_a.py "orthogonal adapter fine-tuning" \
     results/search_iter3.json --state state.json
   ```
   Or auto-generate the topic from the last iteration:
   ```
   python research_agent/function_a.py --auto results/search_iter3.json --state state.json
   ```
   - Calls Claude API with web_search to find papers.
   - Auto-deduplicates against papers already used in previous iterations.
   - Skip when the user gives a specific instruction or the next step is obvious.

4. **Function B — implement the change:**
   ```
   # From papers:
   python research_agent/function_b.py --papers results/search_iter3.json \
     --project-dir . --state state.json \
     --files models/sam/modeling/common.py

   # Or from direct instruction:
   python research_agent/function_b.py --instruction "increase spd_rank to 8" \
     --project-dir . --state state.json
   ```
   - Claude API agentic loop: reads code, plans edits, modifies files.
   - Returns JSON summary: `{hypothesis, change_summary, files_modified, papers_used}`.

5. **Review changes** — read what Function B modified, verify correctness.

6. **Create branch + commit code:**
   ```
   python -m research_agent.git_ops branch-start --iteration 3 --change "..."
   python -m research_agent.git_ops commit-code --iteration 3 \
     --hypothesis "..." --change "..." --papers "..." \
     --checkpoint "checkpoints/exp_..."
   python -m research_agent.git_ops push
   ```

7. **Execute experiment** — launch in background:
   ```
   bash research_agent/run_and_wait.sh <script> <checkpoint_dir>
   ```

8. **Poll** — check completion every ~10 minutes:
   ```
   test -f <checkpoint_dir>/.done && cat <checkpoint_dir>/.done || echo RUNNING
   ```

9. **Analyze** — read results, compare with baseline and previous best.

10. **Update state** — record iteration (auto-updates `progress.md`):
    ```
    python -m research_agent.state add-iteration \
      --hypothesis "..." --change "..." --checkpoint "..." \
      --metric-name <name> --metric-value <value> \
      --feedback "..."
    ```

11. **Commit results:**
    ```
    python -m research_agent.git_ops commit-results --iteration 3 --state state.json
    ```

12. **If new best → merge to main** and push:
    ```
    python -m research_agent.git_ops merge-best --state state.json
    python -m research_agent.git_ops push
    ```

13. **Summarize** — present results and proposed next steps to user.

14. **Next iteration decision:**
    - **Interactive mode:** Wait for user feedback before continuing.
    - **Autonomous mode:** Auto-decide the next step based on results:
      - **Improved?** → Build on it (vary the same knob, combine with another).
      - **Regressed?** → Revert to best config, try a different direction.
      - **Plateaued (3+ iters)?** → Call Function A for fresh ideas, or stop and ask the user.
      - **Goal reached?** → Stop and present final summary.
      - **Unsure?** → Stop and ask the user for direction.

### Autonomous Decision Guidelines

When running autonomously, use these heuristics to decide what to try next:

1. **Read the full iteration history** from state.json — look for trends, not just the last result.
2. **If the last change helped:** Try a variant (e.g., helped with rank 4 → try rank 8) or combine it with the second-best change.
3. **If the last change hurt or was neutral:** Revert to the best config and try something orthogonal (different component, different technique).
4. **After 3+ iterations without improvement:** Call Function A with `--auto` to search for new ideas. If still stuck, stop and ask the user.
5. **Log your reasoning** in the `--feedback` field so the user can review your thought process.

### Git Commands Reference

| Command | When | What it does |
|---------|------|-------------|
| `branch-start --iteration N --change "..."` | Before coding | Creates `iter/N-slug` from main |
| `commit-code --iteration N --hypothesis "..." --change "..."` | After coding, before experiment | Commits code with structured message |
| `commit-results --iteration N --state state.json` | After experiment results | Commits with metrics + delta vs baseline |
| `merge-best --state state.json` | When new best found | Merges best branch into main |
| `push` | After commit or merge | Pushes current branch to remote |
| `push-all` | Periodically | Pushes main + all iter branches |
| `log` | Anytime | Shows all iteration commits |

### Rules

- **ONE principal change per iteration** — isolate variables for clean comparison.
- **NEVER overwrite previous checkpoints** — each iteration gets a unique checkpoint directory.
- **ALWAYS commit before running experiments** — code changes must be tracked in git before any long-running job starts.
- **Re-read state.json** at the start of every iteration to recover context.
- **Primary metric drives decisions**; always report secondary metrics too.
- **Save experiment scripts** — each iteration's script should be reproducible.
- **Cite papers** — when a technique comes from literature, note the reference.
- **Never edit the user's goal section** in `progress.md`.
- **Push after every commit** — keep GitLab in sync so nothing is lost.
- **Present clear summaries** — the user is watching in tmux, make status updates readable.
- **Review Function B's changes** — always verify what it modified before committing.
