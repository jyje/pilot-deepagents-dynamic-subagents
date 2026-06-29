<div align="center">

# jyje/pilot-deepagents-dynamic-subagents

<img width="320" src="https://raw.githubusercontent.com/langchain-ai/deepagentsjs/refs/heads/main/.github/images/logo-light.svg" alt="DeepAgents logo"/>

🚀 Pilot project for validating **Dynamic Subagents** in LangChain Deep Agents

[![GitHub Repo stars](https://img.shields.io/github/stars/jyje/pilot-deepagents-dynamic-subagents?style=social)](https://github.com/jyje/pilot-deepagents-dynamic-subagents)

---

**Found this useful? Please give it a ⭐ — it helps others find it.**

</div>

## Overview

This repository is a focused validation pilot for the Dynamic Subagents concept introduced in LangChain:

- [Introducing Dynamic Subagents in Deep Agents](https://www.langchain.com/blog/introducing-dynamic-subagents-in-deep-agents)

The goal is to structure, run, and document a practical verification path for dynamic delegation behavior in Deep Agents.

## Validation Focus

This pilot validates whether Dynamic Subagents can be used in a clear and repeatable way for:

1. Dynamic task delegation from a primary agent to specialized subagents.
2. Observable and explainable subagent execution behavior.
3. Practical criteria for deciding when dynamic subagents improve workflow quality.

## Initialization Baseline

This repository uses:

- `.github/workflows/copilot-setup-steps.yml`

for deterministic cloud-agent initialization.

Current baseline setup:

1. Checkout repository source.
2. Verify baseline toolchain availability (`git`, `python3`).

## Base Implementation (Applied)

Following the baseline style from `pilot-deepagents-rubrics`, this repository now includes a runnable Dynamic Subagents implementation under `/src`:

- `src/main.py` — CLI demo run with an orchestrator + two subagents (`researcher`, `reviewer`)
- `src/graph.py` — module-level `agent` export for LangGraph Studio
- `src/doctor.py` — environment and API connectivity diagnostic checks
- `src/pyproject.toml` — pinned pilot dependencies
- `src/.env.sample` — required/optional runtime environment variables
- `src/langgraph.json` — LangGraph Studio graph mapping

## Quick Start

```bash
cd src
cp .env.sample .env
# Fill ANTHROPIC_API_KEY in .env

# Install dependencies
uv sync

# Validate setup
uv run python doctor.py

# Run pilot demo
uv run python main.py

# LangGraph Studio (optional)
uv sync --extra studio
uv run langgraph dev --tunnel
```

## Reference Projects

- [`jyje/pilot-deep-agents`](https://github.com/jyje/pilot-deep-agents) — baseline Deep Agents pilot structure used as the setup and repository organization reference.
- [`jyje/pilot-deepagents-rubrics`](https://github.com/jyje/pilot-deepagents-rubrics) — style and validation-documentation reference for presenting experiment intent and outcomes.

## Environment Setup Reference

Environment initialization follows GitHub Copilot cloud-agent customization guidance through `copilot-setup-steps.yml`.
