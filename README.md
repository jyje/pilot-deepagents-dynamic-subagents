<div align="center">

# jyje/pilot-deepagents-dynamic-subagents

<img width="320" src="https://raw.githubusercontent.com/langchain-ai/deepagentsjs/refs/heads/main/.github/images/logo-light.svg" alt="DeepAgents logo"/>

🚀 Pilot project for validating **Dynamic Subagents** in LangChain Deep Agents with **NVIDIA NIM**

[![GitHub Repo stars](https://img.shields.io/github/stars/jyje/pilot-deepagents-dynamic-subagents?style=social)](https://github.com/jyje/pilot-deepagents-dynamic-subagents)
[![NVIDIA NIM](https://img.shields.io/badge/AI%20Provider-NVIDIA%20NIM-76B900)](https://build.nvidia.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org)

---

**Found this useful? Please give it a ⭐ — it helps others find it.**

</div>

## Overview

This repository is a focused validation pilot for the Dynamic Subagents concept introduced in LangChain Deep Agents:

- [Introducing Dynamic Subagents in Deep Agents](https://www.langchain.com/blog/introducing-dynamic-subagents-in-deep-agents)

The goal is to verify that the `deepagents` SDK's subagent delegation feature works end-to-end when **NVIDIA NIM** (`ChatNVIDIA`) is used as the LLM backend instead of Anthropic Claude.

## What Is Validated

| Claim | Verification Method |
|-------|---------------------|
| `ChatNVIDIA` is accepted by `create_deep_agent` | Unit test + notebook cell |
| Subagents are registered as callable tools | Unit test |
| Orchestrator delegates to `researcher` subagent | Integration test + notebook |
| Orchestrator delegates to `reviewer` subagent | Integration test + notebook |
| Full pipeline returns a coherent final answer | Integration test + notebook |

## Architecture

```
create_deep_agent (orchestrator: meta/llama-3.3-70b-instruct)
    │
    ├── researcher subagent (nvidia/nemotron-3-super-120b-a12b)
    │       Gathers facts and provides evidence-based notes.
    │
    └── reviewer subagent (meta/llama-3.1-8b-instruct)
            Critiques drafts and identifies logical gaps.
```

All models are served via [NVIDIA API Catalog](https://build.nvidia.com) or a self-hosted NIM container.

## Quick Start

```bash
cd src
cp .env.sample .env
# Fill NVIDIA_API_KEY in .env  (get yours at https://build.nvidia.com)

# Install base dependencies
uv sync

# Validate environment & API connectivity
uv run python doctor.py

# Run CLI demo
uv run python main.py

# Run unit tests (no API key required)
uv run pytest tests/ -v -m "not integration"

# Run all tests including integration (requires NVIDIA_API_KEY)
uv run pytest tests/ -v

# Launch Jupyter notebook for interactive verification
uv sync --extra notebook
uv run jupyter notebook notebooks/01-verify-dynamic-subagents.ipynb

# LangGraph Studio (optional)
uv sync --extra studio
uv run langgraph dev --tunnel
```

## Repository Structure

```
src/
├── main.py          — CLI demo: orchestrator + researcher + reviewer subagents
├── graph.py         — Module-level agent export for LangGraph Studio
├── doctor.py        — NVIDIA NIM environment & connectivity diagnostics
├── pyproject.toml   — Pinned dependencies (deepagents + langchain-nvidia-ai-endpoints)
├── .env.sample      — Required/optional environment variables
├── langgraph.json   — LangGraph Studio graph mapping
├── tests/
│   └── test_dynamic_subagents.py  — Unit + integration test suite
└── notebooks/
    └── 01-verify-dynamic-subagents.ipynb  — Interactive verification notebook
```

## Initialization Baseline

This repository uses `.github/workflows/copilot-setup-steps.yml` for deterministic cloud-agent initialization.

Current baseline setup:

1. Checkout repository source.
2. Verify baseline toolchain availability (`git`, `python3`).

## Reference Projects

- [`jyje/pilot-deep-agents`](https://github.com/jyje/pilot-deep-agents) — baseline Deep Agents pilot structure.
- [`jyje/pilot-deepagents-rubrics`](https://github.com/jyje/pilot-deepagents-rubrics) — style and validation-documentation reference.
- [`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents/tree/main/examples/nvidia_deep_agent) — official NVIDIA Deep Agent example.
