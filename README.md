<div align="center">

# jyje/pilot-deepagents-dynamic-subagents

<img width="280" src="https://raw.githubusercontent.com/langchain-ai/deepagentsjs/refs/heads/main/.github/images/logo-light.svg#gh-light-mode-only" alt="DeepAgents logo (light mode)"/>
<img width="280" src="https://raw.githubusercontent.com/langchain-ai/deepagentsjs/refs/heads/main/.github/images/logo-dark.svg#gh-dark-mode-only" alt="DeepAgents logo (dark mode)"/>

🚀 Pilot project for validating **Dynamic Subagents** in LangChain Deep Agents

[![GitHub Repo stars](https://img.shields.io/github/stars/jyje/pilot-deepagents-dynamic-subagents?style=social)](https://github.com/jyje/pilot-deepagents-dynamic-subagents)

[English](README.md)

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

## Formatting and Reference Alignment

This README formatting intentionally follows the style direction used in the user's related pilot repositories, including:

- centered project header layout
- image-first presentation style
- concise top summary section

Reference projects:

- [`jyje/pilot-deep-agents`](https://github.com/jyje/pilot-deep-agents)
- [`jyje/pilot-deepagents-rubrics`](https://github.com/jyje/pilot-deepagents-rubrics)

## Agent Skill Reference

Environment initialization follows GitHub Copilot cloud-agent customization guidance through `copilot-setup-steps.yml`.
