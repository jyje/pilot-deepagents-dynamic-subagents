# pilot-deepagents-dynamic-subagents

## Purpose

This repository validates the Dynamic Subagents concept introduced in:

- https://www.langchain.com/blog/introducing-dynamic-subagents-in-deep-agents

## Validation Scope

This pilot focuses on:

1. Defining a clear validation target for Dynamic Subagents behavior.
2. Keeping all project materials and documentation in English.
3. Providing a baseline Copilot cloud-agent setup workflow for consistent initialization.

## Initialization Baseline

The repository now includes a Copilot setup workflow:

- `.github/workflows/copilot-setup-steps.yml`

This workflow provides a minimal, deterministic initialization path for cloud-agent sessions by checking out the repository and verifying baseline toolchain availability.

## Agent Skill Reference

This repository setup aligns with the GitHub Copilot cloud-agent customization guidance (Copilot setup steps workflow), which is the recommended skill path for environment initialization.