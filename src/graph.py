"""Module-level graph export for LangGraph Studio."""

import os
import warnings
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core._api import LangChainBetaWarning
from deepagents import create_deep_agent

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
load_dotenv()


def _make_model(env_key: str, default: str) -> ChatNVIDIA:
    kwargs: dict = {
        "model": os.getenv(env_key, default),
        "max_retries": 4,
    }
    if api_key := os.getenv("NVIDIA_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.getenv("NVIDIA_BASE_URL"):
        kwargs["base_url"] = base_url
    return ChatNVIDIA(**kwargs)


researcher_subagent = {
    "name": "researcher",
    "description": (
        "Researches background facts and returns concise, evidence-based findings. "
        "Use for knowledge-intensive information gathering."
    ),
    "system_prompt": (
        "You gather facts quickly and provide short, structured notes. "
        "Focus on accuracy; cite reasoning where possible."
    ),
    "model": _make_model("RESEARCHER_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    "tools": [],
}

reviewer_subagent = {
    "name": "reviewer",
    "description": (
        "Reviews outputs for correctness, missing assumptions, and logical gaps. "
        "Use to validate or critique a draft before final delivery."
    ),
    "system_prompt": (
        "You critique drafts and identify logical gaps clearly. "
        "Be specific: quote the exact passage and explain the issue."
    ),
    "model": _make_model("REVIEWER_MODEL", "meta/llama-3.1-8b-instruct"),
    "tools": [],
}

agent = create_deep_agent(
    model=_make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct"),
    system_prompt=(
        "You are an orchestrator agent. "
        "Break complex requests into subtasks and delegate to specialized subagents. "
        "Always delegate fact-finding to the researcher subagent, "
        "then pass the draft to the reviewer subagent for validation "
        "before returning a consolidated final answer."
    ),
    subagents=[researcher_subagent, reviewer_subagent],
)
