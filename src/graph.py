"""Module-level graph export for LangGraph Studio."""

import os
import warnings
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core._api import LangChainBetaWarning
from deepagents import create_deep_agent

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
load_dotenv()


def _make_model(env_key: str, default: str) -> ChatAnthropic:
    kwargs: dict = {
        "model": os.getenv(env_key, default),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "max_retries": 4,
    }
    if base_url := os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = base_url
    return ChatAnthropic(**kwargs)


researcher_subagent = {
    "name": "researcher",
    "description": "Researches background facts and returns concise findings.",
    "system_prompt": "You gather facts quickly and provide short evidence-based notes.",
    "model": _make_model("RESEARCHER_MODEL", "claude-haiku-4-5-20251001"),
    "tools": [],
}

reviewer_subagent = {
    "name": "reviewer",
    "description": "Reviews outputs for correctness and missing assumptions.",
    "system_prompt": "You critique drafts and identify logical gaps clearly.",
    "model": _make_model("REVIEWER_MODEL", "claude-haiku-4-5-20251001"),
    "tools": [],
}

agent = create_deep_agent(
    model=_make_model("MAIN_MODEL", "claude-sonnet-4-6"),
    system_prompt=(
        "You are an orchestrator. Use subagents when decomposition improves quality. "
        "Delegate fact-finding to researcher and validation to reviewer."
    ),
    subagents=[researcher_subagent, reviewer_subagent],
)
