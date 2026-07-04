import os
import warnings
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from deepagents import create_deep_agent

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
load_dotenv()


def _make_model(env_key: str, default: str) -> ChatNVIDIA:
    kwargs: dict = {
        "model": os.getenv(env_key, default),
    }
    if api_key := os.getenv("NVIDIA_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.getenv("NVIDIA_BASE_URL"):
        kwargs["base_url"] = base_url
    return ChatNVIDIA(**kwargs)


def _print_result(result: dict) -> None:
    W   = 60
    SEP = "─" * W

    def box(label: str) -> None:
        print(f"\n┌{'─' * (W - 2)}┐")
        print(f"│ {label:<{W - 4}} │")
        print(f"└{'─' * (W - 2)}┘")

    ROLE_STYLE = {
        "human":  ("\033[1;34m", "👤 Human"),
        "ai":     ("\033[1;32m", "🤖 AI"),
        "tool":   ("\033[1;33m", "🔧 Tool"),
    }

    # ── 1. Messages ──────────────────────────────────
    box("\033[1m Conversation\033[0m")
    subagents_called: list[str] = []

    for msg in result.get("messages", []):
        role = msg.type if hasattr(msg, "type") else "unknown"
        style, label = ROLE_STYLE.get(role, ("", role.upper()))

        tokens = ""
        if isinstance(msg, AIMessage):
            usage = msg.usage_metadata or {}
            if usage:
                tokens = (
                    f"  [{usage.get('input_tokens', 0)}"
                    f"→{usage.get('output_tokens', 0)} tok]"
                )
        print(f"\n{style}{label}{tokens}\033[0m")
        print(SEP)

        content = msg.content
        if isinstance(content, list):
            text = "\n".join(
                b["text"]
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = content or ""
        if text:
            print(text)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], ensure_ascii=False, indent=2)
                print(f"\n  \033[1;33m⤷ call  {tc['name']}\033[0m")
                for line in args_str.splitlines():
                    print(f"    {line}")
                # track subagent calls
                name = tc["name"]
                if name in ("researcher", "reviewer") and name not in subagents_called:
                    subagents_called.append(name)

        if isinstance(msg, ToolMessage):
            print(f"  name : {msg.name}")

    # ── 2. Dynamic subagent delegation summary ────────
    box("\033[1m Dynamic Subagent Delegation\033[0m")
    if subagents_called:
        print(f"\n  Subagents invoked: {', '.join(subagents_called)}")
        for name in subagents_called:
            print(f"  ✓ {name}")
    else:
        print("\n  No subagent delegation detected in this run.")

    print(f"\n{'═' * W}\n")


def main() -> None:
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

    prompt = (
        "Create a short implementation plan for a REST API that manages tasks (TODO app). "
        "Delegate background research to the researcher, then have the reviewer "
        "validate the plan for missing security or scalability considerations. "
        "Return a final consolidated plan."
    )

    result: dict = {}
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        stream_mode="values",
    ):
        result = chunk

    _print_result(result)


if __name__ == "__main__":
    main()
