import os
import warnings
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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


def _print_result(result: dict) -> None:
    width = 80
    sep = "-" * width

    print("\n" + "=" * width)
    print("Dynamic Subagents Run")
    print("=" * width)

    for msg in result.get("messages", []):
        role = getattr(msg, "type", "unknown")
        if role == "human":
            label = "👤 Human"
        elif role == "ai":
            label = "🤖 AI"
        elif role == "tool":
            label = "🔧 Tool"
        else:
            label = role

        print(f"\n{label}")
        print(sep)

        content = msg.content
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = content or ""

        if text:
            print(text)

        if isinstance(msg, AIMessage) and msg.tool_calls:
            print("\nTool Calls:")
            for tool_call in msg.tool_calls:
                print(f"- {tool_call.get('name')}: {tool_call.get('args')}")

        if isinstance(msg, ToolMessage):
            print(f"Tool name: {msg.name}")

    print("\n" + "=" * width)


def main() -> None:
    researcher_subagent = {
        "name": "researcher",
        "description": "Researches background facts and returns concise findings.",
        "system_prompt": "You gather facts quickly and provide short evidence-based notes.",
        "model": _make_model("RESEARCHER_MODEL", "claude-haiku-4-5"),
        "tools": [],
    }

    reviewer_subagent = {
        "name": "reviewer",
        "description": "Reviews outputs for correctness and missing assumptions.",
        "system_prompt": "You critique drafts and identify logical gaps clearly.",
        "model": _make_model("REVIEWER_MODEL", "claude-haiku-4-5"),
        "tools": [],
    }

    agent = create_deep_agent(
        model=_make_model("MAIN_MODEL", "claude-sonnet-4-5"),
        system_prompt=(
            "You are an orchestrator. Use subagents when decomposition improves quality. "
            "Delegate fact-finding to researcher and validation to reviewer."
        ),
        subagents=[researcher_subagent, reviewer_subagent],
    )

    prompt = (
        "Create a short implementation plan for a TODO app API and validate it for risks. "
        "Delegate to subagents where helpful, then return a consolidated answer."
    )

    result = {}
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        stream_mode="values",
    ):
        result = chunk

    _print_result(result)


if __name__ == "__main__":
    main()
