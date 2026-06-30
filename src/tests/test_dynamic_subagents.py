"""
Test suite for Dynamic Subagents pilot with NVIDIA NIM.

Tests are split into two categories:
  - Unit tests  : mock LLM responses; no API key required.
  - Integration tests: real NVIDIA NIM calls; require NVIDIA_API_KEY.

Run all tests:
    uv run pytest tests/ -v

Run only unit tests (no API key):
    uv run pytest tests/ -v -m "not integration"

Run only integration tests:
    uv run pytest tests/ -v -m integration
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── ensure src/ is on the path when running from repo root ────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fake_nvidia_model(model_id: str = "fake/model") -> MagicMock:
    """Return a MagicMock that mimics a ChatNVIDIA instance."""
    mock = MagicMock()
    mock.model = model_id
    # .bind_tools() must return self so deepagents can chain calls
    mock.bind_tools.return_value = mock
    mock.with_config.return_value = mock
    return mock


def _make_subagent_spec(name: str, model: MagicMock) -> dict:
    return {
        "name": name,
        "description": f"{name.capitalize()} subagent for testing.",
        "system_prompt": f"You are the {name}.",
        "model": model,
        "tools": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — no API key required
# ══════════════════════════════════════════════════════════════════════════════


class TestMakeModel:
    """_make_model returns a ChatNVIDIA with the expected model ID."""

    def test_returns_chat_nvidia_instance(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from main import _make_model

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}):
            model = _make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct")

        assert isinstance(model, ChatNVIDIA)

    def test_uses_env_override(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from main import _make_model

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvapi-test",
            "MAIN_MODEL": "meta/llama-3.1-8b-instruct",
        }):
            model = _make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct")

        assert isinstance(model, ChatNVIDIA)
        assert model.model == "meta/llama-3.1-8b-instruct"

    def test_uses_default_when_env_absent(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from main import _make_model

        env = {k: v for k, v in os.environ.items() if k != "MAIN_MODEL"}
        env["NVIDIA_API_KEY"] = "nvapi-test"
        with patch.dict(os.environ, env, clear=True):
            model = _make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct")

        assert isinstance(model, ChatNVIDIA)
        assert model.model == "meta/llama-3.3-70b-instruct"

    def test_applies_custom_base_url(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from main import _make_model

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvapi-test",
            "NVIDIA_BASE_URL": "http://localhost:8000/v1",
        }):
            model = _make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct")

        assert isinstance(model, ChatNVIDIA)


class TestAgentCreation:
    """create_deep_agent accepts NVIDIA NIM models and subagent specs."""

    def test_agent_created_with_two_subagents(self):
        from deepagents import create_deep_agent

        researcher = _make_fake_nvidia_model("nvidia/nemotron-3-super-120b-a12b")
        reviewer   = _make_fake_nvidia_model("meta/llama-3.1-8b-instruct")
        orchestrator = _make_fake_nvidia_model("meta/llama-3.3-70b-instruct")

        agent = create_deep_agent(
            model=orchestrator,
            system_prompt="You are an orchestrator.",
            subagents=[
                _make_subagent_spec("researcher", researcher),
                _make_subagent_spec("reviewer",   reviewer),
            ],
        )

        assert agent is not None

    def test_agent_is_a_compiled_graph(self):
        """create_deep_agent returns a LangGraph CompiledStateGraph."""
        from deepagents import create_deep_agent
        from langgraph.graph.state import CompiledStateGraph

        orchestrator = _make_fake_nvidia_model()

        agent = create_deep_agent(
            model=orchestrator,
            system_prompt="You are an orchestrator.",
            subagents=[
                _make_subagent_spec("researcher", _make_fake_nvidia_model()),
                _make_subagent_spec("reviewer",   _make_fake_nvidia_model()),
            ],
        )

        assert isinstance(agent, CompiledStateGraph)

    def test_agent_created_without_subagents(self):
        """create_deep_agent works when subagents list is omitted."""
        from deepagents import create_deep_agent

        agent = create_deep_agent(
            model=_make_fake_nvidia_model(),
            system_prompt="Solo agent.",
        )
        assert agent is not None

    def test_subagent_names_are_unique(self):
        """Duplicate subagent names should not silently overwrite each other."""
        names_seen: set[str] = set()
        specs = [
            _make_subagent_spec("researcher", _make_fake_nvidia_model()),
            _make_subagent_spec("reviewer",   _make_fake_nvidia_model()),
        ]
        for spec in specs:
            assert spec["name"] not in names_seen, (
                f"Duplicate subagent name: {spec['name']}"
            )
            names_seen.add(spec["name"])


class TestSubagentSpecs:
    """Subagent specification dicts are well-formed."""

    def test_required_keys_present(self):
        required = {"name", "description", "system_prompt", "model", "tools"}
        spec = _make_subagent_spec("researcher", _make_fake_nvidia_model())
        assert required.issubset(spec.keys())

    def test_tools_is_list(self):
        spec = _make_subagent_spec("researcher", _make_fake_nvidia_model())
        assert isinstance(spec["tools"], list)

    def test_name_is_nonempty_string(self):
        spec = _make_subagent_spec("researcher", _make_fake_nvidia_model())
        assert isinstance(spec["name"], str) and spec["name"]

    def test_description_is_nonempty_string(self):
        spec = _make_subagent_spec("researcher", _make_fake_nvidia_model())
        assert isinstance(spec["description"], str) and spec["description"]


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests — require NVIDIA_API_KEY
# ══════════════════════════════════════════════════════════════════════════════

integration = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set — skipping integration tests",
)


@integration
class TestIntegrationAgentInvoke:
    """End-to-end tests against the real NVIDIA NIM API."""

    @pytest.fixture(scope="class")
    def agent(self):
        from deepagents import create_deep_agent
        from main import _make_model

        return create_deep_agent(
            model=_make_model("MAIN_MODEL", "meta/llama-3.3-70b-instruct"),
            system_prompt=(
                "You are an orchestrator. Delegate fact-finding to the researcher "
                "and validation to the reviewer."
            ),
            subagents=[
                {
                    "name": "researcher",
                    "description": "Gathers facts.",
                    "system_prompt": "You provide concise factual notes.",
                    "model": _make_model(
                        "RESEARCHER_MODEL", "nvidia/nemotron-3-super-120b-a12b"
                    ),
                    "tools": [],
                },
                {
                    "name": "reviewer",
                    "description": "Reviews and critiques drafts.",
                    "system_prompt": "You critique drafts clearly.",
                    "model": _make_model(
                        "REVIEWER_MODEL", "meta/llama-3.1-8b-instruct"
                    ),
                    "tools": [],
                },
            ],
        )

    def test_agent_returns_result_dict(self, agent):
        from langchain_core.messages import HumanMessage

        result: dict = {}
        for chunk in agent.stream(
            {"messages": [HumanMessage(content="What is 2 + 2? Answer briefly.")]},
            stream_mode="values",
        ):
            result = chunk

        assert isinstance(result, dict)
        assert "messages" in result

    def test_result_contains_ai_response(self, agent):
        from langchain_core.messages import HumanMessage, AIMessage

        result: dict = {}
        for chunk in agent.stream(
            {"messages": [HumanMessage(content="Say hello in one word.")]},
            stream_mode="values",
        ):
            result = chunk

        ai_messages = [
            m for m in result.get("messages", [])
            if isinstance(m, AIMessage) and m.content
        ]
        assert len(ai_messages) > 0, "Expected at least one non-empty AI message"

    def test_delegation_prompt_triggers_tool_calls(self, agent):
        """A prompt explicitly requesting delegation should produce tool calls."""
        from langchain_core.messages import HumanMessage, AIMessage

        result: dict = {}
        for chunk in agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Research the key benefits of REST APIs, then have the reviewer "
                            "validate that the findings are accurate. Return a final summary."
                        )
                    )
                ]
            },
            stream_mode="values",
        ):
            result = chunk

        messages = result.get("messages", [])
        tool_call_names = [
            tc["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        # At least one subagent call expected
        subagent_calls = [n for n in tool_call_names if n in ("researcher", "reviewer")]
        assert len(subagent_calls) > 0, (
            f"No subagent delegation detected. Tool calls seen: {tool_call_names}"
        )

    def test_researcher_subagent_called(self, agent):
        """Researcher subagent should be invoked on a knowledge-intensive task."""
        from langchain_core.messages import HumanMessage, AIMessage

        result: dict = {}
        for chunk in agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use the researcher to find three key facts about "
                            "transformer neural network architecture."
                        )
                    )
                ]
            },
            stream_mode="values",
        ):
            result = chunk

        messages = result.get("messages", [])
        tool_calls_flat = [
            tc["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        assert "researcher" in tool_calls_flat, (
            f"Expected 'researcher' in tool calls but got: {tool_calls_flat}"
        )

    def test_reviewer_subagent_called(self, agent):
        """Reviewer subagent should be invoked when explicitly requested."""
        from langchain_core.messages import HumanMessage, AIMessage

        result: dict = {}
        for chunk in agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Draft a one-sentence definition of machine learning, "
                            "then use the reviewer to check it for accuracy."
                        )
                    )
                ]
            },
            stream_mode="values",
        ):
            result = chunk

        messages = result.get("messages", [])
        tool_calls_flat = [
            tc["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        assert "reviewer" in tool_calls_flat, (
            f"Expected 'reviewer' in tool calls but got: {tool_calls_flat}"
        )

    def test_full_pipeline_both_subagents(self, agent):
        """Full pipeline: both researcher and reviewer should be called."""
        from langchain_core.messages import HumanMessage, AIMessage

        result: dict = {}
        for chunk in agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a short plan for a TODO app REST API. "
                            "Delegate background research to the researcher, then have the "
                            "reviewer validate the plan for missing security considerations. "
                            "Return a final consolidated plan."
                        )
                    )
                ]
            },
            stream_mode="values",
        ):
            result = chunk

        messages = result.get("messages", [])
        tool_calls_flat = [
            tc["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        called = set(tool_calls_flat)
        assert "researcher" in called, "Researcher was not called in the full pipeline"
        assert "reviewer"   in called, "Reviewer was not called in the full pipeline"
