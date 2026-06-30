"""Tests for agents.py: AgentState routing and AgentLLM initialization."""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 9. test_agent_state_routing
# ---------------------------------------------------------------------------

def test_agent_state_routing():
    from agents import AgentState, should_continue_tools
    from langchain_core.messages import AIMessage

    # 1. Last message has no tool_calls → "answer_summary".
    state = AgentState(
        messages=[AIMessage(content="hello")],
        memory=[],
        context=None,
    )
    assert should_continue_tools(state) == "answer_summary"

    # 2. Last AIMessage has non-empty tool_calls → "tools".
    ai_with_tools = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_file_list", "args": {}, "id": "call_1", "type": "tool_call"}
        ],
    )
    state2 = AgentState(messages=[ai_with_tools], memory=[], context=None)
    assert should_continue_tools(state2) == "tools"

    # 3. run_agent_on_text before initialize_agent → "Agent not initialized" string.
    from agents import AgentLLM

    agent = AgentLLM()
    result = agent.run_agent_on_text("some text", memory=[])
    assert "Agent not initialized" in result

    # 4. run_agent_on_text with empty string → "No text provided" string.
    agent._agent_executor = MagicMock()
    result = agent.run_agent_on_text("", memory=[])
    assert "No text provided" in result


# ---------------------------------------------------------------------------
# 10. test_agent_llm_initialization_pipeline
# ---------------------------------------------------------------------------

def test_agent_llm_initialization_pipeline():
    from agents import AgentLLM

    agent = AgentLLM()

    # 1. Empty api_key with OpenAI model → False.
    assert agent.initialize_agent(api_key="", model_name="gpt-4.1-mini") is False

    # 2. Unsupported model name → False.
    assert agent.initialize_agent(api_key="key", model_name="unsupported-model") is False

    # 3. Ollama model with mocked ChatOllama → True.
    with patch("agents.ChatOllama") as MockOllama, \
         patch("agents.create_main_agent") as mock_create:
        mock_llm = MagicMock()
        MockOllama.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        mock_create.return_value = MagicMock()

        assert agent.initialize_agent(api_key="key", model_name="lfm2.5:350m") is True

    # 4. get_model_name → "lfm2.5:350m"; get_agent_executor is not None.
    assert agent.get_model_name == "lfm2.5:350m"
    assert agent.get_agent_executor is not None

    # 5. change_summary_language("eng") → no exception; agent executor rebuilt.
    with patch("agents.create_main_agent") as mock_create:
        mock_create.return_value = MagicMock()
        agent.change_summary_language("eng")
        mock_create.assert_called_once()

    # 6. change_summary_language("xyz") → executor unchanged.
    old_executor = agent.get_agent_executor
    agent.change_summary_language("xyz")
    assert agent.get_agent_executor is old_executor

    # 7. set_context with non-existent path → _context tuple set, no crash.
    agent.set_context("some/path", "file info")
    assert agent._context is not None
