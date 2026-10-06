import asyncio
from types import SimpleNamespace

import pytest

from agentic_framework.core.agent import Agent
from agentic_framework.core.prompting import AgentPrompt, PromptSection, PromptTemplate
from agentic_framework.core.stream_events import (
    FinalAnswerEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from agentic_framework.tools.base import BaseTool, Skill


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class FakeClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []
        async def create(**kwargs):
            self.calls.append(kwargs)
            return FakeStream(self._responses.pop(0))
        self.responses = SimpleNamespace(create=create)


def chunk(content="", tool_calls=None, finish_reason=None, reasoning_content=None):
    if tool_calls:
        tc = tool_calls[0]
        item = SimpleNamespace(
            id=f"item_{tc.index}", type="function_call", call_id=tc.id,
            name=tc.function.name, arguments=tc.function.arguments,
        )
        return SimpleNamespace(type="response.output_item.done", item=item)
    return SimpleNamespace(type="response.output_text.delta", delta=content)


def tool_call_delta(index, call_id=None, name=None, arguments=""):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def collect_events(agent, message):
    async def _collect():
        return [event async for event in agent.stream(message)]

    return asyncio.run(_collect())


def test_agent_streams_final_answer_and_records_visible_messages():
    client = FakeClient(
        responses=[
            [
                chunk(content="Hello, ", reasoning_content="hidden "),
                chunk(content="world!", finish_reason="stop"),
            ]
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client)

    events = collect_events(agent, "Hi")

    assert [event.delta for event in events if isinstance(event, TextDeltaEvent)] == [
        "Hello, ",
        "world!",
    ]
    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Hello, world!")
    assert agent.conversation.get_messages() == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello, world!"},
    ]

    create_call = client.calls[0]
    assert create_call["model"] == "test-model"
    assert create_call["stream"] is True
    assert "tools" not in create_call
    assert "tool_choice" not in create_call


def test_agent_executes_tool_call_then_continues_to_final_answer():
    echo_tool = BaseTool(
        name="echo",
        description="Echo text.",
        func=lambda text: text.upper(),
    )
    client = FakeClient(
        responses=[
            [
                chunk(
                    tool_calls=[
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="echo",
                            arguments='{"text": "hello"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [chunk(content="Done.", finish_reason="stop")],
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client, tools=[echo_tool])

    events = collect_events(agent, "Echo this")

    assert any(
        event == ToolCallStartEvent(
            agent_name="assistant",
            call_id="call_1",
            tool_name="echo",
            arguments_raw='{"text": "hello"}',
        )
        for event in events
    )
    assert any(
        event == ToolResultEvent(
            agent_name="assistant",
            call_id="call_1",
            tool_name="echo",
            result="HELLO",
            is_error=False,
        )
        for event in events
    )
    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Done.")
    assert agent.conversation.get_messages() == [
        {"role": "user", "content": "Echo this"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "HELLO"},
        {"role": "assistant", "content": "Done."},
    ]


def test_agent_records_tool_parse_errors_as_tool_results():
    tool_instance = BaseTool(name="echo", description="Echo text.", func=lambda text: text)
    client = FakeClient(
        responses=[
            [
                chunk(
                    tool_calls=[
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="echo",
                            arguments='{"text": ',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [chunk(content="Recovered.", finish_reason="stop")],
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client, tools=[tool_instance])

    events = collect_events(agent, "Call badly")

    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Recovered.")
    tool_message = agent.conversation.get_messages()[2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "Failed to parse arguments for tool 'echo'" in tool_message["content"]


def test_agent_parse_tool_arguments_truncates_extra_json_data():
    parsed, clean = Agent._parse_tool_arguments('{"text": "hi"} trailing')

    assert parsed == {"text": "hi"}
    assert clean == '{"text": "hi"}'


def test_agent_execute_tool_supports_sync_and_async_callables():
    async def async_echo(text):
        return f"async {text}"

    agent = Agent(
        name="assistant",
        model="test-model",
        tools=[
            BaseTool(name="sync_echo", description="Sync.", func=lambda text: f"sync {text}"),
            BaseTool(name="async_echo", description="Async.", func=async_echo),
        ],
    )

    assert asyncio.run(agent._execute_tool("sync_echo", {"text": "ok"})) == "sync ok"
    assert asyncio.run(agent._execute_tool("async_echo", {"text": "ok"})) == "async ok"

    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        asyncio.run(agent._execute_tool("missing", {}))


def test_agent_skill_activation_switches_tool_overlay_and_can_deactivate():
    base_tool = BaseTool(name="base", description="Base.", func=lambda: "base")
    research_tool = BaseTool(name="search", description="Search.", func=lambda query: query)
    write_tool = BaseTool(name="draft", description="Draft.", func=lambda text: text)
    research_skill = Skill(name="research", description="Research.", tools=[research_tool])
    writing_skill = Skill(name="writing", description="Writing.", tools=[write_tool])
    agent = Agent(
        name="assistant",
        model="test-model",
        tools=[base_tool],
        skills=[research_skill, writing_skill],
    )

    assert agent.list_tools() == ["base", "skill_research", "skill_writing"]

    assert agent._activate_skill(research_skill) == "Skill 'skill_research' activated."
    assert set(agent.list_tools()) == {"base", "skill_research", "skill_writing", "search"}

    assert (
        agent._activate_skill(writing_skill)
        == "Skill 'skill_research' deactivated. Skill 'skill_writing' activated."
    )
    assert set(agent.list_tools()) == {"base", "skill_research", "skill_writing", "draft"}

    assert agent._deactivate_skill() == "Skill 'skill_writing' deactivated."
    assert agent.list_tools() == ["base", "skill_research", "skill_writing"]


def test_agent_renders_structured_prompt_object():
    agent = Agent(
        name="assistant",
        model="test-model",
        prompt=AgentPrompt(
            role="You are a billing support specialist.",
            objectives=[
                "Resolve subscription questions accurately.",
                "Offer the clearest next step.",
            ],
            constraints=["Do not guess account-specific details."],
            output_requirements=["Use short sections."],
        ),
    )

    prompt = agent.render_system_prompt()

    assert prompt.startswith("Role:\nYou are a billing support specialist.")
    assert "Role:\nYou are a billing support specialist." in prompt
    assert "Objectives:\n- Resolve subscription questions accurately." in prompt
    assert "Constraints:\n- Do not guess account-specific details." in prompt
    assert "Output Requirements:\n- Use short sections." in prompt


def test_agent_renders_segmented_system_prompt_with_skill_context():
    research_tool = BaseTool(name="search", description="Search.", func=lambda query: query)
    research_skill = Skill(
        name="research",
        description="Use for document research.",
        tools=[research_tool],
    )
    agent = Agent(
        name="assistant",
        model="test-model",
        system_prompt="You are a concise assistant.",
        skills=[research_skill],
    )

    agent._rebuild_system_prompt()
    prompt = agent.conversation.system_prompt

    assert prompt.startswith("You are a concise assistant.")
    assert "You are a concise assistant." in prompt
    assert "Available skills:" in prompt
    assert "skill_research" in prompt
    assert "Activate this skill to unlock: search." in prompt
    assert "Active skill: skill_research" not in prompt

    agent._activate_skill(research_skill)
    agent._rebuild_system_prompt()
    prompt = agent.conversation.system_prompt

    assert "Active skill: skill_research" in prompt
    assert "Current skill focus: Use for document research." in prompt
    assert "Unlocked tools from this skill: search" in prompt


def test_agent_prompt_sections_can_be_replaced_and_inserted():
    research_tool = BaseTool(name="search", description="Search.", func=lambda query: query)
    research_skill = Skill(name="research", description="Research.", tools=[research_tool])
    template = PromptTemplate(
        sections=[
            PromptSection("tone", "Stay calm and structured."),
            PromptSection("skills", "Use the custom skill router when appropriate."),
        ]
    )
    agent = Agent(
        name="assistant",
        model="test-model",
        system_prompt="Base instructions.",
        skills=[research_skill],
        prompt_template=template,
    )

    agent.set_prompt_section("format", "Reply with short paragraphs.", after="tone")
    agent._rebuild_system_prompt()
    prompt = agent.conversation.system_prompt

    assert prompt.startswith("Base instructions.")
    assert "Base instructions." in prompt
    assert "Stay calm and structured." in prompt
    assert "Reply with short paragraphs." in prompt
    assert "Use the custom skill router when appropriate." in prompt
    assert "Available skills:" not in prompt


def test_agent_prompt_sections_can_be_removed():
    agent = Agent(
        name="assistant",
        model="test-model",
        system_prompt="Base instructions.",
    )

    assert agent.remove_prompt_section("system_prompt") is True
    assert agent.render_system_prompt() == ""


def test_agent_max_iteration_warning_uses_dedicated_prompt_section():
    agent = Agent(
        name="assistant",
        model="test-model",
        system_prompt="Base instructions.",
    )

    assert "maximum number of iterations" not in agent.render_system_prompt()

    agent.modify_prompt_on_max_iterations()

    assert agent.system_prompt == "Base instructions."
    assert "Base instructions." in agent.render_system_prompt()
    assert "maximum number of iterations" in agent.render_system_prompt()
