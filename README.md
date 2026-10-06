# Agentic Framework

A small Python framework for building streaming LLM agents with tools, skills, and multi-agent crew delegation.

## What Is Included

- **Agents**: async streaming agents backed by the OpenAI Responses API.
- **Tools**: Python callables exposed to the model as OpenAI tool schemas.
- **Skills**: grouped toolsets that an agent can activate during a run.
- **Conversations**: event-based message history converted to Responses API input items.
- **Crews**: orchestration for multiple agents with delegation or ask-only collaboration.
- **Streaming events**: typed events for text deltas, tool calls, tool results, delegation, errors, and final answers.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/adamsoja1/agent-backend.git
```

Install a tagged version:

```bash
pip install git+https://github.com/adamsoja1/agent-backend.git@v0.1.0
```

Install for local development:

```bash
git clone https://github.com/adamsoja1/agent-backend.git
cd agent-backend
python -m pip install -e ".[dev]"
```

When the package is published to PyPI:

```bash
pip install agentic-framework
```

## Configuration

`Agent` uses an `AsyncOpenAI`-compatible client. You can pass a client explicitly, which is recommended:

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("LLM_PROVIDER", "https://ollama.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
)
```

If you do not pass `client=...`, the framework creates a default `AsyncOpenAI` client using:

- `LLM_PROVIDER`, defaulting to `https://ollama.com/v1`
- `OPENAI_API_KEY`, defaulting to `EMPTY`

## Quick Start

```python
import asyncio
from agentic_framework import Agent, AgentPrompt
from agentic_framework.core.stream_events import TextDeltaEvent, FinalAnswerEvent
from agentic_framework.tools.base import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: rainy"


agent = Agent(
    name="assistant",
    model="gpt-4o-mini",
    prompt=AgentPrompt(
        role="You are a concise assistant.",
        objectives=[
            "Answer the user's question accurately.",
            "Keep the answer short and helpful.",
        ],
        output_requirements=[
            "Use short paragraphs.",
        ],
    ),
    tools=[get_weather],
)


async def main() -> None:
    async for event in agent.stream("What is the weather in London?"):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\nFinal answer: {event.answer}")


asyncio.run(main())
```

For a non-streaming convenience wrapper:

```python
answer = await agent.invoke("What is the weather in London?")
```

## Structured Agent Prompts

Use `AgentPrompt` when you want the base prompt to stay readable and structured instead of writing one long system string.

```python
from agentic_framework import Agent, AgentPrompt, PromptBlock


support_prompt = AgentPrompt(
    role="You are a senior product support agent.",
    objectives=[
        "Diagnose the user's issue accurately.",
        "Guide the user toward the most practical next step.",
    ],
    context="You help customers with product setup, billing, and troubleshooting.",
    instructions=[
        "Use tools when they improve the answer.",
        "Explain tradeoffs when presenting multiple options.",
    ],
    constraints=[
        "Do not invent account-specific data.",
        "Be explicit when you are uncertain.",
    ],
    style=[
        "Prefer short sections.",
        "Use calm, direct language.",
    ],
    output_requirements=[
        "End with a recommendation or next action.",
    ],
    additional_sections=[
        PromptBlock(
            title="Escalation Rules",
            items=[
                "Recommend human support when the issue requires account-only access.",
            ],
        )
    ],
)

agent = Agent(
    name="support",
    model="gpt-4o-mini",
    prompt=support_prompt,
)
```

`system_prompt="..."` still works for backward compatibility, but `prompt=AgentPrompt(...)` is the recommended format for base instructions.

You can also define the exact segments yourself and choose how points render:

```python
custom_prompt = AgentPrompt(
    sections=[
        PromptBlock(title="Persona", content="You are a release manager."),
        PromptBlock(
            title="Workflow",
            items=[
                "Check the changelog.",
                "Summarize risk before recommendation.",
            ],
            point_prefix="{index}. ",
        ),
        PromptBlock(
            title="Guardrails",
            items=["Do not invent release dates."],
            point_prefix="* ",
        ),
    ]
)
```

For the built-in segments, set `point_prefix="{index}. "` for numbered points, `point_prefix="* "` for star bullets, or `point_prefix=None` for plain lines. `heading_suffix=""` removes the default colon after section titles.

Prompt sections can also live in markdown files:

```markdown
# Persona
You are a release manager.

## Workflow
1. Check the changelog.
2. Summarize risk before recommendation.

## Guardrails
- Do not invent release dates.
- Name uncertainty clearly.
```

```python
agent = Agent(
    name="release_manager",
    model="gpt-4o-mini",
    prompt=AgentPrompt.from_markdown("prompts/release_manager.md"),
)

template = PromptTemplate.from_markdown("prompts/support_sections.md")
```

Markdown headings become prompt sections. Bullet and numbered lists are loaded as section points, and paragraph text becomes section content.

## Tools

Decorate a Python function with `@tool` to expose it to an agent. The function name becomes the tool name, the docstring becomes the description, and type hints are used to build a JSON schema.

```python
from agentic_framework.tools.base import tool


@tool
def get_user_items(user: str) -> str:
    """List items owned by a user."""
    return f"Items for {user}: laptop, phone, keys"
```

You can also override the generated name and description:

```python
@tool(name="lookup_items", description="Look up a user's items.")
def get_user_items(user: str) -> str:
    return f"Items for {user}: laptop, phone, keys"
```

Supported primitive schema mappings are `str`, `int`, `float`, `bool`, `list`, and `dict`. Unknown or complex annotations fall back to `"string"`.

## Skills

Skills group tools behind an activation tool. A skill named `"research"` is exposed as `skill_research`; when activated, its tools are added to the agent's available tools until another skill is activated or the skill is deactivated internally. The default skill prompt section also explains how to activate skills, what tools they unlock, and that switching skills deactivates the previously active one.

```python
from agentic_framework.tools.base import BaseTool, Skill


search_tool = BaseTool(
    name="search_docs",
    description="Search internal docs.",
    func=lambda query: f"Results for {query}",
)

research_skill = Skill(
    name="research",
    description="Use when the task needs document research.",
    tools=[search_tool],
)

agent = Agent(
    name="assistant",
    model="gpt-4o-mini",
    skills=[research_skill],
)
```

### Segmented System Prompts

`Agent` can compose its system prompt from named sections so you can swap or extend behavior without rewriting one large string. The default template renders your structured `prompt`, legacy `system_prompt`, skill guidance, active skill reminders, crew coordination, and iteration-limit warning as separate sections. All of those sections are ordinary template sections, so you can replace or remove them when needed.

```python
from agentic_framework import Agent, PromptSection, PromptTemplate
from agentic_framework.tools.base import BaseTool, Skill


search_tool = BaseTool(
    name="search_docs",
    description="Search internal docs.",
    func=lambda query: f"Results for {query}",
)

research_skill = Skill(
    name="research",
    description="Use when the task needs document research.",
    tools=[search_tool],
)

prompt_template = PromptTemplate(
    sections=[
        PromptSection("tone", "Be decisive, concise, and cite the tools you use."),
        PromptSection(
            "skills",
            lambda agent: (
                "Skill routing:\n"
                "- Prefer `skill_research` for docs or knowledge-base work.\n"
                "- Activate a skill before calling the tools it unlocks."
            ) if agent.skills else "",
        ),
    ]
)

agent = Agent(
    name="assistant",
    model="gpt-4o-mini",
    system_prompt="You are a product support agent.",
    skills=[research_skill],
    prompt_template=prompt_template,
)

agent.set_prompt_section(
    "formatting",
    "Answer in short sections with a final recommendation.",
    after="tone",
)
```

If you do not provide a custom `skills` section, the framework injects a default one that lists each skill and the tools it unlocks.

## Conversations

`Conversation` stores events, not just plain messages. `get_response_input()` converts visible events into Responses API input items. `get_messages()` remains available for inspecting the event history in chat message form.

```python
from agentic_framework import Conversation

conversation = Conversation(system_prompt="You are helpful.")
conversation.add_user_message("Hello")
conversation.add_assistant_message("Hi!")

print(conversation.get_messages())
```

Internal events such as reasoning and text deltas are intentionally skipped when building messages for the LLM.

## Crews

Use `Crew` to connect multiple agents. The `entrypoint_agent` receives the user's input first. During initialization, agents are registered as tools on their peers:

- `delegate_to_agent_<name>` in normal delegation mode
- `ask_agent_<name>` when `only_ask_for_info=True`

```python
import asyncio
from agentic_framework import Agent, Crew
from agentic_framework.core.stream_events import ErrorEvent, FinalAnswerEvent, TextDeltaEvent
from agentic_framework.tools.base import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: rainy"


router = Agent(
    name="router",
    model="gpt-4o-mini",
    description="Routes work to specialists.",
)

weather_agent = Agent(
    name="weather_agent",
    model="gpt-4o-mini",
    description="Answers weather questions.",
    tools=[get_weather],
)

crew = Crew(
    agents=[router, weather_agent],
    entrypoint_agent=router,
    transfer_limit=5,
)


async def main() -> None:
    async for event in crew.invoke("What is the weather in Warsaw?"):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\nFinal answer from {event.agent_name}: {event.answer}")
        elif isinstance(event, ErrorEvent):
            print(f"\nError: {event.error}")


asyncio.run(main())
```

For a non-streaming crew response:

```python
answer = await crew.get_response("What is the weather in Warsaw?")
```

## Streaming Events

Agents and crews yield dataclass events from `agentic_framework.core.stream_events`:

| Event | Description |
| --- | --- |
| `TextDeltaEvent` | A chunk of assistant text. |
| `ToolCallStartEvent` | A tool call is about to execute. |
| `ToolResultEvent` | A tool call completed or failed. |
| `DelegationEvent` | An agent delegated work to another agent. |
| `AskAgentEventResult` | An agent asked another agent for information. |
| `FinalAnswerEvent` | The agent produced its final answer. |
| `ErrorEvent` | A non-recoverable error occurred. |

## API Reference

### Agent

```python
Agent(
    name: str,
    model: str,
    description: str = "",
    system_prompt: str = "",
    prompt: AgentPrompt | None = None,
    can_delegate: bool = True,
    tools: list[BaseTool] = [],
    skills: list[Skill] = [],
    conversation: Conversation = Conversation(),
    prompt_template: PromptTemplate | None = None,
    max_iterations: int = 7,
    client: Any = default AsyncOpenAI client,
    tool_auto_choice: bool = False,
    web_search: bool = False,
    crew: Crew | None = None,
    output_format: Any = None,
    reasoning_effort: str = "medium",
)
```

Useful methods:

- `stream(user_message: str)`: async generator of streaming events.
- `invoke(user_message: str) -> str`: returns the final answer string.
- `add_tool(tool: BaseTool | Agent)`: adds a tool, or registers an agent as a delegation/ask tool when part of a crew.
- `remove_tool(name: str) -> bool`: removes a tool by name.
- `list_tools() -> list[str]`: lists available tool names.
- `as_dict() -> dict`: serializes basic agent state.

Set `web_search=True` to enable the Responses API's built-in web search tool:

```python
agent = Agent(name="researcher", model="gpt-4.1", web_search=True)
answer = await agent.invoke("What changed in the latest release?")
```

### Crew

```python
Crew(
    agents: list[Agent],
    entrypoint_agent: Agent,
    delegate_to_agent: bool = True,
    only_ask_for_info: bool = False,
    shared_knowledge: bool = True,
    shared_identity: bool = False,
    conversation: Conversation = Conversation(),
    system_prompt: str = "...",
    transfer_limit: int = 5,
)
```

Useful methods:

- `invoke(input_message: str)`: async generator of crew events.
- `get_response(input_message: str) -> str`: returns the final answer string.
- `get_agent_by_name(name: str) -> Agent | None`: finds an agent in the crew.

### BaseTool

```python
BaseTool(
    name: str,
    description: str,
    func: Callable[..., Any] | None = None,
)
```

Useful methods:

- `execute(*args, **kwargs) -> Any`: calls the wrapped function.
- `to_openai_schema() -> dict`: returns an OpenAI-compatible function schema.

### Skill

```python
Skill(
    name: str,
    description: str,
    func: Callable[..., Any] | None = None,
    tools: list[BaseTool] = [],
)
```

`Skill` requires at least one tool and prefixes its name with `skill_` during initialization.

## Packaging And Releases

The package version lives in:

```python
agentic_framework/_version.py
```

To release a new version:

1. Update `__version__`.
2. Create a tag that matches the version, for example `v0.1.1`.
3. Publish a GitHub release for that tag.

The CI workflow runs tests and builds wheel/source distributions. The publish workflow checks that the release tag matches the package version before uploading to PyPI.

## Development

Run tests:

```bash
python -m pytest -q
```

Build the package:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Requirements

- Python 3.10+
- Runtime dependencies listed in `pyproject.toml`

## Current Limitations

- There is no implemented memory API yet; `agentic_framework.core.memory` is currently empty.
- `agentic_framework.core.swarm` is currently empty.
- `tool_auto_choice`, `output_format`, and `shared_identity` are stored on the dataclasses but do not currently change runtime behavior.

## License

MIT
