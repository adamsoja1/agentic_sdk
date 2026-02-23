# Agentic Framework

A modular framework for building AI agents with support for Discord integration, crew management, and memory systems.

## Features

- **Agent System**: Create and manage autonomous AI agents
- **Conversation Management**: Handle multi-turn conversations
- **Crew Management**: Orchestrate multiple agents working together
- **Memory System**: Persistent memory for agents
- **Streaming Events**: Real-time event streaming
- **LLM Client**: Unified interface for various LLM providers
- **Tool System**: Extensible tool framework

## Installation

### From GitHub (Recommended for now)

```bash
# Install latest version
pip install git+https://github.com/adamsoja1/agent-backend.git

# Install specific version/tag
pip install git+https://github.com/adamsoja1/agent-backend.git@v0.1.0

```

### From PyPI (when published)

```bash
pip install agentic-framework
```

### Development Installation

```bash
git clone git+https://github.com/adamsoja1/agent-backend.git
cd discord-ai-app
pip install -e ".[dev]"
```

## Quick Start

```python
from agentic_framework import Agent, Crew

# Create an agent
agent = Agent(name="assistant")

# Create a crew of agents
crew = Crew(agents=[agent])

# Run the crew
result = crew.run("Hello, how are you?")
```

---

## Detailed Usage Guide

### Architecture Overview

```
┌────────────────────────────────────────────────���────────────┐
│                         Crew                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Entrypoint Agent                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│  │
│  │  │  Agent 1     │  │  Agent 2     │  │  Agent 3     ││  │
│  │  │  (weather)   │  │  (helper)    │  │  (main)      ││  │
│  │  │  • Tools     │  │  • Tools     │  │  • Router    ││  │
│  │  │  • Delegate  │  │  • Delegate  │  │  • Route     ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘│  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────────────────���────────────────────┘
```

The framework consists of four main components:
- **Tools**: Python functions decorated with `@tool` that agents can call
- **Agent**: An AI agent with specific capabilities, tools, and configuration
- **Conversation**: Manages message history and context
- **Crew**: Orchestrates multiple agents with delegation and shared knowledge

### Step 1: Configure the LLM Client

The framework uses `AsyncOpenAI` client for LLM communication. Configure it with your base URL and API key.

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", ""),
)
```

**Environment Variables:**
- `LLM_BASE_URL`: URL of your LLM provider (e.g., Ollama, OpenAI, etc.)
- `OLLAMA_API_KEY`: API key for authentication

### Step 2: Create Tools

Tools are Python functions decorated with `@tool`. They allow agents to perform actions or retrieve information.

```python
from agentic_framework.tools.base import tool

@tool
def get_weather(city: str):
    """Gets weather for specific city name."""
    return f"Weather in {city} is turbo rainy!"

@tool
def get_user_items(user: str):
    """Lists items for a specific user name."""
    return f"Items for {user}: laptop, phone, keys"
```

**Tool Best Practices:**
- Use type hints for parameters
- Write clear docstrings (used as tool description)
- Keep tools focused on a single responsibility
- Return strings or serializable data

### Step 3: Create a Conversation

A `Conversation` manages the message history and system prompt for agents.

```python
from agentic_framework import Conversation

conversation = Conversation(
    id='1',
    system_prompt="You are friendly AITIS assistant."
)
```

### Step 4: Create Agents

Agents are the core building blocks. Each agent can have specific tools and capabilities.

```python
from agentic_framework import Agent

# Router agent (no tools, routes to other agents)
agent1 = Agent(
    description='You are helpful assistant.',
    name='AITIS',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[]
)

# Weather specialist agent
agent2 = Agent(
    description='Agent to get weather from the API - weather agent',
    name='weather_agent',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[get_weather]
)

# User items specialist agent
agent3 = Agent(
    description='Agent to get items from the API',
    name='user_helper_agent',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[get_user_items]
)
```

**Agent Parameters:**
| Parameter | Description |
|-----------|-------------|
| `name` | Unique agent identifier |
| `description` | Agent's purpose (visible to other agents for delegation) |
| `model` | LLM model name (e.g., 'kimi-k2.5', 'gpt-4') |
| `client` | AsyncOpenAI client instance |
| `conversation` | Shared conversation object |
| `tools` | List of available tools |
| `max_iterations` | Max tool calls per invocation (default: 7) |
| `can_delegate` | Allow agent to delegate to others (default: True) |
| `tool_auto_choice` | Auto-select tools without reasoning (default: False) |

### Step 5: Create a Crew

A `Crew` orchestrates multiple agents, enabling delegation and collaboration.

```python
from agentic_framework import Crew

crew = Crew(
    agents=[agent1, agent2, agent3],
    entrypoint_agent=agent1,
    conversation=conversation
)
```

**Crew Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `agents` | - | List of all agents (must include entrypoint) |
| `entrypoint_agent` | - | First agent to receive user messages |
| `delegate_to_agent` | `True` | Allow agents to delegate tasks |
| `only_ask_for_info` | `False` | Ask only vs full delegation |
| `shared_knowledge` | `True` | Share conversation history |
| `transfer_limit` | 5 | Max delegation depth |

**How Crew Delegation Works:**
1. User sends message to `entrypoint_agent`
2. Entrypoint agent analyzes the request
3. If another agent has the right tool, entrypoint **delegates** to that agent
4. Specialized agent processes the request and returns result
5. All results are streamed back to the user

```
User → AITIS (entrypoint) → weather_agent (delegation) → Tool Execution → Result
```

### Step 6: Run the Crew

Use `crew.invoke()` to stream responses asynchronously.

```python
from agentic_framework.core.stream_events import (
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    FinalAnswerEvent,
    ErrorEvent
)

# Example: Ask about weather
async for event in crew.invoke("What is the weather in London?"):
    if isinstance(event, TextDeltaEvent):
        print(event.delta, end="")  # Stream text chunks
    elif isinstance(event, ToolCallStartEvent):
        print(f"\n[Tool called: {event.tool_name}]")
    elif isinstance(event, ToolResultEvent):
        print(f"[Tool result: {event.result}]")
    elif isinstance(event, FinalAnswerEvent):
        print(f"\n[Final answer from {event.agent_name}]")
    elif isinstance(event, ErrorEvent):
        print(f"[Error: {event.error}]")
```

**Streaming Events:**
| Event | Description |
|-------|-------------|
| `TextDeltaEvent` | Streaming text chunk |
| `ToolCallStartEvent` | Agent started calling a tool |
| `ToolResultEvent` | Tool execution completed |
| `DelegationEvent` | Agent delegating to another agent |
| `AskAgentEventResult` | Agent asking another for information |
| `FinalAnswerEvent` | Agent produced final answer |
| `ErrorEvent` | Error occurred during processing |

---

## Complete Example

```python
import os
import asyncio
from openai import AsyncOpenAI
from agentic_framework import Agent, Conversation, Crew
from agentic_framework.tools.base import tool
from agentic_framework.core.stream_events import TextDeltaEvent, FinalAnswerEvent

# 1. Configure LLM client
client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", ""),
)

# 2. Define tools
@tool
def get_weather(city: str):
    """Gets weather for specific city name."""
    return f"Weather in {city} is turbo rainy!"

@tool
def get_user_items(user: str):
    """Lists items for a specific user name."""
    return f"Items for {user}: laptop, phone, keys"

# 3. Create shared conversation
conversation = Conversation(
    id='1',
    system_prompt="You are friendly AITIS assistant."
)

# 4. Create agents
agent1 = Agent(
    description='Main router assistant',
    name='AITIS',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[]
)

agent2 = Agent(
    description='Weather information specialist',
    name='weather_agent',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[get_weather]
)

agent3 = Agent(
    description='User items specialist',
    name='user_helper_agent',
    model='kimi-k2.5',
    client=client,
    conversation=conversation,
    tools=[get_user_items]
)

# 5. Create crew
crew = Crew(
    agents=[agent1, agent2, agent3],
    entrypoint_agent=agent1,
    conversation=conversation
)

# 6. Run
async def main():
    async for event in crew.invoke("What's the weather in London?"):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\n\n[Done! Agent: {event.agent_name}]")

asyncio.run(main())
```

---

## Advanced Features

### Memory System

Agents can persist information across conversations:

```python
from agentic_framework import Memory

memory = Memory()
memory.store("user_preference", "likes_python")
value = memory.retrieve("user_preference")
```

### Custom System Prompts

Tailor agent behavior with detailed prompts:

```python
agent = Agent(
    name="code_reviewer",
    description="Expert code reviewer",
    system_prompt="""You are an expert code reviewer. Focus on:
1. Code correctness
2. Best practices
3. Performance optimizations
4. Security issues
""",
    model="kimi-k2.5",
    client=client,
    tools=[]
)
```

### Single Agent Usage (without Crew)

Agents can work independently:

```python
agent = Agent(
    name="simple_agent",
    model="kimi-k2.5",
    client=client,
    conversation=conversation,
    tools=[get_weather]
)

# Stream response
async for event in agent.stream("What's the weather?"):
    print(event)

# Access conversation history
print(agent.conversation.get_messages())
```

---

## Best Practices

1. **Clear Agent Descriptions**: Help the entrypoint agent decide where to delegate
2. **Tool Granularity**: One tool per specific task
3. **Specialized Agents**: Each agent should have a focused responsibility
4. **Shared Conversation**: Enable `shared_knowledge` for context awareness
5. **Transfer Limits**: Set `transfer_limit` to prevent infinite loops
6. **Error Handling**: Always check for `ErrorEvent` in streaming
7. **Type Hints**: Use type hints in tools for better schema generation

---

## API Reference

### Agent
```python
Agent(
    name: str,                           # Required: unique name
    model: str,                         # Required: LLM model
    description: str = "",              # Purpose for delegation
    system_prompt: str = "",            # Instructions for behavior
    tools: list[BaseTool] = [],         # Available tools
    conversation: Conversation = None,  # Message history
    max_iterations: int = 7,             # Max tool calls
    can_delegate: bool = True,         # Allow delegation
    client: Any = None,                 # AsyncOpenAI client
)
```

### Crew
```python
Crew(
    agents: list[Agent],                 # Required: all agents
    entrypoint_agent: Agent,            # Required: first agent
    delegate_to_agent: bool = True,     # Enable delegation
    only_ask_for_info: bool = False,    # Ask-only mode
    shared_knowledge: bool = True,      # Share conversation
    conversation: Conversation = None, # Shared context
    transfer_limit: int = 5,           # Max delegation depth
)
```

---

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## License

MIT License
