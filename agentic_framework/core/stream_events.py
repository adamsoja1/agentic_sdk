from dataclasses import dataclass
from typing import Any

@dataclass
class StreamEvent:
    """Base class for all streaming events emitted by an agent."""
    agent_name: str


@dataclass
class TextDeltaEvent(StreamEvent):
    """A chunk of assistant text arrived."""
    delta: str


@dataclass
class ToolCallStartEvent(StreamEvent):
    """The model decided to call a tool."""
    call_id: str
    tool_name: str
    arguments_raw: str  # accumulates during streaming


@dataclass
class ToolResultEvent(StreamEvent):
    """Tool execution finished."""
    call_id: str
    tool_name: str
    result: Any
    is_error: bool = False


@dataclass
class DelegationEvent(StreamEvent):
    """This agent is delegating to another agent."""
    target_agent: str
    task: str

@dataclass
class AskAgentEventResult(StreamEvent):
    """This agent is asking another agent for information."""
    target_agent: str
    question: str
    result: str


@dataclass
class FinalAnswerEvent(StreamEvent):
    """The agent finished and produced a final answer."""
    answer: str


@dataclass
class ErrorEvent(StreamEvent):
    """An unrecoverable error occurred."""
    error: str
