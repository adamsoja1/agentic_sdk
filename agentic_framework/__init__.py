"""Agentic Framework - tools for building streaming LLM agents and crews."""

from agentic_framework._version import __version__
__author__ = "Adam Soja"
__email__ = "your@email.com"

# Applications configure logging explicitly; importing the SDK has no side effects.

from agentic_framework.core.agent import Agent
from agentic_framework.core.conversation import Conversation, ImageInput, ImageInputs
from agentic_framework.core.crew import Crew
from agentic_framework.core.prompting import AgentPrompt, PromptBlock, PromptSection, PromptTemplate
from agentic_framework.core.stream_events import (
    AskAgentEventResult,
    DelegationEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from agentic_framework.tools.base import BaseTool
from agentic_framework.tools.base import tool
from agentic_framework.logging_config import setup_logging, get_logger

__all__ = [
    "Agent",
    "Conversation",
    "ImageInput",
    "ImageInputs",
    "Crew",
    "AgentPrompt",
    "PromptBlock",
    "PromptSection",
    "PromptTemplate",
    "StreamEvents",
    "BaseTool",
    "tool",
    "AskAgentEventResult",
    "DelegationEvent",
    "ErrorEvent",
    "FinalAnswerEvent",
    "StreamEvent",
    "TextDeltaEvent",
    "ToolCallStartEvent",
    "ToolResultEvent",
    "setup_logging",
    "get_logger",
]
