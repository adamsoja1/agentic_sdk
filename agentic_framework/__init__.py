"""Agentic Framework - A modular framework for building AI agents.

This package provides tools and components for building AI-powered agents
with support for Discord integration, crew management, and memory systems.
"""

__version__ = "0.1.0"
__author__ = "Adam Soja"
__email__ = "your@email.com"

from agentic_framework.core.agent import Agent
from agentic_framework.core.conversation import Conversation
from agentic_framework.core.crew import Crew
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

__all__ = [
    "Agent",
    "Conversation",
    "Crew",
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
]

