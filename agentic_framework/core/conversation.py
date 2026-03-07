from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    summarized_history: str = ""  # to be added later

    def add_message(self, role: str, content: str) -> None:
        """Add a simple user or assistant message."""
        self.messages.append({"role": role, "content": content})

    def add_assistant_with_tool_calls(
        self,
        content: str,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """
        Add an assistant message that includes tool call requests.
        `tool_calls` should be a list of OpenAI-formatted tool call dicts:
            {"id": "call_abc", "type": "function",
             "function": {"name": "...", "arguments": "..."}}
        """
        self.messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            }
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> None:
        """Add a tool result message that corresponds to a prior tool call."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def get_messages(self) -> list[dict[str, Any]]:
        """Return all messages, optionally prepending the system prompt."""
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + self.messages
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()