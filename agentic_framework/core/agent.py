from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI

from .conversation import Conversation
from ..tools.base import BaseTool
from ..core.stream_events import (
    DelegationEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from crew import Crew

logger = logging.getLogger(__name__)

_default_client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)


@dataclass
class Agent:
    name: str
    model: str
    tools: list[BaseTool] = field(default_factory=list)
    conversation: Conversation = field(default_factory=Conversation)
    max_iterations: int = 7
    client: Any = field(default_factory=lambda: _default_client)
    tool_auto_choice: bool = False
    # Set by Crew at registration time – agents should not set this directly
    crew: Crew | None = field(default=None, repr=False, compare=False)


    def __post_init__(self):
        if isinstance(self.tools, list):
            self.tools = {tool.name: tool for tool in self.tools}

    def remove_tool(self, name: str) -> bool:
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "tools": list(self.tools.keys()),
            "conversation": self.conversation.get_messages(),
            "max_iterations": self.max_iterations,
            "tool_auto_choice": self.tool_auto_choice,
        }


    def _build_openai_tools(self) -> list[dict[str, Any]]:
        schemas = [t.to_openai_schema() for t in self.tools.values()]
        if self.crew:
            schemas.append(self._delegation_tool_schema())
        return schemas

    @staticmethod
    def _delegation_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "delegate_to_agent",
                "description": (
                    "Delegate a sub-task to another agent in the crew. "
                    "Use this when a task is better handled by a specialist agent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the target agent.",
                        },
                        "task": {
                            "type": "string",
                            "description": "The exact task description to send to the target agent.",
                        },
                    },
                    "required": ["agent_name", "task"],
                },
            },
        }

    def _system_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.conversation.system_prompt:
            msgs.append({"role": "system", "content": self.conversation.system_prompt})
        if self.crew:
            agent_list = ", ".join(a for a in self.crew.agents if a != self.name)
            crew_hint = (
                f"\nYou are part of a crew. "
                f"Other available agents: [{agent_list}]. "
                "Use `delegate_to_agent` when a task falls outside your expertise."
            )
            if msgs:
                msgs[0]["content"] += crew_hint
            else:
                msgs.append({"role": "system", "content": crew_hint.strip()})
        return msgs

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found on agent '{self.name}'.")
        result = tool.execute(**arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _delegate(self, target_name: str, task: str) -> AsyncGenerator[StreamEvent, None]:
        if self.crew is None:
            yield ErrorEvent(agent_name=self.name, error="Agent is not part of a crew; cannot delegate.")
            return
        target = self.crew.agents.get(target_name)
        if target is None:
            yield ErrorEvent(agent_name=self.name, error=f"Unknown agent '{target_name}' in crew.")
            return
        yield DelegationEvent(agent_name=self.name, target_agent=target_name, task=task)


    async def stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        if self.client is None:
            yield ErrorEvent(agent_name=self.name, error="No OpenAI client configured.")
            return
        
        self.conversation.add_message("user", user_message)

        openai_tools = self._build_openai_tools()
        tool_choice: Any = "auto" if openai_tools else "none"

        final_answer = ""

        for _ in range(self.max_iterations):
            messages = self._system_messages() + self.conversation.get_messages()

            stream_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                stream=True,
            )
            if openai_tools:
                stream_kwargs["tools"] = openai_tools
                stream_kwargs["tool_choice"] = tool_choice

            try:
                response_stream = await self.client.chat.completions.create(**stream_kwargs)
            except Exception as exc:
                yield ErrorEvent(agent_name=self.name, error=str(exc))
                return

            assistant_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}

            async for chunk in response_stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                if delta.content:
                    assistant_text += delta.content
                    yield TextDeltaEvent(agent_name=self.name, delta=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        acc = tool_calls_acc[idx]
                        if tc.id:
                            acc["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

                if choice.finish_reason == "stop":
                    break

            if not tool_calls_acc:
                self.conversation.add_message("assistant", assistant_text)
                final_answer = assistant_text
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": [
                    {
                        "id": acc["id"],
                        "type": "function",
                        "function": {"name": acc["name"], "arguments": acc["arguments"]},
                    }
                    for acc in tool_calls_acc.values()
                ],
            }
            self.conversation.messages.append(assistant_msg)

            for acc in tool_calls_acc.values():
                call_id = acc["id"]
                tool_name = acc["name"]
                try:
                    arguments = json.loads(acc["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {}
                    logger.warning("Failed to parse tool arguments: %s", exc)

                yield ToolCallStartEvent(
                    agent_name=self.name,
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments_raw=acc["arguments"],
                )

                if tool_name == "delegate_to_agent":
                    target_name = arguments.get("agent_name", "")
                    task = arguments.get("task", "")
                    delegation_result = ""
                    async for event in self._delegate(target_name, task):
                        yield event
                        if isinstance(event, FinalAnswerEvent):
                            delegation_result = event.answer
                    tool_result = delegation_result or f"Delegation to {target_name} completed."
                    is_error = False
                else:
                    try:
                        tool_result = await self._execute_tool(tool_name, arguments)
                        is_error = False
                    except Exception as exc:
                        tool_result = f"Error: {exc}"
                        is_error = True

                yield ToolResultEvent(
                    agent_name=self.name,
                    call_id=call_id,
                    tool_name=tool_name,
                    result=tool_result,
                    is_error=is_error,
                )

                self.conversation.messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": str(tool_result)}
                )

        else:
            logger.warning("Agent '%s' reached max_iterations=%d", self.name, self.max_iterations)

        yield FinalAnswerEvent(agent_name=self.name, answer=final_answer)


    async def invoke(self, user_message: str) -> str:
        answer = ""
        async for event in self.stream(user_message):
            if isinstance(event, FinalAnswerEvent):
                answer = event.answer
        return answer