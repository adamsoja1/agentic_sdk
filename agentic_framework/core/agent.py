from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI

from agentic_framework.core.conversation import Conversation
from agentic_framework.tools.base import BaseTool
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
    description: str = ""
    system_prompt: str = ""
    can_delegate: bool = True
    tools: list[BaseTool] = field(default_factory=list)
    conversation: Conversation = field(default_factory=Conversation)
    max_iterations: int = 7
    client: Any = field(default_factory=lambda: _default_client)
    tool_auto_choice: bool = False
    crew: Crew | None = field(default=None, repr=False, compare=False)
    output_format: Any = None

    def __post_init__(self):
        if isinstance(self.tools, list):
            self.tools = {tool.name: tool for tool in self.tools}

    def __prepare_system_prompt(self):
        self.conversation.system_prompt = self.system_prompt

    def add_tool(self, tool: BaseTool | Agent):
        if isinstance(tool, Agent):
            if self.crew.only_ask_for_info or not self.can_delegate:
                tool = BaseTool(
                    name=f"ask_agent_{tool.name}",
                    description=tool.description,
                    func=lambda question, target_agent=tool.name: self._ask_agent(target_agent, question),
                )
            else:
                tool = BaseTool(
                    name=f"delegate_to_agent_{tool.name}",
                    description=tool.description,
                    func=lambda task, target_agent=tool.name: self._delegate(target_agent, task),
                )

        self.tools[tool.name] = tool

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
        logger.warning("Agent '%s' tool keys: %s", self.name, list(self.tools.keys()))
        return schemas

    def _system_messages(self) -> list[dict[str, Any]]:
        self.__prepare_system_prompt()
        msgs: list[dict[str, Any]] = []
        if self.conversation.system_prompt:
            msgs.append({"role": "system", "content": self.conversation.system_prompt})

        if self.crew and self.can_delegate and not self.crew.only_ask_for_info:
            agent_list = ", ".join(a.name for a in self.crew.agents if a.name != self.name)
            crew_hint = (
                f"\nYou are part of a crew. "
                f"Other available agents: [{agent_list}]. "
                "Use `delegate_to_agent_<agent_name>` when a task falls outside your expertise."
            )
            if msgs:
                msgs[0]["content"] += crew_hint
            else:
                msgs.append({"role": "system", "content": crew_hint.strip()})

        if self.crew and self.crew.only_ask_for_info:
            agent_list = ", ".join(a.name for a in self.crew.agents if a.name != self.name)
            info_hint = (
                f"\nYou can ask other specialists for help. "
                f"Other available agents: [{agent_list}]. "
                "Ask them for information when needed, using the `ask_agent_<agent_name>` tool."
            )
            if msgs:
                msgs[0]["content"] += info_hint
            else:
                msgs.append({"role": "system", "content": info_hint.strip()})

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

        target = next((a for a in self.crew.agents if a.name == target_name), None)
        if target is None:
            yield ErrorEvent(agent_name=self.name, error=f"Unknown agent '{target_name}' in crew.")
            return

        yield DelegationEvent(agent_name=self.name, target_agent=target_name, task=task)

    async def _ask_agent(self, target_name: str, question: str) -> AsyncGenerator[StreamEvent, None]:
        if self.crew is None:
            yield ErrorEvent(agent_name=self.name, error="Agent is not part of a crew; cannot ask other agents.")
            return

        target = next((a for a in self.crew.agents if a.name == target_name), None)
        if target is None:
            yield ErrorEvent(agent_name=self.name, error=f"Unknown agent '{target_name}' in crew.")
            return

        original_conversation = target.conversation
        target.conversation = Conversation(id="temp_for_ask_agent")
        target.conversation.system_prompt = original_conversation.system_prompt

        try:
            result = await target.invoke(question)
        finally:
            target.conversation = original_conversation

        yield AskAgentEventResult(
            agent_name=self.name,
            target_agent=target_name,
            question=question,
            result=result,
        )

    def max_iterations_reached(self, iteration_count: int) -> bool:
        return iteration_count >= self.max_iterations - 1

    def modify_prompt_on_max_iterations(self):
        limit_warning = """
        You have reached the maximum number of iterations allowed for this task.
        If you are unable to solve the problem within this limit, provide the best possible answer based on the information you have and explain that you have reached the iteration limit.
        """
        if self.system_prompt:
            self.system_prompt += "\n" + limit_warning

    @staticmethod
    def _parse_tool_arguments(raw: str) -> dict[str, Any]:
        """
        Parse tool call arguments from a raw JSON string produced by the LLM.

        Local/smaller models (e.g. Ollama) sometimes emit extra content after
        a valid JSON object — e.g. two concatenated objects, or a trailing
        comment.  The json.JSONDecodeError.pos attribute tells us exactly where
        the first complete value ended, so we can truncate and retry before
        giving up.

        Raises json.JSONDecodeError if the string cannot be recovered.
        """
        raw = raw.strip() if raw else "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # "Extra data" means a valid JSON value was followed by junk.
            # Truncate at exc.pos and retry once.
            if "Extra data" in str(exc) and exc.pos > 0:
                truncated = raw[: exc.pos]
                logger.warning(
                    "Truncating malformed tool arguments at pos %d (original length %d). "
                    "Truncated: %r",
                    exc.pos,
                    len(raw),
                    truncated,
                )
                return json.loads(truncated)  # may raise again — caller handles it
            raise

    async def stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        if self.client is None:
            yield ErrorEvent(agent_name=self.name, error="No OpenAI client configured.")
            return

        self.conversation.add_message("user", user_message)

        openai_tools = self._build_openai_tools()
        tool_choice: Any = "auto" if openai_tools else "none"

        final_answer = ""
        found_answer = False
        assistant_text = ""

        for iteration in range(self.max_iterations):
            if self.max_iterations_reached(iteration):
                self.modify_prompt_on_max_iterations()
                openai_tools = []  # Disable tools after reaching iteration limit
                tool_choice = "none"

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
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                # Only set name if not yet set — name arrives once per tool call index;
                                # never append — two tools called in parallel have different indices.
                                if not acc["name"]:
                                    acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

                if choice.finish_reason == "stop":
                    break

            # No tool calls — the LLM gave a final text answer
            if not tool_calls_acc:
                self.conversation.add_message("assistant", assistant_text)
                final_answer = assistant_text
                found_answer = True
                break

            # Build the assistant message with tool_calls so conversation history is valid
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or "",
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

            for i, acc in enumerate(tool_calls_acc.values()):
                call_id = acc["id"]
                tool_name = acc["name"]

                try:
                    arguments = self._parse_tool_arguments(acc["arguments"])
                except json.JSONDecodeError as exc:
                    error_msg = (
                        f"Failed to parse arguments for tool '{tool_name}': {exc}. "
                        "Please retry the call with valid JSON arguments."
                    )
                    logger.warning(
                        "Agent '%s' bad tool arguments for '%s': %s", self.name, tool_name, exc
                    )
                    # Append error for this tool call
                    self.conversation.messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": f"Error: {error_msg}"}
                    )
                    # Append placeholder results for all remaining tool calls so the API
                    # sees a complete set of tool results and doesn't return 400
                    remaining = list(tool_calls_acc.values())[i + 1:]
                    for remaining_acc in remaining:
                        self.conversation.messages.append({
                            "role": "tool",
                            "tool_call_id": remaining_acc["id"],
                            "content": "Skipped due to earlier parse error in this batch."
                        })
                    break

                yield ToolCallStartEvent(
                    agent_name=self.name,
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments_raw=acc["arguments"],
                )

                if tool_name.startswith("delegate_to_agent_"):
                    target_name = tool_name.replace("delegate_to_agent_", "")
                    task = arguments.get("task", "")
                    delegation_result = ""
                    async for event in self._delegate(target_name, task):
                        yield event
                    tool_result = delegation_result or f"Delegation to {target_name} completed."
                    is_error = False

                elif tool_name.startswith("ask_agent_"):
                    target_name = tool_name.replace("ask_agent_", "")
                    question = arguments.get("question", "")
                    ask_result = ""
                    async for event in self._ask_agent(target_name, question):
                        if isinstance(event, AskAgentEventResult):
                            ask_result += event.result
                        yield event
                    tool_result = ask_result or f"Asked {target_name} for information."
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

        if not found_answer:
            logger.warning(
                "Agent '%s' never produced a clean final answer after %d iterations",
                self.name,
                self.max_iterations,
            )
            # Surface last assistant text rather than returning empty string
            final_answer = assistant_text if assistant_text else "I was unable to complete this task."

        yield FinalAnswerEvent(agent_name=self.name, answer=final_answer)

    async def invoke(self, user_message: str) -> str:
        answer = ""
        async for event in self.stream(user_message):
            if isinstance(event, FinalAnswerEvent):
                answer = event.answer
                logger.info("Agent '%s' produced final answer: %s", self.name, answer)
        return answer