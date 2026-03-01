from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI

from agentic_framework.core.conversation import Conversation
from agentic_framework.tools.base import BaseTool, Skill
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

_DEACTIVATE_TOOL = BaseTool(
    name="deactivate_skill",
    description=(
        "Deactivate the current skill and return to the base toolset. "
        "Call this when you are done using the current skill's tools."
    ),
)


@dataclass
class Agent:
    name: str
    model: str
    description: str = ""
    system_prompt: str = ""
    can_delegate: bool = True
    tools: list[BaseTool] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    conversation: Conversation = field(default_factory=Conversation)
    max_iterations: int = 7
    client: Any = field(default_factory=lambda: _default_client)
    tool_auto_choice: bool = False
    crew: Crew | None = field(default=None, repr=False, compare=False)
    output_format: Any = None

    def __post_init__(self):
        if isinstance(self.tools, list):
            self.tools = {tool.name: tool for tool in self.tools}

        if self.skills:
            skills_list = "\n".join(f"- {skill.name}: {skill.description}" for skill in self.skills)
            self._skill_prompt = (
                f"\n\nYou have the following skills available. "
                f"Activate a skill when the task matches its domain — this will unlock its specific tools. "
                f"You should first call skill_<skill_name> to activate a skill, then use its tools as needed. "
                f"Call `deactivate_skill` when you are done with the skill's tools to return to the base toolset. "
                f"You can switch directly between skills without calling `deactivate_skill` first.\n\n"
                f"Available skills:\n{skills_list}"
                f"You can activate only one skill at a time, and activating a new skill will deactivate the previous one."
            )
            for skill in self.skills:
                self.tools[skill.name] = skill

        # Snapshot of the base toolset — used to restore after skill deactivation
        self._base_tools: dict[str, BaseTool] = dict(self.tools)
        self._active_skill: Skill | None = None
        self._active_skill_prompt: str = ""

    def _activate_skill(self, skill: Skill) -> str:
        self.tools = {}
        self.tools.update({t.name: t for t in skill.tools})
        self.tools[_DEACTIVATE_TOOL.name] = _DEACTIVATE_TOOL
        self._active_skill = skill

        self._active_skill_prompt = (
            f"\n\n[Currently Active Skill: {skill.name}]\n"
            f"Call `deactivate_skill` when you are done."
        )

        return f"Skill '{skill.name}' activated."

    def _deactivate_skill(self) -> str:
        if self._active_skill is None:
            return "No skill is currently active."
        skill_name = self._active_skill.name
        self.tools = dict(self._base_tools)
        self._active_skill = None
        self._active_skill_prompt = ""
        return f"Skill '{skill_name}' deactivated."

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
        # Keep base snapshot in sync when tools are added externally
        self._base_tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        self._base_tools.pop(name, None)
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
        parts = [self.system_prompt]

        if self.skills:
            parts.append(self._skill_prompt)

        if self._active_skill:
            parts.append(self._active_skill_prompt)

        if self.crew:
            agent_list = ", ".join(a.name for a in self.crew.agents if a.name != self.name)
            if self.can_delegate and not self.crew.only_ask_for_info:
                parts.append(
                    f"\nYou are part of a crew. "
                    f"Other available agents: [{agent_list}]. "
                    "Use `delegate_to_agent_<agent_name>` when a task falls outside your expertise."
                )
            elif self.crew.only_ask_for_info:
                parts.append(
                    f"\nYou can ask other specialists for help. "
                    f"Other available agents: [{agent_list}]. "
                    "Ask them for information when needed, using the `ask_agent_<agent_name>` tool."
                )

        system_prompt = "".join(parts)
        self.conversation.system_prompt = system_prompt

        return [{"role": "system", "content": system_prompt}] if system_prompt else []

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
        limit_warning = (
            "\nYou have reached the maximum number of iterations allowed for this task. "
            "Provide the best possible answer based on the information you have and explain "
            "that you have reached the iteration limit."
        )
        self.system_prompt += limit_warning

    @staticmethod
    def _parse_tool_arguments(raw: str) -> tuple[dict[str, Any], str]:
        raw = raw.strip() if raw else "{}"
        try:
            parsed = json.loads(raw)
            return parsed, raw
        except json.JSONDecodeError as exc:
            if "Extra data" in str(exc) and exc.pos > 0:
                truncated = raw[: exc.pos]
                logger.warning(
                    "Truncating malformed tool arguments at pos %d (original length %d). Truncated: %r",
                    exc.pos, len(raw), truncated,
                )
                parsed = json.loads(truncated)
                clean = json.dumps(parsed)
                return parsed, clean
            raise

    @staticmethod
    def _extract_text_from_delta(delta: Any) -> str:
        """Extract only the final answer text from a delta, ignoring reasoning_content.

        Reasoning models (deepseek-r1, qwq, etc.) stream thinking tokens via
        ``delta.reasoning_content``.  We deliberately skip that field so that
        ``assistant_text`` — and ultimately the conversation history — contains
        only the actual response, not the internal chain-of-thought.
        """
        content = delta.content
        if not content:
            return ""
        return content

    async def stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        if self.client is None:
            yield ErrorEvent(agent_name=self.name, error="No OpenAI client configured.")
            return

        self.conversation.add_message("user", user_message)

        final_answer = ""
        found_answer = False
        assistant_text = ""

        for iteration in range(self.max_iterations):
            openai_tools = self._build_openai_tools()
            tool_choice: Any = "auto" if openai_tools else "none"

            if self.max_iterations_reached(iteration):
                self.modify_prompt_on_max_iterations()
                openai_tools = []
                tool_choice = "none"

            messages = self._system_messages() + self.conversation.get_messages()

            stream_kwargs: dict[str, Any] = dict(model=self.model, messages=messages, stream=True)
            if openai_tools:
                stream_kwargs["tools"] = openai_tools
                stream_kwargs["tool_choice"] = tool_choice

            try:
                response_stream = await self.client.chat.completions.create(**stream_kwargs)
            except Exception as exc:
                logger.error("Agent '%s' API error on iteration %d: %s", self.name, iteration, exc)
                yield ErrorEvent(agent_name=self.name, error=str(exc))
                break

            assistant_text = ""
            reasoning_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            async for chunk in response_stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                # ── Reasoning content (thinking tokens) ──────────────────────
                # Present on deepseek-r1, qwq and similar models.
                # We accumulate it separately so it never leaks into
                # assistant_text or the conversation history.
                raw_reasoning = getattr(delta, "reasoning_content", None)
                if raw_reasoning:
                    reasoning_text += raw_reasoning
                    logger.debug(
                        "Agent '%s' reasoning token (iteration %d): %r",
                        self.name, iteration, raw_reasoning,
                    )

                # ── Visible assistant content ─────────────────────────────────
                text_chunk = self._extract_text_from_delta(delta)
                if text_chunk:
                    assistant_text += text_chunk
                    yield TextDeltaEvent(agent_name=self.name, delta=text_chunk)

                # ── Tool call fragments ───────────────────────────────────────
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        acc = tool_calls_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name and not acc["name"]:
                                acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                    # Do NOT break here — let the async-for exhaust the stream
                    # naturally.  Some reasoning models send content *after* the
                    # chunk that carries finish_reason, and breaking early would
                    # drop those tokens.

            logger.debug(
                "Agent '%s' iteration %d finished. finish_reason=%r tool_calls=%d "
                "assistant_text_len=%d reasoning_text_len=%d",
                self.name, iteration, finish_reason,
                len(tool_calls_acc), len(assistant_text), len(reasoning_text),
            )

            # ── No tool calls → model produced its final answer ───────────────
            if not tool_calls_acc:
                if not assistant_text.strip():
                    # Model returned nothing visible (only reasoning, or truly
                    # empty).  Log and retry rather than yielding a blank answer.
                    logger.warning(
                        "Agent '%s' returned empty assistant_text on iteration %d "
                        "(finish_reason=%r, reasoning_len=%d). Retrying.",
                        self.name, iteration, finish_reason, len(reasoning_text),
                    )
                    # Feed a gentle nudge so the model actually answers.
                    self.conversation.add_message(
                        "user",
                        "Please provide your answer based on your reasoning above.",
                    )
                    continue

                self.conversation.add_message("assistant", assistant_text)
                final_answer = assistant_text
                found_answer = True
                break

            # ── Parse tool-call arguments ─────────────────────────────────────
            parsed_tool_calls: list[tuple[dict[str, Any], str, str, str]] = []
            parse_error: str | None = None
            parse_error_index: int = 0

            for i, acc in enumerate(tool_calls_acc.values()):
                try:
                    parsed_args, clean_args = self._parse_tool_arguments(acc["arguments"])
                    parsed_tool_calls.append((parsed_args, clean_args, acc["id"], acc["name"]))
                except json.JSONDecodeError as exc:
                    parse_error = (
                        f"Failed to parse arguments for tool '{acc['name']}': {exc}. "
                        "Please retry the call with valid JSON arguments."
                    )
                    logger.warning(
                        "Agent '%s' bad tool arguments for '%s': %s",
                        self.name, acc["name"], exc,
                    )
                    parse_error_index = i
                    for remaining_acc in list(tool_calls_acc.values())[i:]:
                        parsed_tool_calls.append(({}, "{}", remaining_acc["id"], remaining_acc["name"]))
                    break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": clean_args},
                    }
                    for _, clean_args, call_id, tool_name in parsed_tool_calls
                ],
            }
            self.conversation.messages.append(assistant_msg)

            if parse_error is not None:
                failed_call_id = parsed_tool_calls[parse_error_index][2]
                self.conversation.messages.append(
                    {"role": "tool", "tool_call_id": failed_call_id, "content": f"Error: {parse_error}"}
                )
                for _, _, remaining_call_id, _ in parsed_tool_calls[parse_error_index + 1:]:
                    self.conversation.messages.append({
                        "role": "tool",
                        "tool_call_id": remaining_call_id,
                        "content": "Skipped due to earlier parse error in this batch.",
                    })
                continue

            # Rebuild openai_tools after each iteration — toolset may have
            # changed due to skill activation/deactivation.
            openai_tools = self._build_openai_tools()

            for arguments, clean_args, call_id, tool_name in parsed_tool_calls:
                yield ToolCallStartEvent(
                    agent_name=self.name,
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments_raw=clean_args,
                )

                is_error = False

                if tool_name.startswith("delegate_to_agent_"):
                    target_name = tool_name.replace("delegate_to_agent_", "")
                    task = arguments.get("task", "")
                    delegation_result = ""
                    async for event in self._delegate(target_name, task):
                        yield event
                    tool_result = delegation_result or f"Delegation to {target_name} completed."

                elif tool_name.startswith("ask_agent_"):
                    target_name = tool_name.replace("ask_agent_", "")
                    question = arguments.get("question", "")
                    ask_result = ""
                    async for event in self._ask_agent(target_name, question):
                        if isinstance(event, AskAgentEventResult):
                            ask_result += event.result
                        yield event
                    tool_result = ask_result or f"Asked {target_name} for information."

                elif tool_name.startswith("skill_"):
                    skill = next((s for s in self.skills if s.name == tool_name), None)
                    if skill is None:
                        tool_result = f"Error: Skill '{tool_name}' not found."
                        is_error = True
                    else:
                        tool_result = self._activate_skill(skill)

                elif tool_name == "deactivate_skill":
                    tool_result = self._deactivate_skill()

                else:
                    try:
                        tool_result = await self._execute_tool(tool_name, arguments)
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
                self.name, self.max_iterations,
            )
            final_answer = assistant_text if assistant_text else "I was unable to complete this task."

        yield FinalAnswerEvent(agent_name=self.name, answer=final_answer)

    async def invoke(self, user_message: str) -> str:
        answer = ""
        async for event in self.stream(user_message):
            if isinstance(event, FinalAnswerEvent):
                answer = event.answer
                logger.info("Agent '%s' produced final answer: %s", self.name, answer)
        return answer