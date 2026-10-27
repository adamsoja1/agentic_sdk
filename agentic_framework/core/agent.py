from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI

from agentic_framework.core.conversation import Conversation, ImageInputs
from agentic_framework.core.prompting import AgentPrompt, PromptContent, PromptTemplate
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
    base_url=os.getenv("LLM_PROVIDER", "https://ollama.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
)


@dataclass
class Agent:
    name: str
    model: str
    description: str = ""
    system_prompt: str = ""
    prompt: AgentPrompt | None = None
    can_delegate: bool = True
    tools: list[BaseTool] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    conversation: Conversation = field(default_factory=Conversation)
    prompt_template: PromptTemplate | None = None
    max_iterations: int = 7
    client: Any = field(default_factory=lambda: _default_client)
    tool_auto_choice: bool = False
    web_search: bool = False
    crew: Crew | None = field(default=None, repr=False, compare=False)
    output_format: Any = None
    reasoning_effort: str = "medium"  # "low", "medium", "high"
    max_iteration_warning_prompt: str = (
        "You have reached the maximum number of iterations allowed for this task. "
        "Provide the best possible answer based on the information you have and explain "
        "that you have reached the iteration limit."
    )

    def __post_init__(self):
        if isinstance(self.tools, list):
            self.tools = {tool.name: tool for tool in self.tools}

        if self.skills:
            for skill in self.skills:
                self.tools[skill.name] = skill

        # Snapshot of the base toolset — used to restore after skill deactivation
        self._base_tools: dict[str, BaseTool] = dict(self.tools)
        self._active_skill: Skill | None = None
        self._max_iterations_warning_enabled = False
        self.prompt_template = self._initialize_prompt_template()

    def _activate_skill(self, skill: Skill) -> str:
        prev_skill_name = self._active_skill.name if self._active_skill else None

        # Start from base toolset so all skills remain visible
        self.tools = dict(self._base_tools)
        # Overlay the new skill's tools
        self.tools.update({t.name: t for t in skill.tools})
        self._active_skill = skill

        msg = f"Skill '{skill.name}' activated."
        if prev_skill_name:
            msg = f"Skill '{prev_skill_name}' deactivated. {msg}"
        return msg

    def _deactivate_skill(self) -> str:
        if self._active_skill is None:
            return "No skill is currently active."
        skill_name = self._active_skill.name
        self.tools = dict(self._base_tools)
        self._active_skill = None
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
            "web_search": self.web_search,
        }

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        schemas = [
            {"type": "web_search"}
        ] if self.web_search else []
        schemas.extend([
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t._build_parameters(),
            }
            for t in self.tools.values()
        ])
        logger.warning("Agent '%s' tool keys: %s", self.name, list(self.tools.keys()))
        return schemas

    def _initialize_prompt_template(self) -> PromptTemplate:
        template = self.prompt_template.clone() if self.prompt_template is not None else PromptTemplate()

        self._prepend_prompt_section(template, "system_prompt", self._render_system_prompt_section)
        self._prepend_prompt_section(template, "prompt", self._render_agent_prompt)

        if template.get_section("skills") is None:
            template.set_section("skills", self._render_skills_prompt)

        if template.get_section("active_skill") is None:
            template.set_section("active_skill", self._render_active_skill_prompt)

        if template.get_section("crew") is None:
            template.set_section("crew", self._render_crew_prompt)

        if template.get_section("iteration_limit") is None:
            template.set_section("iteration_limit", self._render_iteration_limit_prompt)

        return template

    @staticmethod
    def _prepend_prompt_section(
        template: PromptTemplate,
        name: str,
        content: PromptContent,
    ) -> None:
        if template.get_section(name) is not None:
            return
        if template.sections:
            template.set_section(name, content, before=template.sections[0].name)
            return
        template.set_section(name, content)

    def _render_agent_prompt(self, _: Agent | None = None) -> str:
        if self.prompt is not None:
            return self.prompt.render().strip()
        return ""

    def _render_system_prompt_section(self, _: Agent | None = None) -> str:
        return self.system_prompt.strip()

    @staticmethod
    def _format_skill_tool_list(skill: Skill) -> str:
        return ", ".join(tool.name for tool in skill.tools)

    def _render_skills_prompt(self, _: Agent | None = None) -> str:
        if not self.skills:
            return ""

        lines = [
            "Skills are available for domain-specific work.",
            "How to use skills:",
            "- Call the matching `skill_<name>` tool before using the tools it unlocks.",
            "- Activating a new skill unlocks that skill's tools and automatically deactivates the previously active skill.",
            "- After activation, use the unlocked tools for the current task until you switch skills again.",
            "Available skills:",
        ]

        for skill in self.skills:
            tool_list = self._format_skill_tool_list(skill) or "No unlocked tools."
            description = skill.description or "No description provided."
            lines.append(f"- {skill.name}: {description} Activate this skill to unlock: {tool_list}.")

        return "\n".join(lines)

    def _render_active_skill_prompt(self, _: Agent | None = None) -> str:
        if self._active_skill is None:
            return ""

        tool_list = self._format_skill_tool_list(self._active_skill) or "No unlocked tools."
        description = self._active_skill.description or "No description provided."
        return "\n".join(
            [
                f"Active skill: {self._active_skill.name}",
                f"Current skill focus: {description}",
                f"Unlocked tools from this skill: {tool_list}",
                "The previously active skill, if any, has been deactivated. Activate another skill at any time to switch context.",
            ]
        )

    def _render_crew_prompt(self, _: Agent | None = None) -> str:
        if self.crew is None:
            return ""

        agent_list = ", ".join(a.name for a in self.crew.agents if a.name != self.name)
        if not agent_list:
            return ""

        if self.can_delegate and not self.crew.only_ask_for_info:
            return (
                "You are part of a crew.\n"
                f"Other available agents: [{agent_list}].\n"
                "Use `delegate_to_agent_<agent_name>` when a task falls outside your expertise."
            )

        if self.crew.only_ask_for_info:
            return (
                "You can ask other specialists for help.\n"
                f"Other available agents: [{agent_list}].\n"
                "Ask them for information when needed, using the `ask_agent_<agent_name>` tool."
            )

        return ""

    def _render_iteration_limit_prompt(self, _: Agent | None = None) -> str:
        if not self._max_iterations_warning_enabled:
            return ""
        return self.max_iteration_warning_prompt

    def render_system_prompt(self) -> str:
        return self.prompt_template.render(self)

    def set_prompt_section(
        self,
        name: str,
        content: PromptContent,
        *,
        enabled: bool = True,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        self.prompt_template.set_section(
            name=name,
            content=content,
            enabled=enabled,
            before=before,
            after=after,
        )
        self._rebuild_system_prompt()

    def remove_prompt_section(self, name: str) -> bool:
        removed = self.prompt_template.remove_section(name)
        if removed:
            self._rebuild_system_prompt()
        return removed

    def _rebuild_system_prompt(self) -> None:
        """Rebuild and store the system prompt on the conversation object."""
        self.conversation.system_prompt = self.render_system_prompt()

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
        target.conversation = Conversation()
        target.conversation.system_prompt = original_conversation.system_prompt
        try:
            result = await target.invoke(question, images=self.conversation.get_latest_user_images())
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
        self._max_iterations_warning_enabled = True

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

    async def stream(self, user_message: str, *, images: ImageInputs = ()) -> AsyncGenerator[StreamEvent, None]:
        if self.client is None:
            yield ErrorEvent(agent_name=self.name, error="No OpenAI client configured.")
            return

        self.conversation.add_user_message(user_message, images=images)

        final_answer = ""
        found_answer = False
        assistant_text = ""

        for iteration in range(self.max_iterations):
            openai_tools = self._build_openai_tools()
            tool_choice: Any = "auto" if openai_tools else "none"

            if self.max_iterations_reached(iteration):
                self.modify_prompt_on_max_iterations()
                tool_choice = "none"

            self._rebuild_system_prompt()
            stream_kwargs: dict[str, Any] = dict(
                model=self.model,
                input=self.conversation.get_response_input(),
                instructions=self.conversation.system_prompt,
                stream=True,
            )

            if openai_tools:
                stream_kwargs["tools"] = openai_tools
                stream_kwargs["tool_choice"] = tool_choice
                stream_kwargs["reasoning"] = {"effort": self.reasoning_effort}

            try:
                response_stream = await self.client.responses.create(**stream_kwargs)
            except Exception as exc:
                logger.error("Agent '%s' API error on iteration %d: %s", self.name, iteration, exc)
                yield ErrorEvent(agent_name=self.name, error=str(exc))
                break

            assistant_text = ""
            tool_calls_acc: dict[str, dict[str, Any]] = {}
            async for chunk in response_stream:
                event_type = getattr(chunk, "type", "")
                text_chunk = getattr(chunk, "delta", "") if event_type == "response.output_text.delta" else ""
                if text_chunk:
                    assistant_text += text_chunk
                    yield TextDeltaEvent(agent_name=self.name, delta=text_chunk)

                if event_type == "response.output_item.added":
                    item = getattr(chunk, "item", None)
                    if getattr(item, "type", None) == "function_call":
                        tool_calls_acc[item.id] = {
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": getattr(item, "arguments", "") or "",
                        }
                elif event_type == "response.function_call_arguments.delta":
                    item_id = getattr(chunk, "item_id", None)
                    if item_id in tool_calls_acc:
                        tool_calls_acc[item_id]["arguments"] += chunk.delta
                elif event_type == "response.output_item.done":
                    item = getattr(chunk, "item", None)
                    if getattr(item, "type", None) == "function_call":
                        acc = tool_calls_acc.setdefault(
                            item.id,
                            {"id": item.call_id, "name": item.name, "arguments": ""},
                        )
                        acc["arguments"] = getattr(item, "arguments", "") or acc["arguments"]

            # ─────────────────────────────────────────
            # FINAL ANSWER
            # ─────────────────────────────────────────
            if not tool_calls_acc:
                if not assistant_text.strip():
                    logger.warning(
                        "Agent '%s' returned empty assistant_text on iteration %d",
                        self.name, iteration,
                    )
                    self.conversation.add_user_message(
                        "Please provide your answer based on your reasoning above."
                    )
                    continue

                self.conversation.add_assistant_message(assistant_text)

                final_answer = assistant_text
                found_answer = True
                break

            # ─────────────────────────────────────────
            # PARSE TOOL CALLS
            # ─────────────────────────────────────────
            parsed_tool_calls: list[tuple[dict[str, Any], str, str, str]] = []
            parse_error: str | None = None
            parse_error_index: int = 0

            for i, acc in enumerate(tool_calls_acc.values()):
                try:
                    parsed_args, clean_args = self._parse_tool_arguments(acc["arguments"])
                    parsed_tool_calls.append((parsed_args, clean_args, acc["id"], acc["name"]))
                except json.JSONDecodeError as exc:
                    parse_error = f"Failed to parse arguments for tool '{acc['name']}': {exc}"
                    parse_error_index = i

                    for remaining_acc in list(tool_calls_acc.values())[i:]:
                        parsed_tool_calls.append(({}, "{}", remaining_acc["id"], remaining_acc["name"]))
                    break

            self.conversation.add_assistant_tool_calls(
                content=assistant_text or "",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": clean_args},
                    }
                    for _, clean_args, call_id, tool_name in parsed_tool_calls
                ],
            )

            if parse_error is not None:
                failed_call_id = parsed_tool_calls[parse_error_index][2]

                self.conversation.add_tool_result(
                    tool_call_id=failed_call_id,
                    content=f"Error: {parse_error}",
                )

                for _, _, remaining_call_id, _ in parsed_tool_calls[parse_error_index + 1:]:
                    self.conversation.add_tool_result(
                        tool_call_id=remaining_call_id,
                        content="Skipped due to earlier parse error.",
                    )
                continue

            # ─────────────────────────────────────────
            # EXECUTE TOOLS
            # ─────────────────────────────────────────
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
                    async for event in self._delegate(target_name, arguments.get("task", "")):
                        yield event
                    tool_result = f"Delegation to {target_name} completed."

                elif tool_name.startswith("ask_agent_"):
                    target_name = tool_name.replace("ask_agent_", "")
                    ask_result = ""

                    async for event in self._ask_agent(target_name, arguments.get("question", "")):
                        if isinstance(event, AskAgentEventResult):
                            ask_result += event.result
                        yield event

                    tool_result = ask_result or f"Asked {target_name}."

                elif tool_name.startswith("skill_"):
                    skill = next((s for s in self.skills if s.name == tool_name), None)
                    if skill:
                        tool_result = self._activate_skill(skill)
                    else:
                        tool_result = f"Error: Skill not found"
                        is_error = True

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

                self.conversation.add_tool_result(
                    tool_call_id=call_id,
                    content=str(tool_result),
                )

        if not found_answer:
            final_answer = assistant_text or "I was unable to complete this task."

        yield FinalAnswerEvent(agent_name=self.name, answer=final_answer)

    async def invoke(self, user_message: str, *, images: ImageInputs = ()) -> str:
        answer = ""
        async for event in self.stream(user_message, images=images):
            if isinstance(event, FinalAnswerEvent):
                answer = event.answer
                logger.info("Agent '%s' produced final answer: %s", self.name, answer)
        return answer
