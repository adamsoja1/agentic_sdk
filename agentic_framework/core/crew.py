from dataclasses import dataclass, field
from agentic_framework.core.agent import Agent
from agentic_framework.core.conversation import Conversation
from agentic_framework.core.stream_events import (
    DelegationEvent,
    ErrorEvent,
    FinalAnswerEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)


@dataclass
class Crew:
    agents: list[Agent]
    entrypoint_agent: Agent
    delegate_to_agent: bool = True
    only_ask_for_info: bool = False
    shared_knowledge: bool = True
    shared_identity: bool = False
    conversation: Conversation = field(default_factory=Conversation)
    system_prompt: str = "You are part of a team of agents working together to solve problems. Collaborate effectively and share information to achieve your goals."
    transfer_limit: int = 5

    def __post_init__(self):
        if self.entrypoint_agent not in self.agents:
            raise ValueError("Entrypoint agent must be part of the agents list.")

        if self.shared_knowledge and self.only_ask_for_info:
            for agent in self.agents:
                agent.conversation = self.conversation

        for agent in self.agents:
            agent.crew = self

        # FIX: register tools once here instead of on every invoke() loop iteration,
        # which was causing tool names to be duplicated (e.g. search_scraped_websitesearch_scraped_website)
        self._register_agents_as_tools()

    def get_agent_by_name(self, name: str) -> Agent | None:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def _register_agents_as_tools(self):
        for agent in self.agents:
            for other_agent in self.agents:
                if agent != other_agent and agent.can_delegate:
                    # FIX: skip if already registered
                    tool_name = (
                        f"ask_agent_{other_agent.name}"
                        if self.only_ask_for_info or not agent.can_delegate
                        else f"delegate_to_agent_{other_agent.name}"
                    )
                    if tool_name not in agent.tools:
                        agent.add_tool(other_agent)

    async def invoke(self, input_message: str):
        current_agent = self.entrypoint_agent
        transfer_count = 0

        while True:
            # FIX: removed _register_agents_as_tools() call from here

            async for event in current_agent.stream(input_message):
                if isinstance(event, TextDeltaEvent):
                    yield event
                elif isinstance(event, ToolCallStartEvent):
                    yield event
                elif isinstance(event, ToolResultEvent):
                    yield event
                elif isinstance(event, ErrorEvent):
                    yield event
                elif isinstance(event, FinalAnswerEvent):
                    yield event
                    return
                elif isinstance(event, DelegationEvent):
                    transfer_count += 1
                    if transfer_count > self.transfer_limit:
                        yield ErrorEvent(
                            agent_name=current_agent.name,
                            error="Transfer limit exceeded. Ending delegation.",
                        )
                        return
                    current_agent = self.get_agent_by_name(event.target_agent)
                    if current_agent is None:
                        yield ErrorEvent(
                            agent_name=event.agent_name,
                            error=f"Delegation target '{event.target_agent}' not found in crew.",
                        )
                        return
                    break

            if not self.delegate_to_agent:
                break

    async def get_response(self, input_message: str) -> str:
        final_answer = ""
        async for event in self.invoke(input_message):
            if isinstance(event, FinalAnswerEvent):
                final_answer = event.answer
        return final_answer