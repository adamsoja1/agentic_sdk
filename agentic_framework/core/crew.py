from dataclasses import dataclass, field
from .agent import Agent
from .conversation import Conversation
from ..core.stream_events import (
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
        if self.shared_knowledge:
            for agent in self.agents:
                agent.conversation = self.conversation

        for agent in self.agents:
            agent.crew = self

    def get_agent_by_name(self, name: str) -> Agent | None:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def _register_agents_as_tools(self):
        for agent in self.agents:
            for other_agent in self.agents:
                if agent != other_agent and agent.can_delegate:
                    agent.add_tool(other_agent)

    async def invoke(self, input_message: str):
        current_agent = self.entrypoint_agent
        transfer_count = 0

        while True:
            self._register_agents_as_tools()

            async for event in current_agent.stream(input_message):
                if isinstance(event, TextDeltaEvent):
                    yield event
                elif isinstance(event, ToolCallStartEvent):
                    yield event
                elif isinstance(event, ToolResultEvent):
                    yield event
                elif isinstance(event, ErrorEvent):
                    yield event
                    return
                elif isinstance(event, FinalAnswerEvent):
                    yield event
                    return
                elif isinstance(event, DelegationEvent):
                    transfer_count += 1
                    if transfer_count > self.transfer_limit:
                        yield ErrorEvent(
                            agent_name=current_agent.name,
                            error="Transfer limit exceeded. Ending delegation."
                        )
                    current_agent = self.get_agent_by_name(event.target_agent)
                    break
                
            if not self.delegate_to_agent:
                break
