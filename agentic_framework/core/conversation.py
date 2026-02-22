
from dataclasses import dataclass


@dataclass
class Conversation:
    id: str
    system_prompt: str
    messages: list[dict[str, str]] = None
    summarized_history: str = "" #to be added later on
    

    def __post_init__(self):
        if self.messages is None:
            self.messages = [self._prepare_system_prompt()]
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages
    
    def clear(self) -> None:
        self.messages.clear()

    def _prepare_system_prompt(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}
