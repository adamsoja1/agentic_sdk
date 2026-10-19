from __future__ import annotations

import base64
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence, Union


# ─────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────

Role = Literal["user", "assistant", "tool"]


# ─────────────────────────────────────────────────────────────
# EVENTS (SOURCE OF TRUTH)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImageInput:
    """An image URL or base64 data URL for a vision-capable model."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"

    def __post_init__(self):
        if not self.url.startswith(("https://", "http://", "data:image/")):
            raise ValueError("Image URL must be HTTP(S) or an image data URL.")
        if self.detail not in ("auto", "low", "high"):
            raise ValueError("Image detail must be auto, low, or high.")

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str, *, detail: Literal["auto", "low", "high"] = "auto") -> ImageInput:
        if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError("Unsupported image media type.")
        encoded = base64.b64encode(data).decode("ascii")
        return cls(url=f"data:{media_type};base64,{encoded}", detail=detail)


ImageInputs = Sequence[ImageInput | str]


@dataclass
class BaseEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MessageEvent(BaseEvent):
    role: Role = "user"
    content: str = ""
    images: list[ImageInput] = field(default_factory=list)


@dataclass
class AssistantToolCallEvent(BaseEvent):
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolResultEvent(BaseEvent):
    tool_call_id: str = ""
    content: str = ""


@dataclass
class ReasoningEvent(BaseEvent):
    content: str = ""  # NIE trafia do LLM


@dataclass
class TextDeltaEvent(BaseEvent):
    delta: str = ""  # streaming (opcjonalne)


# Union typu dla wygody
ConversationEvent = Union[
    MessageEvent,
    AssistantToolCallEvent,
    ToolResultEvent,
    ReasoningEvent,
    TextDeltaEvent,
]


# ─────────────────────────────────────────────────────────────
# CONVERSATION
# ─────────────────────────────────────────────────────────────

@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[ConversationEvent] = field(default_factory=list)

    system_prompt: str = ""
    summarized_history: str = ""

    # ─────────────────────────────────────────
    # ADD EVENTS
    # ─────────────────────────────────────────

    def add_user_message(self, content: str, *, images: ImageInputs = ()) -> None:
        normalized = [ImageInput(image) if isinstance(image, str) else image for image in images]
        self.events.append(MessageEvent(role="user", content=content, images=normalized))

    def add_assistant_message(self, content: str) -> None:
        self.events.append(MessageEvent(role="assistant", content=content))

    def add_assistant_tool_calls(
        self,
        content: str,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """
        tool_calls format:
        {
            "id": "...",
            "type": "function",
            "function": {"name": "...", "arguments": "..."}
        }
        """
        self.events.append(
            AssistantToolCallEvent(
                content=content or "",
                tool_calls=tool_calls,
            )
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> None:
        self.events.append(
            ToolResultEvent(
                tool_call_id=tool_call_id,
                content=content,
            )
        )

    # ─────────────────────────────────────────
    # OPTIONAL EVENTS
    # ─────────────────────────────────────────

    def add_reasoning(self, content: str) -> None:
        """Hidden reasoning (NOT sent to LLM)"""
        self.events.append(ReasoningEvent(content=content))

    def add_delta(self, delta: str) -> None:
        """Streaming token (optional)"""
        self.events.append(TextDeltaEvent(delta=delta))

    # ─────────────────────────────────────────
    # TRANSFORM → OPENAI FORMAT
    # ─────────────────────────────────────────

    def get_messages(self) -> list[dict[str, Any]]:
        """
        Transform events into OpenAI-compatible messages.
        """
        messages: list[dict[str, Any]] = []

        # system prompt
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        for event in self.events:
            # ── user / assistant messages ─────────────────
            if isinstance(event, MessageEvent):
                messages.append(
                    {
                        "role": event.role,
                        "content": self._message_content(event),
                    }
                )

            # ── assistant tool calls ─────────────────────
            elif isinstance(event, AssistantToolCallEvent):
                messages.append(
                    {
                        "role": "assistant",
                        "content": event.content,
                        "tool_calls": event.tool_calls,
                    }
                )

            # ── tool results ─────────────────────────────
            elif isinstance(event, ToolResultEvent):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.tool_call_id,
                        "content": event.content,
                    }
                )

            # ── skip hidden/internal events ──────────────
            elif isinstance(event, (ReasoningEvent, TextDeltaEvent)):
                continue

        return messages

    @staticmethod
    def _message_content(event: MessageEvent, *, responses: bool = False) -> str | list[dict[str, Any]]:
        if not event.images:
            return event.content
        parts: list[dict[str, Any]] = []
        if event.content:
            parts.append({"type": "input_text" if responses else "text", "text": event.content})
        for image in event.images:
            if responses:
                parts.append({"type": "input_image", "image_url": image.url, "detail": image.detail})
            else:
                parts.append({"type": "image_url", "image_url": {"url": image.url, "detail": image.detail}})
        return parts

    def get_latest_user_images(self) -> list[ImageInput]:
        for event in reversed(self.events):
            if isinstance(event, MessageEvent) and event.role == "user":
                return list(event.images)
        return []

    def get_response_input(self) -> list[dict[str, Any]]:
        """Transform events into the Responses API input item format."""
        items: list[dict[str, Any]] = []
        for event in self.events:
            if isinstance(event, MessageEvent):
                items.append({"role": event.role, "content": self._message_content(event, responses=True)})
            elif isinstance(event, AssistantToolCallEvent):
                if event.content:
                    items.append({"role": "assistant", "content": event.content})
                for call in event.tool_calls:
                    function = call.get("function", {})
                    items.append({
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    })
            elif isinstance(event, ToolResultEvent):
                items.append({
                    "type": "function_call_output",
                    "call_id": event.tool_call_id,
                    "output": event.content,
                })
        return items

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def clear(self) -> None:
        self.events.clear()

    def dump_events(self) -> list[dict[str, Any]]:
        """Debug helper"""
        return [asdict(event) for event in self.events]

    def get_last_messages(self, n: int) -> list[dict[str, Any]]:
        return self.get_messages()[-n:]

    def __len__(self) -> int:
        return len(self.events)
