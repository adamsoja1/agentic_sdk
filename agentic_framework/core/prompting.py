from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


PromptContent = str | Callable[[Any], str]


@dataclass
class PromptBlock:
    title: str
    content: str = ""
    items: list[str] = field(default_factory=list)
    enabled: bool = True

    def render(self) -> str:
        if not self.enabled:
            return ""

        lines: list[str] = []
        heading = self.title.strip()
        body = self.content.strip()
        bullets = [item.strip() for item in self.items if item and item.strip()]

        if heading:
            lines.append(f"{heading}:")
        if body:
            lines.append(body)
        lines.extend(f"- {item}" for item in bullets)

        if not body and not bullets:
            return ""
        return "\n".join(lines)


@dataclass
class AgentPrompt:
    role: str = ""
    objectives: list[str] = field(default_factory=list)
    context: str = ""
    instructions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    output_requirements: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    additional_sections: list[PromptBlock] = field(default_factory=list)
    separator: str = "\n\n"

    def render(self) -> str:
        sections = [
            PromptBlock(title="Role", content=self.role),
            PromptBlock(title="Objectives", items=self.objectives),
            PromptBlock(title="Context", content=self.context),
            PromptBlock(title="Instructions", items=self.instructions),
            PromptBlock(title="Constraints", items=self.constraints),
            PromptBlock(title="Style", items=self.style),
            PromptBlock(title="Output Requirements", items=self.output_requirements),
            PromptBlock(title="Success Criteria", items=self.success_criteria),
            *self.additional_sections,
        ]

        rendered_sections = [section.render() for section in sections]
        return self.separator.join(section for section in rendered_sections if section)


@dataclass
class PromptSection:
    name: str
    content: PromptContent = ""
    enabled: bool = True
    locked: bool = False

    def render(self, owner: Any = None) -> str:
        if not self.enabled:
            return ""

        rendered = self.content(owner) if callable(self.content) else self.content
        if rendered is None:
            return ""
        return str(rendered).strip()


@dataclass
class PromptTemplate:
    sections: list[PromptSection] = field(default_factory=list)
    separator: str = "\n\n"

    def clone(self) -> PromptTemplate:
        return PromptTemplate(
            sections=[
                PromptSection(
                    name=section.name,
                    content=section.content,
                    enabled=section.enabled,
                    locked=section.locked,
                )
                for section in self.sections
            ],
            separator=self.separator,
        )

    def get_section(self, name: str) -> PromptSection | None:
        return next((section for section in self.sections if section.name == name), None)

    def set_section(
        self,
        name: str,
        content: PromptContent,
        *,
        enabled: bool = True,
        locked: bool | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> PromptSection:
        if before and after:
            raise ValueError("Use either 'before' or 'after' when placing a prompt section.")

        existing = self.get_section(name)

        if existing is not None and existing.locked:
            raise ValueError(f"Prompt section '{name}' is locked and cannot be replaced.")

        new_section = PromptSection(
            name=name,
            content=content,
            enabled=enabled,
            locked=locked if locked is not None else (existing.locked if existing is not None else False),
        )

        if existing is not None:
            index = self.sections.index(existing)
            self.sections[index] = new_section
            return new_section

        if before is not None:
            target = self.get_section(before)
            if target is None:
                raise KeyError(f"Prompt section '{before}' not found.")
            self.sections.insert(self.sections.index(target), new_section)
            return new_section

        if after is not None:
            target = self.get_section(after)
            if target is None:
                raise KeyError(f"Prompt section '{after}' not found.")
            self.sections.insert(self.sections.index(target) + 1, new_section)
            return new_section

        self.sections.append(new_section)
        return new_section

    def remove_section(self, name: str) -> bool:
        section = self.get_section(name)
        if section is None:
            return False
        if section.locked:
            raise ValueError(f"Prompt section '{name}' is locked and cannot be removed.")
        self.sections.remove(section)
        return True

    def render(self, owner: Any = None) -> str:
        parts = [section.render(owner) for section in self.sections]
        return self.separator.join(part for part in parts if part)
