from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable


PromptContent = str | Callable[[Any], str]
PointPrefix = str | Callable[[int, str], str] | None

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_UNORDERED_ITEM_RE = re.compile(r"^\s*([-*+])\s+(.*)$")
_ORDERED_ITEM_RE = re.compile(r"^\s*\d+([.)])\s+(.*)$")
_CHECKBOX_ITEM_RE = re.compile(r"^\s*([-*+])\s+\[( |x|X)\]\s+(.*)$")


@dataclass
class PromptBlock:
    title: str
    content: str = ""
    items: list[str] = field(default_factory=list)
    heading_suffix: str = ":"
    point_prefix: PointPrefix = "- "
    enabled: bool = True

    def render(self) -> str:
        if not self.enabled:
            return ""

        lines: list[str] = []
        heading = self.title.strip()
        body = self.content.strip()
        bullets = [item.strip() for item in self.items if item and item.strip()]

        if heading:
            lines.append(f"{heading}{self.heading_suffix}")
        if body:
            lines.append(body)
        lines.extend(self._render_item(index, item) for index, item in enumerate(bullets, start=1))

        if not body and not bullets:
            return ""
        return "\n".join(lines)

    def _render_item(self, index: int, item: str) -> str:
        if self.point_prefix is None:
            return item
        if callable(self.point_prefix):
            return f"{self.point_prefix(index, item)}{item}"
        return f"{self.point_prefix.format(index=index, index0=index - 1)}{item}"

    @classmethod
    def from_markdown(
        cls,
        markdown: str,
        *,
        title: str = "",
        heading_suffix: str = ":",
        point_prefix: PointPrefix = "- ",
    ) -> PromptBlock:
        blocks = _parse_markdown_blocks(
            markdown,
            heading_suffix=heading_suffix,
            point_prefix=point_prefix,
        )
        if blocks:
            return blocks[0]
        return cls(
            title=title,
            content=markdown.strip(),
            heading_suffix=heading_suffix,
            point_prefix=point_prefix,
        )


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
    sections: list[PromptBlock] | None = None
    additional_sections: list[PromptBlock] = field(default_factory=list)
    heading_suffix: str = ":"
    point_prefix: PointPrefix = "- "
    separator: str = "\n\n"

    @classmethod
    def from_markdown(
        cls,
        path: str | Path,
        *,
        heading_suffix: str = ":",
        point_prefix: PointPrefix = "- ",
        separator: str = "\n\n",
        encoding: str = "utf-8",
    ) -> AgentPrompt:
        markdown = Path(path).read_text(encoding=encoding)
        return cls.from_markdown_text(
            markdown,
            heading_suffix=heading_suffix,
            point_prefix=point_prefix,
            separator=separator,
        )

    @classmethod
    def from_markdown_text(
        cls,
        markdown: str,
        *,
        heading_suffix: str = ":",
        point_prefix: PointPrefix = "- ",
        separator: str = "\n\n",
    ) -> AgentPrompt:
        return cls(
            sections=_parse_markdown_blocks(
                markdown,
                heading_suffix=heading_suffix,
                point_prefix=point_prefix,
            ),
            heading_suffix=heading_suffix,
            point_prefix=point_prefix,
            separator=separator,
        )

    def render(self) -> str:
        if self.sections is not None:
            return self._render_sections(self.sections)

        sections = [
            self._block(title="Role", content=self.role),
            self._block(title="Objectives", items=self.objectives),
            self._block(title="Context", content=self.context),
            self._block(title="Instructions", items=self.instructions),
            self._block(title="Constraints", items=self.constraints),
            self._block(title="Style", items=self.style),
            self._block(title="Output Requirements", items=self.output_requirements),
            self._block(title="Success Criteria", items=self.success_criteria),
            *self.additional_sections,
        ]

        return self._render_sections(sections)

    def _block(self, title: str, content: str = "", items: list[str] | None = None) -> PromptBlock:
        return PromptBlock(
            title=title,
            content=content,
            items=items or [],
            heading_suffix=self.heading_suffix,
            point_prefix=self.point_prefix,
        )

    def _render_sections(self, sections: list[PromptBlock]) -> str:
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

    @classmethod
    def from_markdown(
        cls,
        path: str | Path,
        *,
        separator: str = "\n\n",
        encoding: str = "utf-8",
    ) -> PromptTemplate:
        markdown = Path(path).read_text(encoding=encoding)
        return cls.from_markdown_text(markdown, separator=separator)

    @classmethod
    def from_markdown_text(
        cls,
        markdown: str,
        *,
        separator: str = "\n\n",
    ) -> PromptTemplate:
        sections = [
            PromptSection(name=_section_name(block.title), content=block.render())
            for block in _parse_markdown_blocks(markdown)
        ]
        return cls(sections=sections, separator=separator)

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


def _parse_markdown_blocks(
    markdown: str,
    *,
    heading_suffix: str = ":",
    point_prefix: PointPrefix = "- ",
) -> list[PromptBlock]:
    blocks: list[PromptBlock] = []
    current_title = ""
    content_lines: list[str] = []
    items: list[str] = []
    current_prefix = point_prefix
    in_code_fence = False

    def flush() -> None:
        nonlocal content_lines, items, current_prefix
        content = _markdown_content(content_lines)
        if content or items:
            blocks.append(
                PromptBlock(
                    title=current_title,
                    content=content,
                    items=items,
                    heading_suffix=heading_suffix,
                    point_prefix=current_prefix,
                )
            )
        content_lines = []
        items = []
        current_prefix = point_prefix

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            content_lines.append(line)
            continue

        if not in_code_fence:
            heading = _HEADING_RE.match(line)
            if heading:
                flush()
                current_title = heading.group(2).strip()
                continue

            parsed_item = _parse_markdown_item(line)
            if parsed_item is not None:
                marker_prefix, item = parsed_item
                if not items:
                    current_prefix = marker_prefix
                items.append(item)
                continue

        content_lines.append(line)

    flush()
    return blocks


def _markdown_content(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def _parse_markdown_item(line: str) -> tuple[PointPrefix, str] | None:
    checkbox = _CHECKBOX_ITEM_RE.match(line)
    if checkbox:
        marker, checked, item = checkbox.groups()
        return f"{marker} ", f"[{checked}] {item.strip()}"

    unordered = _UNORDERED_ITEM_RE.match(line)
    if unordered:
        marker, item = unordered.groups()
        return f"{marker} ", item.strip()

    ordered = _ORDERED_ITEM_RE.match(line)
    if ordered:
        marker, item = ordered.groups()
        return f"{{index}}{marker} ", item.strip()

    return None


def _section_name(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return slug or "prompt"
