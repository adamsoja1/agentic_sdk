import pytest

from agentic_framework.core.prompting import AgentPrompt, PromptBlock, PromptSection, PromptTemplate


def test_agent_prompt_renders_structured_sections():
    prompt = AgentPrompt(
        role="You are a support specialist for API integrations.",
        objectives=[
            "Resolve the user's issue accurately.",
            "Keep the answer concise and actionable.",
        ],
        context="You help developers integrate third-party systems.",
        instructions=[
            "Use tools when they materially improve the answer.",
            "Explain tradeoffs when recommending an approach.",
        ],
        constraints=[
            "Do not invent external data.",
            "Ask for clarification when key inputs are missing.",
        ],
        style=[
            "Prefer short paragraphs.",
            "Use calm, direct language.",
        ],
        output_requirements=[
            "End with a recommended next step.",
        ],
        success_criteria=[
            "The user can act on the answer immediately.",
        ],
        additional_sections=[
            PromptBlock(
                title="Domain Notes",
                items=["Prefer SDK-native solutions before custom wrappers."],
            )
        ],
    )

    rendered = prompt.render()

    assert "Role:\nYou are a support specialist for API integrations." in rendered
    assert "Objectives:\n- Resolve the user's issue accurately." in rendered
    assert "Context:\nYou help developers integrate third-party systems." in rendered
    assert "Instructions:\n- Use tools when they materially improve the answer." in rendered
    assert "Constraints:\n- Do not invent external data." in rendered
    assert "Style:\n- Prefer short paragraphs." in rendered
    assert "Output Requirements:\n- End with a recommended next step." in rendered
    assert "Success Criteria:\n- The user can act on the answer immediately." in rendered
    assert "Domain Notes:\n- Prefer SDK-native solutions before custom wrappers." in rendered


def test_agent_prompt_can_render_custom_sections():
    prompt = AgentPrompt(
        sections=[
            PromptBlock(title="Persona", content="You are a release manager."),
            PromptBlock(
                title="Workflow",
                items=[
                    "Check the changelog.",
                    "Summarize risk before recommendation.",
                ],
                point_prefix="{index}. ",
            ),
            PromptBlock(
                title="Do Not",
                items=["Invent release dates."],
                heading_suffix="",
                point_prefix="* ",
            ),
        ]
    )

    assert prompt.render() == (
        "Persona:\n"
        "You are a release manager.\n\n"
        "Workflow:\n"
        "1. Check the changelog.\n"
        "2. Summarize risk before recommendation.\n\n"
        "Do Not\n"
        "* Invent release dates."
    )


def test_agent_prompt_can_customize_default_point_style():
    prompt = AgentPrompt(
        objectives=["Triage the request.", "Recommend a next step."],
        constraints=["Do not invent details."],
        point_prefix="{index}. ",
    )

    rendered = prompt.render()

    assert "Objectives:\n1. Triage the request.\n2. Recommend a next step." in rendered
    assert "Constraints:\n1. Do not invent details." in rendered


def test_prompt_block_can_render_plain_points():
    block = PromptBlock(
        title="Checklist",
        items=["Confirm scope.", "Report result."],
        point_prefix=None,
    )

    assert block.render() == "Checklist:\nConfirm scope.\nReport result."


def test_agent_prompt_can_load_custom_sections_from_markdown_text():
    prompt = AgentPrompt.from_markdown_text(
        """
# Persona
You are a release manager.

## Workflow
1. Check the changelog.
2. Summarize risk before recommendation.

## Guardrails
* Do not invent release dates.
* Name uncertainty clearly.
""".strip()
    )

    assert prompt.render() == (
        "Persona:\n"
        "You are a release manager.\n\n"
        "Workflow:\n"
        "1. Check the changelog.\n"
        "2. Summarize risk before recommendation.\n\n"
        "Guardrails:\n"
        "* Do not invent release dates.\n"
        "* Name uncertainty clearly."
    )


def test_agent_prompt_can_load_custom_sections_from_markdown_file(tmp_path):
    prompt_file = tmp_path / "support_prompt.md"
    prompt_file.write_text(
        """
# Role
You are a support engineer.

# Instructions
- Ask precise follow-up questions.
- Prefer reproducible steps.
""".strip(),
        encoding="utf-8",
    )

    prompt = AgentPrompt.from_markdown(prompt_file)

    assert prompt.render() == (
        "Role:\n"
        "You are a support engineer.\n\n"
        "Instructions:\n"
        "- Ask precise follow-up questions.\n"
        "- Prefer reproducible steps."
    )


def test_agent_prompt_markdown_preserves_paragraph_breaks():
    prompt = AgentPrompt.from_markdown_text(
        """
# Context
First paragraph.

Second paragraph.
""".strip()
    )

    assert prompt.render() == "Context:\nFirst paragraph.\n\nSecond paragraph."


def test_agent_prompt_markdown_skips_heading_only_sections():
    prompt = AgentPrompt.from_markdown_text(
        """
# Prompt
## Role
You are a support engineer.

## Empty
### Instructions
- Ask precise follow-up questions.
""".strip()
    )

    assert prompt.render() == (
        "Role:\n"
        "You are a support engineer.\n\n"
        "Instructions:\n"
        "- Ask precise follow-up questions."
    )


def test_prompt_template_can_load_sections_from_markdown_text():
    template = PromptTemplate.from_markdown_text(
        """
## Tone
Be concise.

## Output Requirements
- Include a recommendation.
- Mention uncertainty.
""".strip()
    )

    assert [section.name for section in template.sections] == ["tone", "output_requirements"]
    assert template.render() == (
        "Tone:\n"
        "Be concise.\n\n"
        "Output Requirements:\n"
        "- Include a recommendation.\n"
        "- Mention uncertainty."
    )


def test_prompt_template_renders_sections_and_skips_disabled_ones():
    template = PromptTemplate(
        sections=[
            PromptSection("base", "You are helpful."),
            PromptSection("hidden", "Do not show.", enabled=False),
            PromptSection("dynamic", lambda owner: f"Agent: {owner}"),
        ]
    )

    assert template.render("assistant") == "You are helpful.\n\nAgent: assistant"


def test_prompt_template_can_insert_replace_and_remove_sections():
    template = PromptTemplate(sections=[PromptSection("base", "Base"), PromptSection("tail", "Tail")])

    template.set_section("middle", "Middle", after="base")
    template.set_section("tail", "Updated tail")

    assert template.render() == "Base\n\nMiddle\n\nUpdated tail"
    assert template.remove_section("middle") is True
    assert template.remove_section("missing") is False
    assert template.render() == "Base\n\nUpdated tail"


def test_prompt_template_locked_sections_cannot_be_replaced_or_removed():
    template = PromptTemplate(sections=[PromptSection("locked_base", "Base", locked=True)])

    with pytest.raises(ValueError, match="locked"):
        template.set_section("locked_base", "Override")

    with pytest.raises(ValueError, match="locked"):
        template.remove_section("locked_base")
