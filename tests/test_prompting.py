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
